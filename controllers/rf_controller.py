from __future__ import annotations

from flask import Blueprint, render_template, request, flash, redirect, url_for, send_file, current_app
import os
import json
from datetime import datetime

from models import db
from models.dataset import Dataset
from models.pipeline_models import PreprocessRun, PCARun, KMeansRun, DataSplit, RFRuleSet, RFEvaluation
from models.random_forest import run_rf_manual_evaluation, DEFAULT_AMBANG, validate_ambang


rf_bp = Blueprint(
    "rf",
    __name__,
    url_prefix="/rf"
)


def _safe_within_outputs(path: str, outputs_root: str) -> bool:
    allowed = os.path.abspath(outputs_root)
    target = os.path.abspath(path)
    return target.startswith(allowed)


def _get_latest_kmeans_for_dataset(dataset_id: int):
    # Join: KMeansRun -> PCARun -> PreprocessRun (filter dataset_id)

    # ============================
    # sebelum
    # ============================
    # q = (KMeansRun.query
    #      .join(PCARun, KMeansRun.pca_run_id == PCARun.id)
    #      .join(PreprocessRun, PCARun.preprocess_run_id == PreprocessRun.id)
    #      .filter(PreprocessRun.dataset_id == dataset_id, KMeansRun.status == "success")
    #      .order_by(KMeansRun.created_at.desc()))

    # ============================
    # sesudah
    # ============================
    # RF manual rules saat ini hanya mendukung 3 kelas (C1,C2,C3),
    # jadi kita ambil KMeans sukses terakhir khusus K=3.
    q = (KMeansRun.query
         .join(PCARun, KMeansRun.pca_run_id == PCARun.id)
         .join(PreprocessRun, PCARun.preprocess_run_id == PreprocessRun.id)
         .filter(
            PreprocessRun.dataset_id == dataset_id,
            KMeansRun.status == "success",
            KMeansRun.k == 3,
         )
         .order_by(KMeansRun.created_at.desc()))

    return q.first()


def _get_or_create_default_ruleset() -> RFRuleSet:
    existing = (RFRuleSet.query
                .filter_by(name="RF Manual Ambang PCA", version="v1")
                .order_by(RFRuleSet.created_at.desc())
                .first())
    if existing:
        return existing

    ruleset = RFRuleSet(
        name="RF Manual Ambang PCA",
        version="v1",
        ambang=DEFAULT_AMBANG
    )
    db.session.add(ruleset)
    db.session.commit()
    return ruleset


@rf_bp.route("/", methods=["GET", "POST"], strict_slashes=False)
def index():
    datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).all()
    selected_dataset_id = request.args.get("dataset_id", type=int)

    # default ruleset (akan dibuat jika belum ada)
    ruleset = _get_or_create_default_ruleset()

    # tampilkan latest kmeans & evaluasi untuk dataset terpilih
    last_kmeans = _get_latest_kmeans_for_dataset(selected_dataset_id) if selected_dataset_id else None

    last_split = None
    last_eval_train = None
    last_eval_test = None
    if last_kmeans:
        last_split = (DataSplit.query
                      .filter_by(kmeans_run_id=last_kmeans.id)
                      .order_by(DataSplit.created_at.desc())
                      .first())

        last_eval_train = (RFEvaluation.query
                           .filter_by(kmeans_run_id=last_kmeans.id, split_type="train")
                           .order_by(RFEvaluation.created_at.desc())
                           .first())
        last_eval_test = (RFEvaluation.query
                          .filter_by(kmeans_run_id=last_kmeans.id, split_type="test")
                          .order_by(RFEvaluation.created_at.desc())
                          .first())

    # riwayat evaluasi terakhir
    evals = (RFEvaluation.query
             .order_by(RFEvaluation.created_at.desc())
             .limit(30)
             .all())

    if request.method == "POST":
        dataset_id = request.form.get("dataset_id", type=int)
        if not dataset_id:
            flash("Pilih dataset terlebih dahulu.", "error")
            return redirect(url_for("rf.index"))

        last_kmeans = _get_latest_kmeans_for_dataset(dataset_id)
        if not last_kmeans or not last_kmeans.output_path:
            flash("Belum ada hasil KMeans (K=3) untuk dataset ini. Jalankan clustering dulu (K=3) dan pastikan PCA1..PCA4 tersedia.", "error")
            return redirect(url_for("rf.index", dataset_id=dataset_id))

        if not os.path.exists(last_kmeans.output_path):
            flash("File output KMeans tidak ditemukan di server.", "error")
            return redirect(url_for("rf.index", dataset_id=dataset_id))

        # ============================
        # # tambahan
        # ============================
        # Guard tambahan bila suatu saat _get_latest_kmeans_for_dataset diubah lagi
        # atau data lama terselip: RF manual rules saat ini hanya untuk 3 kelas.
        if int(getattr(last_kmeans, "k", 0) or 0) != 3:
            flash("RF manual saat ini hanya mendukung 3 cluster (C1,C2,C3). Jalankan KMeans dengan K=3.", "error")
            return redirect(url_for("rf.index", dataset_id=dataset_id))

        # ambang: default atau custom JSON
        ambang_mode = request.form.get("ambang_mode", "default")
        ambang = ruleset.ambang
        used_ruleset = ruleset

        if ambang_mode == "custom":
            txt = (request.form.get("ambang_json") or "").strip()
            if not txt:
                flash("Ambang custom dipilih, tapi JSON ambang kosong.", "error")
                return redirect(url_for("rf.index", dataset_id=dataset_id))
            try:
                ambang = validate_ambang(json.loads(txt))
                custom = RFRuleSet(
                    name="RF Manual Ambang PCA (Custom)",
                    version=datetime.utcnow().strftime("custom_%Y%m%d_%H%M%S"),
                    ambang=ambang
                )
                db.session.add(custom)
                db.session.commit()
                used_ruleset = custom
            except Exception as e:
                db.session.rollback()
                flash(f"JSON ambang tidak valid: {e}", "error")
                return redirect(url_for("rf.index", dataset_id=dataset_id))

        # parameter split + alignment
        train_ratio = request.form.get("train_ratio", type=float) or 0.8
        random_state = request.form.get("random_state", type=int) or 42
        split_method = request.form.get("split_method", "colab_like")  # colab_like/stratified
        alignment_mode = request.form.get("alignment_mode", "auto")    # auto/none

        output_folder = os.path.join(current_app.config["OUTPUT_FOLDER"], "rf")
        os.makedirs(output_folder, exist_ok=True)

        try:
            result = run_rf_manual_evaluation(
                kmeans_output_path=last_kmeans.output_path,
                output_folder=output_folder,
                ambang=ambang,
                train_ratio=train_ratio,
                random_state=random_state,
                labels=["C1", "C2", "C3"],
                split_method=split_method,
                alignment_mode=alignment_mode
            )

            # catat split
            split = DataSplit(
                kmeans_run_id=last_kmeans.id,
                train_path=result["train_path"],
                test_path=result["test_path"],
                split_method=split_method,
                train_ratio=float(train_ratio),
                random_state=int(random_state)
            )
            db.session.add(split)
            db.session.flush()

            # catat evaluasi train
            eval_train = RFEvaluation(
                kmeans_run_id=last_kmeans.id,
                ruleset_id=used_ruleset.id,
                split_type="train",
                input_path=result["train_path"],
                classification_report=result["train"]["report"],
                accuracy=result["train"]["accuracy"],
                macro_f1=result["train"]["macro_f1"],
                output_path=result["train"]["output_path"],
                status="success",
                error_message=None
            )
            db.session.add(eval_train)

            # catat evaluasi test
            eval_test = RFEvaluation(
                kmeans_run_id=last_kmeans.id,
                ruleset_id=used_ruleset.id,
                split_type="test",
                input_path=result["test_path"],
                classification_report=result["test"]["report"],
                accuracy=result["test"]["accuracy"],
                macro_f1=result["test"]["macro_f1"],
                output_path=result["test"]["output_path"],
                status="success",
                error_message=None
            )
            db.session.add(eval_test)

            db.session.commit()

            # info alignment (biar user paham kenapa hasil bisa beda)
            align_note = result.get("alignment", {}).get("note", "")
            if align_note:
                flash("RF + evaluasi berhasil. " + align_note, "success")
            else:
                flash("Random Forest manual + evaluasi (train & test) berhasil!", "success")

            return redirect(url_for("rf.index", dataset_id=dataset_id))

        except Exception as e:
            db.session.rollback()
            flash(f"RF manual gagal: {e}", "error")
            return redirect(url_for("rf.index", dataset_id=dataset_id))

    return render_template(
        "pages/rf.html",
        datasets=datasets,
        selected_dataset_id=selected_dataset_id,
        last_kmeans=last_kmeans,
        last_split=last_split,
        ruleset=ruleset,
        last_eval_train=last_eval_train,
        last_eval_test=last_eval_test,
        evals=evals
    )


@rf_bp.route("/download/eval/<int:eval_id>", methods=["GET"])
def download_eval(eval_id: int):
    ev = RFEvaluation.query.get_or_404(eval_id)
    if not ev.output_path:
        flash("File output evaluasi belum tersedia.", "error")
        return redirect(url_for("rf.index"))

    outputs_root = current_app.config["OUTPUT_FOLDER"]
    if not _safe_within_outputs(ev.output_path, outputs_root):
        flash("Path file tidak valid.", "error")
        return redirect(url_for("rf.index"))

    if not os.path.exists(ev.output_path):
        flash("File output evaluasi tidak ditemukan di server.", "error")
        return redirect(url_for("rf.index"))

    return send_file(ev.output_path, as_attachment=True)


@rf_bp.route("/download/split/<int:split_id>/<string:which>", methods=["GET"])
def download_split(split_id: int, which: str):
    split = DataSplit.query.get_or_404(split_id)
    path = split.train_path if which == "train" else split.test_path
    if not path:
        flash("File split belum tersedia.", "error")
        return redirect(url_for("rf.index"))

    outputs_root = current_app.config["OUTPUT_FOLDER"]
    if not _safe_within_outputs(path, outputs_root):
        flash("Path file tidak valid.", "error")
        return redirect(url_for("rf.index"))

    if not os.path.exists(path):
        flash("File split tidak ditemukan di server.", "error")
        return redirect(url_for("rf.index"))

    return send_file(path, as_attachment=True)


# =========================
# BATAS
# ======================
# from __future__ import annotations

# from flask import Blueprint, render_template, request, flash, redirect, url_for, send_file, current_app
# import os
# import json
# from datetime import datetime

# from models import db
# from models.dataset import Dataset
# from models.pipeline_models import PreprocessRun, PCARun, KMeansRun, DataSplit, RFRuleSet, RFEvaluation
# from models.random_forest import run_rf_manual_evaluation, DEFAULT_AMBANG, validate_ambang


# rf_bp = Blueprint(
#     "rf",
#     __name__,
#     url_prefix="/rf"
# )


# def _safe_within_outputs(path: str, outputs_root: str) -> bool:
#     allowed = os.path.abspath(outputs_root)
#     target = os.path.abspath(path)
#     return target.startswith(allowed)


# def _get_latest_kmeans_for_dataset(dataset_id: int):
#     # Join: KMeansRun -> PCARun -> PreprocessRun (filter dataset_id)
#     q = (KMeansRun.query
#          .join(PCARun, KMeansRun.pca_run_id == PCARun.id)
#          .join(PreprocessRun, PCARun.preprocess_run_id == PreprocessRun.id)
#          .filter(PreprocessRun.dataset_id == dataset_id, KMeansRun.status == "success")
#          .order_by(KMeansRun.created_at.desc()))
#     return q.first()


# def _get_or_create_default_ruleset() -> RFRuleSet:
#     existing = (RFRuleSet.query
#                 .filter_by(name="RF Manual Ambang PCA", version="v1")
#                 .order_by(RFRuleSet.created_at.desc())
#                 .first())
#     if existing:
#         return existing

#     ruleset = RFRuleSet(
#         name="RF Manual Ambang PCA",
#         version="v1",
#         ambang=DEFAULT_AMBANG
#     )
#     db.session.add(ruleset)
#     db.session.commit()
#     return ruleset


# @rf_bp.route("/", methods=["GET", "POST"], strict_slashes=False)
# def index():
#     datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).all()
#     selected_dataset_id = request.args.get("dataset_id", type=int)

#     # default ruleset (akan dibuat jika belum ada)
#     ruleset = _get_or_create_default_ruleset()

#     # tampilkan latest kmeans & evaluasi untuk dataset terpilih
#     last_kmeans = _get_latest_kmeans_for_dataset(selected_dataset_id) if selected_dataset_id else None

#     last_split = None
#     last_eval_train = None
#     last_eval_test = None
#     if last_kmeans:
#         last_split = (DataSplit.query
#                       .filter_by(kmeans_run_id=last_kmeans.id)
#                       .order_by(DataSplit.created_at.desc())
#                       .first())

#         last_eval_train = (RFEvaluation.query
#                            .filter_by(kmeans_run_id=last_kmeans.id, split_type="train")
#                            .order_by(RFEvaluation.created_at.desc())
#                            .first())
#         last_eval_test = (RFEvaluation.query
#                           .filter_by(kmeans_run_id=last_kmeans.id, split_type="test")
#                           .order_by(RFEvaluation.created_at.desc())
#                           .first())

#     # riwayat evaluasi terakhir
#     evals = (RFEvaluation.query
#              .order_by(RFEvaluation.created_at.desc())
#              .limit(30)
#              .all())

#     if request.method == "POST":
#         dataset_id = request.form.get("dataset_id", type=int)
#         if not dataset_id:
#             flash("Pilih dataset terlebih dahulu.", "error")
#             return redirect(url_for("rf.index"))

#         last_kmeans = _get_latest_kmeans_for_dataset(dataset_id)
#         if not last_kmeans or not last_kmeans.output_path:
#             flash("Belum ada hasil KMeans untuk dataset ini. Jalankan clustering dulu.", "error")
#             return redirect(url_for("rf.index", dataset_id=dataset_id))

#         if not os.path.exists(last_kmeans.output_path):
#             flash("File output KMeans tidak ditemukan di server.", "error")
#             return redirect(url_for("rf.index", dataset_id=dataset_id))

#         # ambang: default atau custom JSON
#         ambang_mode = request.form.get("ambang_mode", "default")
#         ambang = ruleset.ambang
#         used_ruleset = ruleset

#         if ambang_mode == "custom":
#             txt = (request.form.get("ambang_json") or "").strip()
#             if not txt:
#                 flash("Ambang custom dipilih, tapi JSON ambang kosong.", "error")
#                 return redirect(url_for("rf.index", dataset_id=dataset_id))
#             try:
#                 ambang = validate_ambang(json.loads(txt))
#                 custom = RFRuleSet(
#                     name="RF Manual Ambang PCA (Custom)",
#                     version=datetime.utcnow().strftime("custom_%Y%m%d_%H%M%S"),
#                     ambang=ambang
#                 )
#                 db.session.add(custom)
#                 db.session.commit()
#                 used_ruleset = custom
#             except Exception as e:
#                 db.session.rollback()
#                 flash(f"JSON ambang tidak valid: {e}", "error")
#                 return redirect(url_for("rf.index", dataset_id=dataset_id))

#         # parameter split + alignment
#         train_ratio = request.form.get("train_ratio", type=float) or 0.8
#         random_state = request.form.get("random_state", type=int) or 42
#         split_method = request.form.get("split_method", "colab_like")  # colab_like/stratified
#         alignment_mode = request.form.get("alignment_mode", "auto")    # auto/none

#         output_folder = os.path.join(current_app.config["OUTPUT_FOLDER"], "rf")
#         os.makedirs(output_folder, exist_ok=True)

#         try:
#             result = run_rf_manual_evaluation(
#                 kmeans_output_path=last_kmeans.output_path,
#                 output_folder=output_folder,
#                 ambang=ambang,
#                 train_ratio=train_ratio,
#                 random_state=random_state,
#                 labels=["C1", "C2", "C3"],
#                 split_method=split_method,
#                 alignment_mode=alignment_mode
#             )

#             # catat split
#             split = DataSplit(
#                 kmeans_run_id=last_kmeans.id,
#                 train_path=result["train_path"],
#                 test_path=result["test_path"],
#                 split_method=split_method,
#                 train_ratio=float(train_ratio),
#                 random_state=int(random_state)
#             )
#             db.session.add(split)
#             db.session.flush()

#             # catat evaluasi train
#             eval_train = RFEvaluation(
#                 kmeans_run_id=last_kmeans.id,
#                 ruleset_id=used_ruleset.id,
#                 split_type="train",
#                 input_path=result["train_path"],
#                 classification_report=result["train"]["report"],
#                 accuracy=result["train"]["accuracy"],
#                 macro_f1=result["train"]["macro_f1"],
#                 output_path=result["train"]["output_path"],
#                 status="success",
#                 error_message=None
#             )
#             db.session.add(eval_train)

#             # catat evaluasi test
#             eval_test = RFEvaluation(
#                 kmeans_run_id=last_kmeans.id,
#                 ruleset_id=used_ruleset.id,
#                 split_type="test",
#                 input_path=result["test_path"],
#                 classification_report=result["test"]["report"],
#                 accuracy=result["test"]["accuracy"],
#                 macro_f1=result["test"]["macro_f1"],
#                 output_path=result["test"]["output_path"],
#                 status="success",
#                 error_message=None
#             )
#             db.session.add(eval_test)

#             db.session.commit()

#             # info alignment (biar user paham kenapa hasil bisa beda)
#             align_note = result.get("alignment", {}).get("note", "")
#             if align_note:
#                 flash("RF + evaluasi berhasil. " + align_note, "success")
#             else:
#                 flash("Random Forest manual + evaluasi (train & test) berhasil!", "success")

#             return redirect(url_for("rf.index", dataset_id=dataset_id))

#         except Exception as e:
#             db.session.rollback()
#             flash(f"RF manual gagal: {e}", "error")
#             return redirect(url_for("rf.index", dataset_id=dataset_id))

#     return render_template(
#         "pages/rf.html",
#         datasets=datasets,
#         selected_dataset_id=selected_dataset_id,
#         last_kmeans=last_kmeans,
#         last_split=last_split,
#         ruleset=ruleset,
#         last_eval_train=last_eval_train,
#         last_eval_test=last_eval_test,
#         evals=evals
#     )


# @rf_bp.route("/download/eval/<int:eval_id>", methods=["GET"])
# def download_eval(eval_id: int):
#     ev = RFEvaluation.query.get_or_404(eval_id)
#     if not ev.output_path:
#         flash("File output evaluasi belum tersedia.", "error")
#         return redirect(url_for("rf.index"))

#     outputs_root = current_app.config["OUTPUT_FOLDER"]
#     if not _safe_within_outputs(ev.output_path, outputs_root):
#         flash("Path file tidak valid.", "error")
#         return redirect(url_for("rf.index"))

#     if not os.path.exists(ev.output_path):
#         flash("File output evaluasi tidak ditemukan di server.", "error")
#         return redirect(url_for("rf.index"))

#     return send_file(ev.output_path, as_attachment=True)


# @rf_bp.route("/download/split/<int:split_id>/<string:which>", methods=["GET"])
# def download_split(split_id: int, which: str):
#     split = DataSplit.query.get_or_404(split_id)
#     path = split.train_path if which == "train" else split.test_path
#     if not path:
#         flash("File split belum tersedia.", "error")
#         return redirect(url_for("rf.index"))

#     outputs_root = current_app.config["OUTPUT_FOLDER"]
#     if not _safe_within_outputs(path, outputs_root):
#         flash("Path file tidak valid.", "error")
#         return redirect(url_for("rf.index"))

#     if not os.path.exists(path):
#         flash("File split tidak ditemukan di server.", "error")
#         return redirect(url_for("rf.index"))

#     return send_file(path, as_attachment=True)
# =========================
# BATAS
# ======================
# from __future__ import annotations

# from flask import Blueprint, render_template, request, flash, redirect, url_for, send_file, current_app
# import os
# import json
# from datetime import datetime

# from models import db
# from models.dataset import Dataset
# from models.pipeline_models import PreprocessRun, PCARun, KMeansRun, DataSplit, RFRuleSet, RFEvaluation
# from models.random_forest import run_rf_manual_evaluation, DEFAULT_AMBANG, validate_ambang


# rf_bp = Blueprint(
#     "rf",
#     __name__,
#     url_prefix="/rf"
# )


# def _safe_within_outputs(path: str, outputs_root: str) -> bool:
#     allowed = os.path.abspath(outputs_root)
#     target = os.path.abspath(path)
#     return target.startswith(allowed)


# def _get_latest_kmeans_for_dataset(dataset_id: int):
#     # Join: KMeansRun -> PCARun -> PreprocessRun (filter dataset_id)
#     q = (KMeansRun.query
#          .join(PCARun, KMeansRun.pca_run_id == PCARun.id)
#          .join(PreprocessRun, PCARun.preprocess_run_id == PreprocessRun.id)
#          .filter(PreprocessRun.dataset_id == dataset_id, KMeansRun.status == "success")
#          .order_by(KMeansRun.created_at.desc()))
#     return q.first()


# def _get_or_create_default_ruleset() -> RFRuleSet:
#     existing = (RFRuleSet.query
#                 .filter_by(name="RF Manual Ambang PCA", version="v1")
#                 .order_by(RFRuleSet.created_at.desc())
#                 .first())
#     if existing:
#         return existing

#     ruleset = RFRuleSet(
#         name="RF Manual Ambang PCA",
#         version="v1",
#         ambang=DEFAULT_AMBANG
#     )
#     db.session.add(ruleset)
#     db.session.commit()
#     return ruleset


# @rf_bp.route("/", methods=["GET", "POST"], strict_slashes=False)
# def index():
#     datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).all()
#     selected_dataset_id = request.args.get("dataset_id", type=int)

#     # default ruleset (akan dibuat jika belum ada)
#     ruleset = _get_or_create_default_ruleset()

#     # tampilkan latest kmeans & evaluasi untuk dataset terpilih
#     last_kmeans = _get_latest_kmeans_for_dataset(selected_dataset_id) if selected_dataset_id else None

#     last_split = None
#     last_eval_train = None
#     last_eval_test = None
#     if last_kmeans:
#         last_split = (DataSplit.query
#                       .filter_by(kmeans_run_id=last_kmeans.id)
#                       .order_by(DataSplit.created_at.desc())
#                       .first())

#         last_eval_train = (RFEvaluation.query
#                            .filter_by(kmeans_run_id=last_kmeans.id, ruleset_id=ruleset.id, split_type="train")
#                            .order_by(RFEvaluation.created_at.desc())
#                            .first())
#         last_eval_test = (RFEvaluation.query
#                           .filter_by(kmeans_run_id=last_kmeans.id, ruleset_id=ruleset.id, split_type="test")
#                           .order_by(RFEvaluation.created_at.desc())
#                           .first())

#     # riwayat evaluasi terakhir
#     evals = (RFEvaluation.query
#              .order_by(RFEvaluation.created_at.desc())
#              .limit(30)
#              .all())

#     if request.method == "POST":
#         dataset_id = request.form.get("dataset_id", type=int)
#         if not dataset_id:
#             flash("Pilih dataset terlebih dahulu.", "error")
#             return redirect(url_for("rf.index"))

#         last_kmeans = _get_latest_kmeans_for_dataset(dataset_id)
#         if not last_kmeans or not last_kmeans.output_path:
#             flash("Belum ada hasil KMeans untuk dataset ini. Jalankan clustering dulu.", "error")
#             return redirect(url_for("rf.index", dataset_id=dataset_id))

#         if not os.path.exists(last_kmeans.output_path):
#             flash("File output KMeans tidak ditemukan di server.", "error")
#             return redirect(url_for("rf.index", dataset_id=dataset_id))

#         # ambang: default atau custom JSON
#         ambang_mode = request.form.get("ambang_mode", "default")
#         ambang = ruleset.ambang
#         if ambang_mode == "custom":
#             txt = (request.form.get("ambang_json") or "").strip()
#             if not txt:
#                 flash("Ambang custom dipilih, tapi JSON ambang kosong.", "error")
#                 return redirect(url_for("rf.index", dataset_id=dataset_id))
#             try:
#                 ambang = validate_ambang(json.loads(txt))
#                 # simpan sebagai ruleset baru (version increment sederhana)
#                 custom = RFRuleSet(
#                     name="RF Manual Ambang PCA (Custom)",
#                     version=datetime.utcnow().strftime("custom_%Y%m%d_%H%M%S"),
#                     ambang=ambang
#                 )
#                 db.session.add(custom)
#                 db.session.commit()
#                 ruleset = custom
#             except Exception as e:
#                 db.session.rollback()
#                 flash(f"JSON ambang tidak valid: {e}", "error")
#                 return redirect(url_for("rf.index", dataset_id=dataset_id))

#         # parameter split
#         train_ratio = request.form.get("train_ratio", type=float) or 0.7
#         random_state = request.form.get("random_state", type=int) or 42

#         output_folder = os.path.join(current_app.config["OUTPUT_FOLDER"], "rf")
#         os.makedirs(output_folder, exist_ok=True)

#         try:
#             result = run_rf_manual_evaluation(
#                 kmeans_output_path=last_kmeans.output_path,
#                 output_folder=output_folder,
#                 ambang=ambang,
#                 train_ratio=train_ratio,
#                 random_state=random_state,
#                 labels=["C1", "C2", "C3"]
#             )

#             # catat split
#             split = DataSplit(
#                 kmeans_run_id=last_kmeans.id,
#                 train_path=result["train_path"],
#                 test_path=result["test_path"],
#                 split_method="random_stratified",
#                 train_ratio=float(train_ratio),
#                 random_state=int(random_state)
#             )
#             db.session.add(split)
#             db.session.flush()  # dapatkan id jika perlu

#             # catat evaluasi train
#             eval_train = RFEvaluation(
#                 kmeans_run_id=last_kmeans.id,
#                 ruleset_id=ruleset.id,
#                 split_type="train",
#                 input_path=result["train_path"],
#                 classification_report=result["train"]["report"],
#                 accuracy=result["train"]["accuracy"],
#                 macro_f1=result["train"]["macro_f1"],
#                 output_path=result["train"]["output_path"],
#                 status="success",
#                 error_message=None
#             )
#             db.session.add(eval_train)

#             # catat evaluasi test
#             eval_test = RFEvaluation(
#                 kmeans_run_id=last_kmeans.id,
#                 ruleset_id=ruleset.id,
#                 split_type="test",
#                 input_path=result["test_path"],
#                 classification_report=result["test"]["report"],
#                 accuracy=result["test"]["accuracy"],
#                 macro_f1=result["test"]["macro_f1"],
#                 output_path=result["test"]["output_path"],
#                 status="success",
#                 error_message=None
#             )
#             db.session.add(eval_test)

#             db.session.commit()

#             flash("Random Forest manual + evaluasi (train & test) berhasil!", "success")
#             return redirect(url_for("rf.index", dataset_id=dataset_id))

#         except Exception as e:
#             db.session.rollback()
#             flash(f"RF manual gagal: {e}", "error")
#             return redirect(url_for("rf.index", dataset_id=dataset_id))

#     return render_template(
#         "pages/rf.html",
#         datasets=datasets,
#         selected_dataset_id=selected_dataset_id,
#         last_kmeans=last_kmeans,
#         last_split=last_split,
#         ruleset=ruleset,
#         last_eval_train=last_eval_train,
#         last_eval_test=last_eval_test,
#         evals=evals
#     )


# @rf_bp.route("/download/eval/<int:eval_id>", methods=["GET"])
# def download_eval(eval_id: int):
#     ev = RFEvaluation.query.get_or_404(eval_id)
#     if not ev.output_path:
#         flash("File output evaluasi belum tersedia.", "error")
#         return redirect(url_for("rf.index"))

#     outputs_root = current_app.config["OUTPUT_FOLDER"]
#     if not _safe_within_outputs(ev.output_path, outputs_root):
#         flash("Path file tidak valid.", "error")
#         return redirect(url_for("rf.index"))

#     if not os.path.exists(ev.output_path):
#         flash("File output evaluasi tidak ditemukan di server.", "error")
#         return redirect(url_for("rf.index"))

#     return send_file(ev.output_path, as_attachment=True)


# @rf_bp.route("/download/split/<int:split_id>/<string:which>", methods=["GET"])
# def download_split(split_id: int, which: str):
#     split = DataSplit.query.get_or_404(split_id)
#     path = split.train_path if which == "train" else split.test_path
#     if not path:
#         flash("File split belum tersedia.", "error")
#         return redirect(url_for("rf.index"))

#     outputs_root = current_app.config["OUTPUT_FOLDER"]
#     if not _safe_within_outputs(path, outputs_root):
#         flash("Path file tidak valid.", "error")
#         return redirect(url_for("rf.index"))

#     if not os.path.exists(path):
#         flash("File split tidak ditemukan di server.", "error")
#         return redirect(url_for("rf.index"))

#     return send_file(path, as_attachment=True)
