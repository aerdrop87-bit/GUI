from flask import Blueprint, render_template, request, flash, redirect, url_for, send_file
from flask import current_app
import os
from datetime import datetime

import pandas as pd  # NEW: untuk load data visualisasi dari output Excel

from models import db
from models.dataset import Dataset
from models.pipeline_models import PreprocessRun, PCARun, SilhouetteRun, KMeansRun
from models.clustering import run_silhouette, run_kmeans


clustering_bp = Blueprint(
    "clustering",
    __name__,
    url_prefix="/clustering"
)


def _get_last_success_pca_for_dataset(dataset_id: int):
    last_pre = (PreprocessRun.query
                .filter_by(dataset_id=dataset_id, status="success")
                .order_by(PreprocessRun.created_at.desc())
                .first())
    if not last_pre:
        return None, None

    last_pca = (PCARun.query
                .filter_by(preprocess_run_id=last_pre.id, status="success")
                .order_by(PCARun.created_at.desc())
                .first())
    return last_pre, last_pca


@clustering_bp.route("/", methods=["GET", "POST"], strict_slashes=False)
def index():
    datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).all()

    selected_dataset_id = request.args.get("dataset_id", type=int)
    last_preprocess = None
    last_pca = None

    if selected_dataset_id:
        last_preprocess, last_pca = _get_last_success_pca_for_dataset(selected_dataset_id)

    # history (tampilan)
    silhouette_runs = (SilhouetteRun.query
                       .order_by(SilhouetteRun.created_at.desc())
                       .limit(20).all())
    kmeans_runs = (KMeansRun.query
                   .order_by(KMeansRun.created_at.desc())
                   .limit(20).all())

    last_silhouette_for_pca = None
    last_kmeans_for_pca = None

    if last_pca:
        last_silhouette_for_pca = (SilhouetteRun.query
                                   .filter_by(pca_run_id=last_pca.id, status="success")
                                   .order_by(SilhouetteRun.created_at.desc())
                                   .first())
        last_kmeans_for_pca = (KMeansRun.query
                               .filter_by(pca_run_id=last_pca.id, status="success")
                               .order_by(KMeansRun.created_at.desc())
                               .first())

    # NEW: data untuk visualisasi (PCA1, PCA2, Cluster) dari output clustering
    pca_points = None
    if last_kmeans_for_pca and last_kmeans_for_pca.output_path:
        try:
            allowed_dir = os.path.abspath(os.path.join(current_app.config["OUTPUT_FOLDER"], "clustering"))
            file_path = os.path.abspath(last_kmeans_for_pca.output_path)

            if file_path.startswith(allowed_dir) and os.path.exists(file_path):
                df_vis = pd.read_excel(file_path, sheet_name="Data+Cluster")
                if all(c in df_vis.columns for c in ["PCA1", "PCA2", "Cluster"]):
                    pca_points = df_vis[["PCA1", "PCA2", "Cluster"]].to_dict(orient="records")
        except Exception:
            pca_points = None

    if request.method == "POST":
        action = request.form.get("action")
        dataset_id = request.form.get("dataset_id", type=int)

        if not dataset_id:
            flash("Pilih dataset terlebih dahulu.", "error")
            return redirect(url_for("clustering.index"))

        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            flash("Dataset tidak ditemukan!", "error")
            return redirect(url_for("clustering.index"))

        # Ambil PCA terakhir yang sukses
        last_preprocess, last_pca = _get_last_success_pca_for_dataset(dataset.id)

        if not last_pca or not last_pca.output_path:
            flash("Belum ada PCA yang sukses untuk dataset ini. Jalankan PCA dulu.", "error")
            return redirect(url_for("clustering.index", dataset_id=dataset.id))

        # Security: pastikan file PCA ada di outputs/pca
        allowed_pca_dir = os.path.abspath(os.path.join(current_app.config["OUTPUT_FOLDER"], "pca"))
        pca_path = os.path.abspath(last_pca.output_path)

        if not pca_path.startswith(allowed_pca_dir):
            flash("Path file PCA tidak valid. Pastikan PCA dibuat dari sistem ini.", "error")
            return redirect(url_for("clustering.index", dataset_id=dataset.id))

        if not os.path.exists(pca_path):
            flash("File output PCA tidak ditemukan. Jalankan PCA ulang.", "error")
            return redirect(url_for("clustering.index", dataset_id=dataset.id))

        # default komponen dipakai mengikuti PCA (kalau ada), fallback 4
        default_components_count = 4
        if last_pca.selected_components:
            try:
                default_components_count = int(len(last_pca.selected_components))
            except Exception:
                default_components_count = 4

        components_count = request.form.get("components_count", type=int) or default_components_count

        try:
            if action == "silhouette":
                k_min = request.form.get("k_min", type=int) or 2
                k_max = request.form.get("k_max", type=int) or 10
                random_state = request.form.get("random_state", type=int) or 42

                result = run_silhouette(
                    pca_output_path=pca_path,
                    components_count=components_count,
                    k_min=k_min,
                    k_max=k_max,
                    random_state=random_state
                )

                run = SilhouetteRun(
                    pca_run_id=last_pca.id,
                    k_min=k_min,
                    k_max=k_max,
                    scores_by_k=result["scores_by_k"],
                    best_k=result["best_k"],
                    status="success",
                    error_message=None
                )
                db.session.add(run)
                db.session.commit()

                flash(f"Silhouette berhasil! Best K = {result['best_k']}", "success")
                return redirect(url_for("clustering.index", dataset_id=dataset.id))

            elif action == "kmeans":
                k = request.form.get("k", type=int) or 3
                init_method = request.form.get("init_method", "k-means++")
                init_rows_raw = (request.form.get("init_rows") or "").strip()

                n_init = request.form.get("n_init", type=int)
                max_iter = request.form.get("max_iter", type=int) or 300
                random_state = request.form.get("random_state", type=int) or 42

                init_rows = None
                if init_method == "manual":
                    if not init_rows_raw:
                        raise ValueError("Untuk init manual, isi nomor baris centroid awal (contoh: 307,142,235).")
                    init_rows = [int(x.strip()) for x in init_rows_raw.split(",") if x.strip()]
                    # validasi panjang list dilakukan di service

                output_folder = os.path.join(current_app.config["OUTPUT_FOLDER"], "clustering")
                os.makedirs(output_folder, exist_ok=True)
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                output_filename = f"Hasil_KMeans_dataset{dataset.id}_pca{last_pca.id}_k{k}_{ts}.xlsx"

                result = run_kmeans(
                    pca_output_path=pca_path,
                    components_count=components_count,
                    k=k,
                    init_method=init_method,
                    init_rows_excel_1based=init_rows,
                    n_init=n_init,
                    max_iter=max_iter,
                    random_state=random_state,
                    output_folder=output_folder,
                    output_filename=output_filename
                )

                run = KMeansRun(
                    pca_run_id=last_pca.id,
                    k=result["k"],
                    init_method=result["init_method"],
                    init_rows=result["init_rows"],
                    n_init=result["n_init"],
                    max_iter=result["max_iter"],
                    random_state=result["random_state"],
                    centroids_final=result["centroids_final"],
                    counts_by_cluster=result["counts_by_cluster"],
                    output_path=result["output_path"],
                    status="success",
                    error_message=None,

                    # NEW: simpan iterasi untuk ditampilkan di HTML
                    n_iter=result.get("n_iter"),
                    iterations_log=result.get("iterations_log"),
                    labels_by_iteration=result.get("labels_by_iteration"),
                    centroids_by_iteration=result.get("centroids_by_iteration"),
                )
                db.session.add(run)
                db.session.commit()

                flash("KMeans berhasil! Output tersimpan dan tercatat di database.", "success")
                return redirect(url_for("clustering.index", dataset_id=dataset.id))

            else:
                flash("Aksi tidak dikenali.", "error")
                return redirect(url_for("clustering.index", dataset_id=dataset.id))

        except Exception as e:
            db.session.rollback()

            # catat kegagalan bila memungkinkan
            try:
                if action == "silhouette" and last_pca:
                    fail = SilhouetteRun(
                        pca_run_id=last_pca.id,
                        k_min=request.form.get("k_min", type=int) or 2,
                        k_max=request.form.get("k_max", type=int) or 10,
                        scores_by_k=[{"k": None, "score": None}],
                        best_k=None,
                        status="failed",
                        error_message=str(e)
                    )
                    db.session.add(fail)
                    db.session.commit()

                if action == "kmeans" and last_pca:
                    fail = KMeansRun(
                        pca_run_id=last_pca.id,
                        k=request.form.get("k", type=int) or 0,
                        init_method=request.form.get("init_method", "k-means++"),
                        init_rows=None,
                        n_init=request.form.get("n_init", type=int) or 0,
                        max_iter=request.form.get("max_iter", type=int) or 0,
                        random_state=request.form.get("random_state", type=int) or 0,
                        centroids_final=None,
                        counts_by_cluster=None,
                        output_path=None,
                        status="failed",
                        error_message=str(e)
                    )
                    db.session.add(fail)
                    db.session.commit()
            except Exception:
                db.session.rollback()

            flash(f"Proses clustering gagal: {e}", "error")
            return redirect(url_for("clustering.index", dataset_id=dataset.id))

    return render_template(
        "pages/clustering.html",
        datasets=datasets,
        selected_dataset_id=selected_dataset_id,
        last_preprocess=last_preprocess,
        last_pca=last_pca,
        last_silhouette_for_pca=last_silhouette_for_pca,
        last_kmeans_for_pca=last_kmeans_for_pca,
        silhouette_runs=silhouette_runs,
        kmeans_runs=kmeans_runs,
        pca_points=pca_points  # NEW
    )


@clustering_bp.route("/download/<int:run_id>", methods=["GET"])
def download(run_id: int):
    run = KMeansRun.query.get_or_404(run_id)

    if not run.output_path:
        flash("Output clustering belum tersedia.", "error")
        return redirect(url_for("clustering.index"))

    allowed_dir = os.path.abspath(os.path.join(current_app.config["OUTPUT_FOLDER"], "clustering"))
    file_path = os.path.abspath(run.output_path)

    if not file_path.startswith(allowed_dir):
        flash("Path file tidak valid.", "error")
        return redirect(url_for("clustering.index"))

    if not os.path.exists(file_path):
        flash("File output tidak ditemukan di server.", "error")
        return redirect(url_for("clustering.index"))

    return send_file(file_path, as_attachment=True)
