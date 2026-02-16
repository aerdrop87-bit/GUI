from flask import Blueprint, render_template, request, flash, redirect, url_for, send_file
from flask import current_app
import os
from datetime import datetime

from models import db
from models.dataset import Dataset
from models.preprocessing import run_preprocessing
from models.pipeline_models import PreprocessRun


preprocessing_bp = Blueprint(
    "preprocessing",
    __name__,
    url_prefix="/preprocessing"
)


# ==========================================
# HALAMAN PREPROCESSING
# ==========================================
@preprocessing_bp.route("/", methods=["GET", "POST"])
def index():

    datasets = Dataset.query.order_by(
        Dataset.uploaded_at.desc()
    ).all()

    preprocess_runs = PreprocessRun.query.order_by(
        PreprocessRun.created_at.desc()
    ).limit(20).all()

    if request.method == "POST":

        dataset_id = request.form.get("dataset_id", type=int)
        dataset = Dataset.query.get(dataset_id)

        if not dataset:
            flash("Dataset tidak ditemukan!", "error")
            return redirect(request.url)

        input_path = dataset.filepath

        output_folder = os.path.join(
            current_app.config["OUTPUT_FOLDER"],
            "preprocessing"
        )

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_filename = f"Data_Standarisasi_dataset{dataset.id}_{ts}.xlsx"

        # === PARAM dari UI (default: match Colab) ===
        mode = (request.form.get("mode") or "colab").strip()              # colab / robust
        imputation_strategy = (request.form.get("imputation_strategy") or "none").strip()  # none/median/mean
        debug_sheets = (request.form.get("debug_sheets") == "on")

        try:
            result = run_preprocessing(
                input_path=input_path,
                output_folder=output_folder,
                output_filename=output_filename,
                imputation_strategy=imputation_strategy,
                mode=mode,
                debug_sheets=debug_sheets
            )

            run = PreprocessRun(
                dataset_id=dataset.id,
                method="standard_scaler",
                feature_columns=result["feature_columns"],
                imputation_strategy=result["imputation_strategy"],
                scaler_params=result.get("scaler_params"),
                output_path=result["output_path"],
                status="success",
                error_message=None
            )
            db.session.add(run)

            if hasattr(dataset, "n_rows"):
                dataset.n_rows = result.get("n_rows")
            if hasattr(dataset, "n_cols"):
                dataset.n_cols = result.get("n_cols")

            db.session.commit()
            flash("Preprocessing berhasil! Output tersimpan dan tercatat di database.", "success")

        except Exception as e:
            db.session.rollback()

            try:
                fail_run = PreprocessRun(
                    dataset_id=dataset.id,
                    method="standard_scaler",
                    feature_columns=[
                        'goltarif',
                        'wilayah',
                        'lama berlangganan',
                        'mawal',
                        'makhir',
                        'pemakaian',
                        'minpemakaianair',
                        'slip tagihan',
                        'jumlah tagihan (Rp.)'
                    ],
                    imputation_strategy=imputation_strategy,
                    scaler_params=None,
                    output_path=None,
                    status="failed",
                    error_message=str(e)
                )
                db.session.add(fail_run)
                db.session.commit()
            except Exception:
                db.session.rollback()

            flash(f"Preprocessing gagal: {e}", "error")

        return redirect(url_for("preprocessing.index"))

    return render_template(
        "pages/preprocessing.html",
        datasets=datasets,
        preprocess_runs=preprocess_runs
    )


# ==========================================
# DOWNLOAD OUTPUT PREPROCESSING
# ==========================================
@preprocessing_bp.route("/download/<int:run_id>", methods=["GET"])
def download(run_id: int):
    run = PreprocessRun.query.get_or_404(run_id)

    if not run.output_path:
        flash("Output preprocessing belum tersedia.", "error")
        return redirect(url_for("preprocessing.index"))

    allowed_dir = os.path.abspath(os.path.join(
        current_app.config["OUTPUT_FOLDER"], "preprocessing"
    ))
    file_path = os.path.abspath(run.output_path)

    if not file_path.startswith(allowed_dir):
        flash("Path file tidak valid.", "error")
        return redirect(url_for("preprocessing.index"))

    if not os.path.exists(file_path):
        flash("File output tidak ditemukan di server.", "error")
        return redirect(url_for("preprocessing.index"))

    return send_file(file_path, as_attachment=True)
