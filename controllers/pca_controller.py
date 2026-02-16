from flask import Blueprint, render_template, request, flash, redirect, url_for, send_file
from flask import current_app
import os
from datetime import datetime

from models import db
from models.dataset import Dataset
from models.pipeline_models import PreprocessRun, PCARun
from models.pca import run_pca


pca_bp = Blueprint(
    "pca",
    __name__,
    url_prefix="/pca"
)


@pca_bp.route("/", methods=["GET", "POST"], strict_slashes=False)
def index():
    datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).all()

    # untuk tampilan: history PCA terbaru
    pca_runs = PCARun.query.order_by(PCARun.created_at.desc()).limit(20).all()

    # dataset yang sedang dipilih (opsional)
    selected_dataset_id = request.args.get("dataset_id", type=int)

    if request.method == "POST":
        dataset_id = request.form.get("dataset_id", type=int)
        method = request.form.get("method", "correlation")
        selected_k = request.form.get("selected_components_count", type=int) or 4

        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            flash("Dataset tidak ditemukan!", "error")
            return redirect(request.url)

        # Ambil PreprocessRun terbaru yang sukses untuk dataset ini
        preprocess_run = (PreprocessRun.query
                          .filter_by(dataset_id=dataset.id, status="success")
                          .order_by(PreprocessRun.created_at.desc())
                          .first())

        if not preprocess_run or not preprocess_run.output_path:
            flash("Belum ada preprocessing yang sukses untuk dataset ini. Jalankan preprocessing dulu.", "error")
            return redirect(url_for("pca.index", dataset_id=dataset.id))

        output_folder = os.path.join(current_app.config["OUTPUT_FOLDER"], "pca")
        os.makedirs(output_folder, exist_ok=True)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_filename = f"Hasil_PCA_dataset{dataset.id}_pre{preprocess_run.id}_{ts}.xlsx"

        try:
            result = run_pca(
                preprocessed_path=preprocess_run.output_path,
                output_folder=output_folder,
                output_filename=output_filename,
                selected_components_count=selected_k,
                method=method,
                include_covariance_sheet=True
            )

            run = PCARun(
                preprocess_run_id=preprocess_run.id,
                method=result["method"],
                n_components_total=result["n_components_total"],
                selected_components=result["selected_components"],
                eigenvalues=result["eigenvalues"],
                explained_variance=result["explained_variance"],
                cumulative_variance=result["cumulative_variance"],
                output_path=result["output_path"],
                status="success",
                error_message=None
            )
            db.session.add(run)
            db.session.commit()

            flash("PCA berhasil! Output tersimpan dan tercatat di database.", "success")
            return redirect(url_for("pca.index", dataset_id=dataset.id))

        except Exception as e:
            db.session.rollback()

            # Catat kegagalan (supaya bisa dilihat di history)
            try:
                fail_run = PCARun(
                    preprocess_run_id=preprocess_run.id,
                    method=method,
                    n_components_total=0,
                    selected_components=[1, 2, 3, 4],
                    eigenvalues=None,
                    explained_variance=None,
                    cumulative_variance=None,
                    output_path=None,
                    status="failed",
                    error_message=str(e)
                )
                db.session.add(fail_run)
                db.session.commit()
            except Exception:
                db.session.rollback()

            flash(f"PCA gagal: {e}", "error")
            return redirect(url_for("pca.index", dataset_id=dataset.id))

    # (GET) jika user memilih dataset -> tampilkan preprocess run terakhir & pca run terakhir untuk dataset tsb
    last_preprocess = None
    last_pca_for_dataset = None
    if selected_dataset_id:
        last_preprocess = (PreprocessRun.query
                           .filter_by(dataset_id=selected_dataset_id, status="success")
                           .order_by(PreprocessRun.created_at.desc())
                           .first())
        if last_preprocess:
            last_pca_for_dataset = (PCARun.query
                                    .filter_by(preprocess_run_id=last_preprocess.id)
                                    .order_by(PCARun.created_at.desc())
                                    .first())

    return render_template(
        "pages/pca.html",
        datasets=datasets,
        pca_runs=pca_runs,
        selected_dataset_id=selected_dataset_id,
        last_preprocess=last_preprocess,
        last_pca_for_dataset=last_pca_for_dataset
    )


@pca_bp.route("/download/<int:run_id>", methods=["GET"])
def download(run_id: int):
    run = PCARun.query.get_or_404(run_id)

    if not run.output_path:
        flash("Output PCA belum tersedia.", "error")
        return redirect(url_for("pca.index"))

    allowed_dir = os.path.abspath(os.path.join(current_app.config["OUTPUT_FOLDER"], "pca"))
    file_path = os.path.abspath(run.output_path)

    if not file_path.startswith(allowed_dir):
        flash("Path file tidak valid.", "error")
        return redirect(url_for("pca.index"))

    if not os.path.exists(file_path):
        flash("File output tidak ditemukan di server.", "error")
        return redirect(url_for("pca.index"))

    return send_file(file_path, as_attachment=True)
