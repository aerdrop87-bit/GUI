# ==========================================
# APP.PY - MAIN FLASK APPLICATION
# Sistem Clustering & Prediksi PDAM
# ==========================================

from flask import Flask, render_template, request
import os 
from config import Config
from models import db
from models.dataset import Dataset
from models.pipeline_models import (
    PreprocessRun,
    PCARun,
    SilhouetteRun,
    KMeansRun,
    RFEvaluation,
)
from controllers.upload_controller import upload_bp
from controllers.preprocessing_controller import preprocessing_bp
from controllers.pca_controller import pca_bp
from controllers.clustering_controller import clustering_bp
from controllers.rf_controller import rf_bp


# ==========================================
# INISIALISASI FLASK
# ==========================================
app = Flask(__name__)
# Load semua config dari config.py 
app.config.from_object(Config)

db.init_app(app)

# Pastikan folder ada
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# Register Blue print
app.register_blueprint(upload_bp)
app.register_blueprint(preprocessing_bp)
app.register_blueprint(pca_bp)
app.register_blueprint(clustering_bp)
app.register_blueprint(rf_bp)


# ==========================================
# ROUTE: HALAMAN UTAMA
# ==========================================
@app.route("/")
def index():
    return render_template("pages/index.html")

# ==========================================
# ROUTE: DASHBOARD (Placeholder)
# ==========================================
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# ==========================================
# ROUTE: EVALUATION METRICS
# ==========================================
@app.route("/evaluasi")
def evaluasi():
    datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).all()
    selected_dataset_id = request.args.get("dataset_id", type=int)

    selected_dataset = None
    preprocess_run = None
    pca_run = None
    silhouette_run = None
    kmeans_run = None
    rf_eval_train = None
    rf_eval_test = None

    if selected_dataset_id:
        selected_dataset = Dataset.query.get(selected_dataset_id)

        preprocess_run = (
            PreprocessRun.query
            .filter_by(dataset_id=selected_dataset_id, status="success")
            .order_by(PreprocessRun.created_at.desc())
            .first()
        )

        if preprocess_run:
            pca_run = (
                PCARun.query
                .filter_by(preprocess_run_id=preprocess_run.id, status="success")
                .order_by(PCARun.created_at.desc())
                .first()
            )

        if pca_run:
            silhouette_run = (
                SilhouetteRun.query
                .filter_by(pca_run_id=pca_run.id, status="success")
                .order_by(SilhouetteRun.created_at.desc())
                .first()
            )
            kmeans_run = (
                KMeansRun.query
                .filter_by(pca_run_id=pca_run.id, status="success")
                .order_by(KMeansRun.created_at.desc())
                .first()
            )

        if kmeans_run:
            rf_eval_train = (
                RFEvaluation.query
                .filter_by(kmeans_run_id=kmeans_run.id, split_type="train", status="success")
                .order_by(RFEvaluation.created_at.desc())
                .first()
            )
            rf_eval_test = (
                RFEvaluation.query
                .filter_by(kmeans_run_id=kmeans_run.id, split_type="test", status="success")
                .order_by(RFEvaluation.created_at.desc())
                .first()
            )

    return render_template(
        "pages/evaluasi.html",
        datasets=datasets,
        selected_dataset_id=selected_dataset_id,
        selected_dataset=selected_dataset,
        preprocess_run=preprocess_run,
        pca_run=pca_run,
        silhouette_run=silhouette_run,
        kmeans_run=kmeans_run,
        rf_eval_train=rf_eval_train,
        rf_eval_test=rf_eval_test,
    )


# ==========================================
# ROUTE: DATA VISUALIZATION
# ==========================================
@app.route("/visualisasi")
def visualisasi():
    return render_template("pages/visualisasi.html")


# ==========================================
# ROUTE: DOWNLOAD REPORT
# ==========================================
@app.route("/unduh")
def unduh():
    return render_template("pages/unduh.html")


# ==========================================
# ROUTE: SETTINGS
# ==========================================
@app.route("/pengaturan")
def pengaturan():
    return render_template("pages/pengaturan.html")


# ==========================================
# RUN SERVER
# ==========================================
if __name__ == "__main__":
    
    with app.app_context():
        db.create_all()
    
    app.run(debug=True)
