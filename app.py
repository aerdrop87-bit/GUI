# ==========================================
# APP.PY - MAIN FLASK APPLICATION
# Sistem Clustering & Prediksi PDAM
# ==========================================

from flask import Flask, render_template, request
import os 
import re
import pandas as pd
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


def _normalize_cluster_label(value):
    """Normalisasi label cluster ke format C1/C2/C3 bila memungkinkan."""
    if value is None:
        return ""
    try:
        # 0/1/2 -> C1/C2/C3
        if isinstance(value, (int, float)):
            if pd.isna(value):
                return ""
            return f"C{int(round(float(value))) + 1}"
    except Exception:
        pass

    s = str(value).strip()
    if not s:
        return ""
    if s.isdigit():
        return f"C{int(s) + 1}"
    m = re.match(r"^[cC]\s*(\d+)$", s)
    if m:
        return f"C{m.group(1)}"
    return s


def _extract_rf_metrics(eval_obj):
    """Ambil accuracy, precision, recall, f1 dari classification_report."""
    if not eval_obj:
        return None
    rep = eval_obj.classification_report or {}
    macro = rep.get("macro avg", {}) if isinstance(rep, dict) else {}
    return {
        "accuracy": eval_obj.accuracy,
        "precision": macro.get("precision"),
        "recall": macro.get("recall"),
        "f1_score": macro.get("f1-score"),
    }


def _build_confusion_matrix(eval_obj):
    """
    Bangun confusion matrix dari output evaluasi RF (sheet prediksi).
    Return None bila data tidak tersedia/format tidak cocok.
    """
    if not eval_obj or not eval_obj.output_path or not os.path.exists(eval_obj.output_path):
        return None

    try:
        xls = pd.ExcelFile(eval_obj.output_path)
        pred_sheet = next((s for s in xls.sheet_names if "Prediksi" in s), xls.sheet_names[0])
        df = pd.read_excel(eval_obj.output_path, sheet_name=pred_sheet)
    except Exception:
        return None

    actual_candidates = ["Cluster Aktual", "Cluster", "Actual", "actual", "y_true"]
    pred_candidates = ["Voting Mayoritas", "Prediksi", "Prediction", "pred", "y_pred"]

    actual_col = next((c for c in actual_candidates if c in df.columns), None)
    pred_col = next((c for c in pred_candidates if c in df.columns), None)
    if not actual_col or not pred_col:
        return None

    actual = df[actual_col].map(_normalize_cluster_label).tolist()
    pred = df[pred_col].map(_normalize_cluster_label).tolist()

    labels = ["C1", "C2", "C3"]
    idx = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]

    total = 0
    correct = 0
    for a, p in zip(actual, pred):
        if a not in idx or p not in idx:
            continue
        total += 1
        if a == p:
            correct += 1
        matrix[idx[a]][idx[p]] += 1

    if total == 0:
        return None

    return {
        "labels": labels,
        "matrix": matrix,
        "total": total,
        "correct": correct,
    }


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
    rf_train_metrics = None
    rf_test_metrics = None
    confusion_train = None
    confusion_test = None

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

        rf_train_metrics = _extract_rf_metrics(rf_eval_train)
        rf_test_metrics = _extract_rf_metrics(rf_eval_test)
        confusion_train = _build_confusion_matrix(rf_eval_train)
        confusion_test = _build_confusion_matrix(rf_eval_test)

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
        rf_train_metrics=rf_train_metrics,
        rf_test_metrics=rf_test_metrics,
        confusion_train=confusion_train,
        confusion_test=confusion_test,
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
