# ==========================================
# APP.PY - MAIN FLASK APPLICATION
# Sistem Clustering & Prediksi PDAM
# ==========================================

from flask import Flask, render_template
import os 
from config import Config
from models import db
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
    return render_template("pages/evaluasi.html")


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