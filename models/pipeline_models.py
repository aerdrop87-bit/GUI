from datetime import datetime
from . import db


class PreprocessRun(db.Model):
    """Mencatat output preprocessing (standarisasi) untuk sebuah dataset."""
    __tablename__ = "preprocess_runs"

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("datasets.id"), nullable=False, index=True)

    method = db.Column(db.String(50), default="standard_scaler")
    feature_columns = db.Column(db.JSON, nullable=False)        # list kolom yg distandarisasi
    imputation_strategy = db.Column(db.String(50), nullable=True)
    scaler_params = db.Column(db.JSON, nullable=True)           # mean/scale (opsional)
    output_path = db.Column(db.String(500), nullable=True)      # path output excel

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default="success")        # success/failed
    error_message = db.Column(db.Text, nullable=True)

    pca_runs = db.relationship(
        "PCARun", backref="preprocess_run", lazy=True, cascade="all, delete-orphan"
    )


class PCARun(db.Model):
    """Mencatat output PCA (scores, eigen info, loading matrix)."""
    __tablename__ = "pca_runs"

    id = db.Column(db.Integer, primary_key=True)
    preprocess_run_id = db.Column(db.Integer, db.ForeignKey("preprocess_runs.id"), nullable=False, index=True)

    method = db.Column(db.String(20), default="correlation")  # correlation/covariance
    n_components_total = db.Column(db.Integer, default=9)
    selected_components = db.Column(db.JSON, nullable=True)   # misal [1,2,3,4]

    eigenvalues = db.Column(db.JSON, nullable=True)
    explained_variance = db.Column(db.JSON, nullable=True)
    cumulative_variance = db.Column(db.JSON, nullable=True)

    output_path = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    status = db.Column(db.String(20), default="success")
    error_message = db.Column(db.Text, nullable=True)

    silhouette_runs = db.relationship(
        "SilhouetteRun", backref="pca_run", lazy=True, cascade="all, delete-orphan"
    )
    kmeans_runs = db.relationship(
        "KMeansRun", backref="pca_run", lazy=True, cascade="all, delete-orphan"
    )


class SilhouetteRun(db.Model):
    """Mencatat evaluasi silhouette untuk range K tertentu."""
    __tablename__ = "silhouette_runs"

    id = db.Column(db.Integer, primary_key=True)
    pca_run_id = db.Column(db.Integer, db.ForeignKey("pca_runs.id"), nullable=False, index=True)

    k_min = db.Column(db.Integer, default=2)
    k_max = db.Column(db.Integer, default=10)

    # format: [{"k":2,"score":0.41}, {"k":3,"score":0.52}, ...]
    scores_by_k = db.Column(db.JSON, nullable=False)
    best_k = db.Column(db.Integer, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default="success")
    error_message = db.Column(db.Text, nullable=True)


class KMeansRun(db.Model):
    """Mencatat hasil clustering KMeans pada data PCA (biasanya PCA1..PCA4)."""
    __tablename__ = "kmeans_runs"

    id = db.Column(db.Integer, primary_key=True)
    pca_run_id = db.Column(db.Integer, db.ForeignKey("pca_runs.id"), nullable=False, index=True)

    k = db.Column(db.Integer, nullable=False)

    init_method = db.Column(db.String(20), default="k-means++")  # manual / k-means++
    init_rows = db.Column(db.JSON, nullable=True)                 # misal [306,141,234] jika manual

    n_init = db.Column(db.Integer, default=10)
    max_iter = db.Column(db.Integer, default=300)
    random_state = db.Column(db.Integer, default=42)

    centroids_final = db.Column(db.JSON, nullable=True)
    counts_by_cluster = db.Column(db.JSON, nullable=True)

    output_path = db.Column(db.String(500), nullable=True)

    # =======================
    # NEW: log iterasi KMeans
    # =======================
    n_iter = db.Column(db.Integer, nullable=True)
    iterations_log = db.Column(db.JSON, nullable=True)           # list iter log
    labels_by_iteration = db.Column(db.JSON, nullable=True)      # list[iter][row] = "C1"
    centroids_by_iteration = db.Column(db.JSON, nullable=True)   # list[iter][k][d]

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default="success")
    error_message = db.Column(db.Text, nullable=True)

    data_splits = db.relationship(
        "DataSplit", backref="kmeans_run", lazy=True, cascade="all, delete-orphan"
    )
    rf_evaluations = db.relationship(
        "RFEvaluation", backref="kmeans_run", lazy=True, cascade="all, delete-orphan"
    )


class DataSplit(db.Model):
    """Opsional: menyimpan referensi file data latih & uji."""
    __tablename__ = "data_splits"

    id = db.Column(db.Integer, primary_key=True)
    kmeans_run_id = db.Column(db.Integer, db.ForeignKey("kmeans_runs.id"), nullable=False, index=True)

    train_path = db.Column(db.String(500), nullable=True)
    test_path = db.Column(db.String(500), nullable=True)

    split_method = db.Column(db.String(20), default="manual")  # manual/random/stratified
    train_ratio = db.Column(db.Float, nullable=True)
    random_state = db.Column(db.Integer, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)



class RFRuleSet(db.Model):
    """Menyimpan rule/ambang PCA untuk Random Forest manual (voting)."""
    __tablename__ = "rf_rulesets"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, default="RF Manual Ambang PCA")
    version = db.Column(db.String(20), default="v1")

    # {"PCA1":{"low":..,"mid":..}, ...}
    ambang = db.Column(db.JSON, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    evaluations = db.relationship(
        "RFEvaluation", backref="ruleset", lazy=True, cascade="all, delete-orphan"
    )


class RFEvaluation(db.Model):
    """Mencatat hasil evaluasi RF manual untuk data latih/uji."""
    __tablename__ = "rf_evaluations"

    id = db.Column(db.Integer, primary_key=True)

    kmeans_run_id = db.Column(db.Integer, db.ForeignKey("kmeans_runs.id"), nullable=False, index=True)
    ruleset_id = db.Column(db.Integer, db.ForeignKey("rf_rulesets.id"), nullable=False, index=True)

    split_type = db.Column(db.String(10), nullable=False)   # train/test
    input_path = db.Column(db.String(500), nullable=False)

    classification_report = db.Column(db.JSON, nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    macro_f1 = db.Column(db.Float, nullable=True)

    output_path = db.Column(db.String(500), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default="success")
    error_message = db.Column(db.Text, nullable=True)
