from datetime import datetime
from . import db


class Dataset(db.Model):
    """
    Dataset mentah yang di-upload user.

    Kolom lama dipertahankan:
    - filename  = nama file yang disimpan (hasil rename)
    - filepath  = path absolut file mentah

    Kolom baru:
    - original_filename: nama file asli sebelum rename
    - n_rows / n_cols: statistik sederhana
    - note: catatan bebas
    """

    __tablename__ = "datasets"

    id = db.Column(db.Integer, primary_key=True)

    # ===== kolom lama (dipertahankan agar code lama tetap jalan) =====
    filename = db.Column(db.String(255), nullable=False, unique=True, index=True)
    filepath = db.Column(db.String(500), nullable=False)

    # ===== kolom baru =====
    original_filename = db.Column(db.String(255), nullable=True)
    n_rows = db.Column(db.Integer, nullable=True)
    n_cols = db.Column(db.Integer, nullable=True)
    note = db.Column(db.Text, nullable=True)

    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    # ===== relasi pipeline =====
    preprocess_runs = db.relationship(
        "PreprocessRun",
        backref="dataset",
        lazy=True,
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Dataset id={self.id} filename={self.filename}>"
