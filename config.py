import os
import secrets

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:

    # ==============================
    # SECRET KEY
    # ==============================
    SECRET_KEY = secrets.token_hex(16)

    # ==============================
    # DATABASE
    # ==============================
    SQLALCHEMY_DATABASE_URI = \
        "sqlite:///" + os.path.join(BASE_DIR, "pdam.db")

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ==============================
    # UPLOAD CONFIG
    # ==============================
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

    ALLOWED_EXTENSIONS = {"xlsx"}
