from flask import Blueprint, render_template, request, redirect, url_for, flash
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename

from models import db
from models.dataset import Dataset

# ==============================
# BLUEPRINT INIT
# ==============================
upload_bp = Blueprint(
    "upload",
    __name__,
    url_prefix="/upload"
)

# ==============================
# CONFIG LOCAL
# ==============================
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"xlsx"}


def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ==============================
# ROUTE: UPLOAD
# ==============================
@upload_bp.route("/", methods=["GET", "POST"])
def upload_file():

    if request.method == "POST":

        if "file" not in request.files:
            flash("File tidak ditemukan!", "error")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("File belum dipilih!", "error")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Format harus .xlsx", "error")
            return redirect(request.url)

        # ==============================
        # RENAME FILE
        # ==============================
        original_name = secure_filename(file.filename)
        name_only = original_name.rsplit(".", 1)[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]

        new_filename = f"{name_only}_{timestamp}_{short_uuid}.xlsx"

        filepath = os.path.join(
            os.path.abspath(UPLOAD_FOLDER),
            new_filename
        )

        file.save(filepath)

        # ==============================
        # SAVE TO DB
        # ==============================
        dataset = Dataset(
            filename=new_filename,
            filepath=filepath
        )

        db.session.add(dataset)
        db.session.commit()

        flash("Upload & simpan database berhasil!", "success")

        return redirect(url_for("upload.upload_file"))

    return render_template("pages/upload.html")