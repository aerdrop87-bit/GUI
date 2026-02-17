# ==========================================
# APP.PY - MAIN FLASK APPLICATION
# Sistem Clustering & Prediksi PDAM
# ==========================================

from flask import Flask, render_template, request, url_for
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
    DataSplit,
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


def _find_col(df, candidates):
    """Cari nama kolom yang cocok (case-insensitive, trim)."""
    if df is None or df.empty:
        return None
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _file_meta(path):
    """Ambil metadata file untuk tampilan unduh."""
    if not path or not os.path.exists(path):
        return None
    try:
        stat = os.stat(path)
        size_bytes = stat.st_size
        if size_bytes < 1024:
            size_text = f"{size_bytes} B"
        elif size_bytes < (1024 * 1024):
            size_text = f"{size_bytes / 1024:.1f} KB"
        else:
            size_text = f"{size_bytes / (1024 * 1024):.2f} MB"
        modified_text = pd.to_datetime(stat.st_mtime, unit="s").strftime("%d %b %Y %H:%M")
        return {
            "filename": os.path.basename(path),
            "size_text": size_text,
            "modified_text": modified_text,
        }
    except Exception:
        return None


def _get_dataset_dimensions(dataset_obj):
    """
    Ambil dimensi dataset (rows, cols).
    Prioritaskan metadata di DB, fallback baca file bila metadata belum ada.
    """
    rows = dataset_obj.n_rows if isinstance(dataset_obj.n_rows, int) and dataset_obj.n_rows >= 0 else None
    cols = dataset_obj.n_cols if isinstance(dataset_obj.n_cols, int) and dataset_obj.n_cols >= 0 else None

    if (rows is None or cols is None) and dataset_obj.filepath and os.path.exists(dataset_obj.filepath):
        try:
            df_tmp = pd.read_excel(dataset_obj.filepath)
            if rows is None:
                rows = int(df_tmp.shape[0])
            if cols is None:
                cols = int(df_tmp.shape[1])
        except Exception:
            pass

    return int(rows or 0), int(cols or 0)


def _get_chart_export_data(dataset_id):
    """
    Ambil data mentah chart untuk ekspor PNG ZIP di halaman unduh.
    Mengembalikan dict berisi dataset chart yang tersedia.
    """
    payload = {
        "usage_distribution": None,
        "region_distribution": None,
        "cluster_comparison": None,
        "monthly_trend": None,
        "scatter_cluster": None,
        "available_chart_count": 0,
    }

    if not dataset_id:
        return payload

    dataset = Dataset.query.get(dataset_id)
    if not dataset or not dataset.filepath or not os.path.exists(dataset.filepath):
        return payload

    try:
        df_raw = pd.read_excel(dataset.filepath)
    except Exception:
        return payload

    if df_raw is None or df_raw.empty:
        return payload

    usage_col = _find_col(df_raw, ["pemakaian", "pemakaian air", "usage", "water usage"])
    region_col = _find_col(df_raw, ["wilayah", "region", "area"])
    revenue_col = _find_col(df_raw, ["jumlah tagihan (rp.)", "jumlah tagihan", "bill amount", "revenue"])
    tenure_col = _find_col(df_raw, ["lama berlangganan", "tenure", "subscription duration"])
    month_col = _find_col(df_raw, ["bulan", "month", "periode", "period", "tanggal", "date"])

    if usage_col:
        usage_series = pd.to_numeric(df_raw[usage_col], errors="coerce").dropna()
        if not usage_series.empty:
            bins = [-float("inf"), 10, 20, 30, 50, float("inf")]
            labels = ["<=10", "11-20", "21-30", "31-50", ">50"]
            dist = pd.cut(usage_series, bins=bins, labels=labels).value_counts(sort=False)
            payload["usage_distribution"] = [{"range": str(idx), "count": int(val)} for idx, val in dist.items()]

    if region_col:
        region_counts = df_raw[region_col].astype(str).str.strip().value_counts().sort_index()
        if len(region_counts):
            payload["region_distribution"] = [{"region": str(idx), "count": int(val)} for idx, val in region_counts.items()]

    if month_col and (usage_col or revenue_col):
        dt = pd.to_datetime(df_raw[month_col], errors="coerce")
        df_m = df_raw.copy()
        df_m["_month"] = dt.dt.to_period("M").astype(str)
        df_m = df_m[df_m["_month"].notna()]
        if not df_m.empty:
            agg = df_m.groupby("_month", as_index=False).agg(
                usage=(usage_col, lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()) if usage_col else ("_month", "size"),
                revenue=(revenue_col, lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()) if revenue_col else ("_month", "size"),
            )
            agg = agg.sort_values("_month").tail(12)
            payload["monthly_trend"] = [
                {"month": r["_month"], "usage": float(r["usage"]) if usage_col else None, "revenue": float(r["revenue"]) if revenue_col else None}
                for _, r in agg.iterrows()
            ]

    preprocess_run = (
        PreprocessRun.query
        .filter_by(dataset_id=dataset_id, status="success")
        .order_by(PreprocessRun.created_at.desc())
        .first()
    )
    latest_kmeans = None
    if preprocess_run:
        pca_run = (
            PCARun.query
            .filter_by(preprocess_run_id=preprocess_run.id, status="success")
            .order_by(PCARun.created_at.desc())
            .first()
        )
        if pca_run:
            latest_kmeans = (
                KMeansRun.query
                .filter_by(pca_run_id=pca_run.id, status="success")
                .order_by(KMeansRun.created_at.desc())
                .first()
            )

    if latest_kmeans and latest_kmeans.output_path and os.path.exists(latest_kmeans.output_path):
        try:
            df_cluster = pd.read_excel(latest_kmeans.output_path, sheet_name="Data+Cluster")
            if "Row" in df_cluster.columns and "Cluster" in df_cluster.columns:
                df_join = df_raw.copy().reset_index(drop=True)
                df_join["Row"] = df_join.index + 1
                df_join = df_join.merge(df_cluster[["Row", "Cluster"]], on="Row", how="left")

                metrics_map = []
                if usage_col:
                    metrics_map.append(("Pemakaian", usage_col))
                if revenue_col:
                    metrics_map.append(("Tagihan (Rp)", revenue_col))
                if tenure_col:
                    metrics_map.append(("Lama Berlangganan", tenure_col))

                if metrics_map and "Cluster" in df_join.columns:
                    rows = []
                    for _, g in df_join.groupby("Cluster", as_index=False):
                        c = str(g["Cluster"].iloc[0])
                        item = {"cluster": c}
                        for label, col in metrics_map:
                            item[label] = float(pd.to_numeric(g[col], errors="coerce").fillna(0).mean())
                        rows.append(item)
                    payload["cluster_comparison"] = rows if rows else None

                if usage_col and revenue_col and "Cluster" in df_join.columns:
                    pts = df_join[[usage_col, revenue_col, "Cluster"]].copy()
                    pts.columns = ["usage", "revenue", "cluster"]
                    pts["usage"] = pd.to_numeric(pts["usage"], errors="coerce")
                    pts["revenue"] = pd.to_numeric(pts["revenue"], errors="coerce")
                    pts = pts.dropna(subset=["usage", "revenue", "cluster"])
                    payload["scatter_cluster"] = pts.to_dict(orient="records") if not pts.empty else None
        except Exception:
            pass

    payload["available_chart_count"] = len(
        [k for k in ["usage_distribution", "region_distribution", "cluster_comparison", "monthly_trend", "scatter_cluster"] if payload.get(k)]
    )
    return payload


# ==========================================
# ROUTE: HALAMAN UTAMA
# ==========================================
@app.route("/")
def index():
    datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).all()
    selected_dataset_id = request.args.get("dataset_id", type=int)

    selected_dataset = None
    if selected_dataset_id:
        selected_dataset = Dataset.query.get(selected_dataset_id)
    elif datasets:
        selected_dataset = datasets[0]
        selected_dataset_id = selected_dataset.id

    upload_rows = []
    total_raw_rows_uploaded = 0
    total_raw_columns_uploaded = 0
    missing_files_count = 0

    for idx, ds in enumerate(datasets, start=1):
        row_count, col_count = _get_dataset_dimensions(ds)
        file_exists = bool(ds.filepath and os.path.exists(ds.filepath))
        if not file_exists:
            missing_files_count += 1

        total_raw_rows_uploaded += row_count
        total_raw_columns_uploaded += col_count

        upload_rows.append(
            {
                "no": idx,
                "id": ds.id,
                "filename": ds.filename,
                "original_filename": ds.original_filename,
                "uploaded_at": ds.uploaded_at.strftime("%d-%m-%Y %H:%M") if ds.uploaded_at else "-",
                "row_count": row_count,
                "col_count": col_count,
                "file_exists": file_exists,
            }
        )

    dataset_count = len(datasets)
    avg_rows_per_dataset = float(total_raw_rows_uploaded / dataset_count) if dataset_count else 0.0
    avg_cols_per_dataset = float(total_raw_columns_uploaded / dataset_count) if dataset_count else 0.0

    preprocess_success_count = PreprocessRun.query.filter_by(status="success").count()
    pca_success_count = PCARun.query.filter_by(status="success").count()
    kmeans_success_count = KMeansRun.query.filter_by(status="success").count()
    silhouette_success_count = SilhouetteRun.query.filter_by(status="success").count()
    rf_eval_success_count = RFEvaluation.query.filter_by(status="success").count()

    latest_upload_name = datasets[0].filename if datasets else "-"
    latest_upload_time = (
        datasets[0].uploaded_at.strftime("%d-%m-%Y %H:%M")
        if datasets and datasets[0].uploaded_at
        else "-"
    )

    raw_preview_columns = []
    raw_preview_rows = []
    selected_dataset_stats = None
    preview_note = None

    if selected_dataset and selected_dataset.filepath and os.path.exists(selected_dataset.filepath):
        try:
            df_raw = pd.read_excel(selected_dataset.filepath)
            row_count = int(df_raw.shape[0])
            col_count = int(df_raw.shape[1])
            preview_df = df_raw.head(20).copy()
            preview_df = preview_df.where(pd.notnull(preview_df), "-")

            raw_preview_columns = [str(c) for c in preview_df.columns]
            raw_preview_rows = preview_df.astype(str).values.tolist()

            selected_dataset_stats = {
                "row_count": row_count,
                "col_count": col_count,
                "preview_count": int(len(raw_preview_rows)),
                "uploaded_at": selected_dataset.uploaded_at.strftime("%d-%m-%Y %H:%M")
                if selected_dataset.uploaded_at
                else "-",
            }
        except Exception:
            preview_note = "Data mentah tidak dapat dibaca dari file dataset terpilih."
    elif selected_dataset:
        preview_note = "File dataset terpilih tidak ditemukan di server."

    return render_template(
        "pages/index.html",
        datasets=datasets,
        selected_dataset=selected_dataset,
        selected_dataset_id=selected_dataset_id,
        dataset_count=dataset_count,
        total_raw_rows_uploaded=total_raw_rows_uploaded,
        avg_rows_per_dataset=avg_rows_per_dataset,
        avg_cols_per_dataset=avg_cols_per_dataset,
        missing_files_count=missing_files_count,
        latest_upload_name=latest_upload_name,
        latest_upload_time=latest_upload_time,
        preprocess_success_count=preprocess_success_count,
        pca_success_count=pca_success_count,
        silhouette_success_count=silhouette_success_count,
        kmeans_success_count=kmeans_success_count,
        rf_eval_success_count=rf_eval_success_count,
        upload_rows=upload_rows,
        raw_preview_columns=raw_preview_columns,
        raw_preview_rows=raw_preview_rows,
        selected_dataset_stats=selected_dataset_stats,
        preview_note=preview_note,
    )

# ==========================================
# ROUTE: DASHBOARD (Placeholder)
# ==========================================
@app.route("/dashboard")
def dashboard():
    # Gunakan data context yang sama dengan halaman index dashboard.
    return index()

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
    datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).all()
    selected_dataset_id = request.args.get("dataset_id", type=int)

    selected_dataset = None
    usage_distribution = None
    region_distribution = None
    cluster_comparison = None
    scatter_cluster = None
    monthly_trend = None
    monthly_trend_note = None
    kpi = {
        "total_water_consumption": None,
        "total_revenue": None,
        "payment_collection_rate": None,
    }
    visual_notes = []
    usage_summary = None
    region_summary = None
    cluster_summary = None
    monthly_summary = None
    scatter_summary = None

    if selected_dataset_id:
        selected_dataset = Dataset.query.get(selected_dataset_id)

        if selected_dataset and selected_dataset.filepath and os.path.exists(selected_dataset.filepath):
            try:
                df_raw = pd.read_excel(selected_dataset.filepath)
            except Exception:
                df_raw = None
                visual_notes.append("Dataset tidak dapat dibaca untuk kebutuhan visualisasi.")

            if df_raw is not None and not df_raw.empty:
                usage_col = _find_col(df_raw, ["pemakaian", "pemakaian air", "usage", "water usage"])
                region_col = _find_col(df_raw, ["wilayah", "region", "area"])
                revenue_col = _find_col(df_raw, ["jumlah tagihan (rp.)", "jumlah tagihan", "bill amount", "revenue"])
                slip_col = _find_col(df_raw, ["slip tagihan", "payment slip", "status pembayaran"])
                tenure_col = _find_col(df_raw, ["lama berlangganan", "tenure", "subscription duration"])

                if usage_col:
                    usage_series = pd.to_numeric(df_raw[usage_col], errors="coerce").dropna()
                    if not usage_series.empty:
                        bins = [-float("inf"), 10, 20, 30, 50, float("inf")]
                        labels = ["<=10", "11-20", "21-30", "31-50", ">50"]
                        binned = pd.cut(usage_series, bins=bins, labels=labels)
                        dist = binned.value_counts(sort=False)
                        usage_distribution = [{"range": str(idx), "count": int(val)} for idx, val in dist.items()]
                        kpi["total_water_consumption"] = float(usage_series.sum())
                        top_range = dist.idxmax() if len(dist) else None
                        top_count = int(dist.max()) if len(dist) else 0
                        usage_summary = {
                            "total_customers": int(len(usage_series)),
                            "avg_usage": float(usage_series.mean()),
                            "top_range": str(top_range) if top_range is not None else "-",
                            "top_count": top_count,
                        }

                if region_col:
                    region_series = df_raw[region_col].astype(str).str.strip()
                    region_counts = region_series.value_counts().sort_index()
                    region_distribution = [{"region": str(idx), "count": int(val)} for idx, val in region_counts.items()]
                    if len(region_counts):
                        top_region = str(region_counts.idxmax())
                        top_region_count = int(region_counts.max())
                        total_region_customers = int(region_counts.sum())
                        top_region_pct = float((top_region_count / total_region_customers) * 100.0) if total_region_customers else 0.0
                        region_summary = {
                            "total_regions": int(region_counts.shape[0]),
                            "top_region": top_region,
                            "top_region_count": top_region_count,
                            "top_region_pct": top_region_pct,
                        }

                if revenue_col:
                    revenue_series = pd.to_numeric(df_raw[revenue_col], errors="coerce").fillna(0)
                    kpi["total_revenue"] = float(revenue_series.sum())

                if slip_col:
                    slip_series = pd.to_numeric(df_raw[slip_col], errors="coerce")
                    if revenue_col:
                        revenue_series = pd.to_numeric(df_raw[revenue_col], errors="coerce").fillna(0)
                        billable = revenue_series > 0
                        if billable.sum() > 0:
                            # Asumsi: slip tagihan = 0 dianggap tidak menunggak / collection baik.
                            paid = (slip_series.fillna(0) == 0) & billable
                            kpi["payment_collection_rate"] = float((paid.sum() / billable.sum()) * 100.0)

                # Monthly trend hanya dibuat jika ada kolom periode/tanggal.
                month_col = _find_col(df_raw, ["bulan", "month", "periode", "period", "tanggal", "date"])
                if month_col and (usage_col or revenue_col):
                    dt = pd.to_datetime(df_raw[month_col], errors="coerce")
                    df_m = df_raw.copy()
                    df_m["_month"] = dt.dt.to_period("M").astype(str)
                    df_m = df_m[df_m["_month"].notna()]
                    if not df_m.empty:
                        agg = df_m.groupby("_month", as_index=False).agg(
                            usage=(usage_col, lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()) if usage_col else ("_month", "size"),
                            revenue=(revenue_col, lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()) if revenue_col else ("_month", "size"),
                        )
                        agg = agg.sort_values("_month").tail(12)
                        monthly_trend = [
                            {"month": r["_month"], "usage": float(r["usage"]) if usage_col else None, "revenue": float(r["revenue"]) if revenue_col else None}
                            for _, r in agg.iterrows()
                        ]
                        if monthly_trend:
                            first = monthly_trend[0]
                            last = monthly_trend[-1]
                            usage_change = None
                            revenue_change = None
                            if usage_col and first.get("usage") is not None and last.get("usage") is not None:
                                usage_change = float(last["usage"] - first["usage"])
                            if revenue_col and first.get("revenue") is not None and last.get("revenue") is not None:
                                revenue_change = float(last["revenue"] - first["revenue"])
                            monthly_summary = {
                                "months_count": int(len(monthly_trend)),
                                "first_month": first.get("month"),
                                "last_month": last.get("month"),
                                "usage_change": usage_change,
                                "revenue_change": revenue_change,
                            }
                else:
                    monthly_trend_note = "Trend bulanan belum dapat ditampilkan karena dataset belum memiliki kolom periode/tanggal."

                # Ambil cluster terbaru untuk dataset terpilih lalu gabungkan dengan data mentah
                preprocess_run = (
                    PreprocessRun.query
                    .filter_by(dataset_id=selected_dataset_id, status="success")
                    .order_by(PreprocessRun.created_at.desc())
                    .first()
                )
                latest_kmeans = None
                if preprocess_run:
                    pca_run = (
                        PCARun.query
                        .filter_by(preprocess_run_id=preprocess_run.id, status="success")
                        .order_by(PCARun.created_at.desc())
                        .first()
                    )
                    if pca_run:
                        latest_kmeans = (
                            KMeansRun.query
                            .filter_by(pca_run_id=pca_run.id, status="success")
                            .order_by(KMeansRun.created_at.desc())
                            .first()
                        )

                if latest_kmeans and latest_kmeans.output_path and os.path.exists(latest_kmeans.output_path):
                    try:
                        df_cluster = pd.read_excel(latest_kmeans.output_path, sheet_name="Data+Cluster")
                        if "Row" in df_cluster.columns and "Cluster" in df_cluster.columns:
                            df_join = df_raw.copy().reset_index(drop=True)
                            df_join["Row"] = df_join.index + 1
                            df_join = df_join.merge(df_cluster[["Row", "Cluster"]], on="Row", how="left")

                            metrics_map = []
                            if usage_col:
                                metrics_map.append(("Pemakaian", usage_col))
                            if revenue_col:
                                metrics_map.append(("Tagihan (Rp)", revenue_col))
                            if tenure_col:
                                metrics_map.append(("Lama Berlangganan", tenure_col))

                            if metrics_map and "Cluster" in df_join.columns:
                                group = df_join.groupby("Cluster", as_index=False)
                                rows = []
                                for _, g in group:
                                    c = str(g["Cluster"].iloc[0])
                                    item = {"cluster": c}
                                    for label, col in metrics_map:
                                        item[label] = float(pd.to_numeric(g[col], errors="coerce").fillna(0).mean())
                                    rows.append(item)
                                cluster_comparison = rows
                                if rows:
                                    # Prioritaskan indikator pemakaian bila ada.
                                    usage_key = "Pemakaian" if any("Pemakaian" in k for k in rows[0].keys()) else None
                                    if usage_key:
                                        top_cluster = max(rows, key=lambda x: x.get(usage_key, 0))
                                        cluster_summary = {
                                            "cluster_count": int(len(rows)),
                                            "top_cluster": str(top_cluster.get("cluster", "-")),
                                            "top_cluster_usage": float(top_cluster.get(usage_key, 0)),
                                        }
                                    else:
                                        cluster_summary = {
                                            "cluster_count": int(len(rows)),
                                            "top_cluster": None,
                                            "top_cluster_usage": None,
                                        }

                            if usage_col and revenue_col and "Cluster" in df_join.columns:
                                pts = df_join[[usage_col, revenue_col, "Cluster"]].copy()
                                pts.columns = ["usage", "revenue", "cluster"]
                                pts["usage"] = pd.to_numeric(pts["usage"], errors="coerce")
                                pts["revenue"] = pd.to_numeric(pts["revenue"], errors="coerce")
                                pts = pts.dropna(subset=["usage", "revenue", "cluster"])
                                scatter_cluster = pts.to_dict(orient="records")
                                if not pts.empty:
                                    corr = pts["usage"].corr(pts["revenue"])
                                    scatter_summary = {
                                        "points_count": int(len(pts)),
                                        "corr_usage_revenue": float(corr) if pd.notna(corr) else None,
                                    }
                    except Exception:
                        visual_notes.append("Data cluster tidak dapat diproses untuk visualisasi lanjutan.")
                else:
                    visual_notes.append("Cluster comparison membutuhkan hasil KMeans terbaru untuk dataset terpilih.")

    return render_template(
        "pages/visualisasi.html",
        datasets=datasets,
        selected_dataset_id=selected_dataset_id,
        selected_dataset=selected_dataset,
        usage_distribution=usage_distribution,
        region_distribution=region_distribution,
        cluster_comparison=cluster_comparison,
        monthly_trend=monthly_trend,
        monthly_trend_note=monthly_trend_note,
        scatter_cluster=scatter_cluster,
        kpi=kpi,
        visual_notes=visual_notes,
        usage_summary=usage_summary,
        region_summary=region_summary,
        cluster_summary=cluster_summary,
        monthly_summary=monthly_summary,
        scatter_summary=scatter_summary,
    )


# ==========================================
# ROUTE: DOWNLOAD REPORT
# ==========================================
@app.route("/unduh")
def unduh():
    datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).all()
    selected_dataset_id = request.args.get("dataset_id", type=int)

    selected_dataset = None
    download_items = []
    chart_export_data = _get_chart_export_data(selected_dataset_id)

    if selected_dataset_id:
        selected_dataset = Dataset.query.get(selected_dataset_id)

        preprocess_run = (
            PreprocessRun.query
            .filter_by(dataset_id=selected_dataset_id, status="success")
            .order_by(PreprocessRun.created_at.desc())
            .first()
        )

        pca_run = None
        if preprocess_run:
            pca_run = (
                PCARun.query
                .filter_by(preprocess_run_id=preprocess_run.id, status="success")
                .order_by(PCARun.created_at.desc())
                .first()
            )

        kmeans_run = None
        if pca_run:
            kmeans_run = (
                KMeansRun.query
                .filter_by(pca_run_id=pca_run.id, status="success")
                .order_by(KMeansRun.created_at.desc())
                .first()
            )

        split_run = None
        rf_eval_train = None
        rf_eval_test = None
        if kmeans_run:
            split_run = (
                DataSplit.query
                .filter_by(kmeans_run_id=kmeans_run.id)
                .order_by(DataSplit.created_at.desc())
                .first()
            )
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

        def add_item(title, description, path, url=None):
            meta = _file_meta(path) if path else None
            download_items.append({
                "title": title,
                "description": description,
                "ready": meta is not None and bool(url),
                "url": url,
                "filename": meta["filename"] if meta else "-",
                "size_text": meta["size_text"] if meta else "-",
                "modified_text": meta["modified_text"] if meta else "-",
            })

        add_item(
            title="Standardized Dataset",
            description="Output preprocessing (standardisasi) terbaru untuk dataset terpilih.",
            path=preprocess_run.output_path if preprocess_run else None,
            url=url_for("preprocessing.download", run_id=preprocess_run.id) if preprocess_run and preprocess_run.output_path else None,
        )
        add_item(
            title="PCA Analysis Results",
            description="Hasil analisis PCA terbaru beserta komponen yang digunakan.",
            path=pca_run.output_path if pca_run else None,
            url=url_for("pca.download", run_id=pca_run.id) if pca_run and pca_run.output_path else None,
        )
        add_item(
            title="Clustering Results",
            description="Hasil KMeans clustering terbaru (cluster assignment dan centroid).",
            path=kmeans_run.output_path if kmeans_run else None,
            url=url_for("clustering.download", run_id=kmeans_run.id) if kmeans_run and kmeans_run.output_path else None,
        )
        add_item(
            title="RF Evaluation (Train)",
            description="Laporan evaluasi Random Forest untuk data train.",
            path=rf_eval_train.output_path if rf_eval_train else None,
            url=url_for("rf.download_eval", eval_id=rf_eval_train.id) if rf_eval_train and rf_eval_train.output_path else None,
        )
        add_item(
            title="RF Evaluation (Test)",
            description="Laporan evaluasi Random Forest untuk data test.",
            path=rf_eval_test.output_path if rf_eval_test else None,
            url=url_for("rf.download_eval", eval_id=rf_eval_test.id) if rf_eval_test and rf_eval_test.output_path else None,
        )
        add_item(
            title="RF Split Data (Train)",
            description="File data train hasil proses split terbaru.",
            path=split_run.train_path if split_run else None,
            url=url_for("rf.download_split", split_id=split_run.id, which="train") if split_run and split_run.train_path else None,
        )
        add_item(
            title="RF Split Data (Test)",
            description="File data test hasil proses split terbaru.",
            path=split_run.test_path if split_run else None,
            url=url_for("rf.download_split", split_id=split_run.id, which="test") if split_run and split_run.test_path else None,
        )

        download_items.append({
            "title": "Visualization Charts (PNG ZIP)",
            "description": "Unduh semua chart visualisasi yang tersedia dalam 1 file ZIP berisi PNG.",
            "ready": chart_export_data.get("available_chart_count", 0) > 0,
            "url": None,
            "filename": "visualization_charts_png.zip",
            "size_text": "-",
            "modified_text": "generated on demand",
            "kind": "chart_zip",
            "chart_count": chart_export_data.get("available_chart_count", 0),
        })

    ready_count = len([x for x in download_items if x["ready"]])

    return render_template(
        "pages/unduh.html",
        datasets=datasets,
        selected_dataset_id=selected_dataset_id,
        selected_dataset=selected_dataset,
        download_items=download_items,
        ready_count=ready_count,
        chart_export_data=chart_export_data,
    )


# ==========================================
# RUN SERVER
# ==========================================
if __name__ == "__main__":
    
    with app.app_context():
        db.create_all()
    
    app.run(debug=True)
