# ==========================================
# CLUSTERING SERVICE
# ==========================================
# Mengikuti alur Colab:
# 1) Ambil data PCA1..PCAk dari output PCA (sheet PCA_k atau PCA_Scores)
# 2) Silhouette score untuk menentukan K optimal
# 3) KMeans clustering (init k-means++ atau manual dari baris tertentu)
# 4) Simpan output ke Excel (data+cluster, centroid akhir, jumlah anggota)
#
# Catatan:
# - PCA Anda menyimpan sheet "PCA_{k}" (mis: PCA_4) dan "PCA_Scores"
# - Clustering default pakai 4 komponen (PCA1..PCA4)

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_pca_components(
    pca_output_path: str,
    components_count: int = 4
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load PCA1..PCAk dari file output PCA.

    Prioritas:
    1) Sheet "PCA_{k}" bila ada (dibuat oleh service PCA Anda)
    2) Sheet "PCA_Scores" lalu ambil kolom PCA1..PCAk

    Returns
    -------
    X_df: pd.DataFrame
        DataFrame berisi PCA1..PCAk
    cols: List[str]
        Nama kolom yang dipakai
    """
    k = int(components_count) if components_count else 4
    if k < 1:
        k = 1

    xls = pd.ExcelFile(pca_output_path)
    sheet_k = f"PCA_{k}"

    if sheet_k in xls.sheet_names:
        X_df = pd.read_excel(pca_output_path, sheet_name=sheet_k)
        # pastikan hanya PCA1..PCAk
        cols = [c for c in X_df.columns if isinstance(c, str) and c.startswith("PCA")]
        cols = cols[:k] if len(cols) >= k else cols
        X_df = X_df[cols].copy()
        return X_df, cols

    # fallback: PCA_Scores
    if "PCA_Scores" not in xls.sheet_names:
        raise ValueError("Sheet PCA_Scores tidak ditemukan pada file PCA output.")

    df_scores = pd.read_excel(pca_output_path, sheet_name="PCA_Scores")
    cols = [f"PCA{i+1}" for i in range(k)]
    missing = [c for c in cols if c not in df_scores.columns]
    if missing:
        raise ValueError("Kolom PCA yang dibutuhkan tidak ada: " + ", ".join(missing))

    X_df = df_scores[cols].copy()
    return X_df, cols


def run_silhouette(
    pca_output_path: str,
    components_count: int = 4,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Hitung silhouette score untuk K = k_min..k_max pada data PCA1..PCAk.

    Returns
    -------
    Dict berisi:
    - scores_by_k: list of dict [{"k":2,"score":0.123}, ...]
    - best_k: int (K dengan score tertinggi)
    - used_components: list kolom PCA yang dipakai
    """
    if k_min < 2:
        k_min = 2
    if k_max < k_min:
        k_max = k_min

    X_df, cols = load_pca_components(pca_output_path, components_count)
    X = X_df.values

    scores_by_k: List[Dict[str, Any]] = []
    best_k: Optional[int] = None
    best_score: Optional[float] = None

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state)
        labels = km.fit_predict(X)
        score = float(silhouette_score(X, labels))
        scores_by_k.append({"k": int(k), "score": score})

        if best_score is None or score > best_score:
            best_score = score
            best_k = int(k)

    return {
        "scores_by_k": scores_by_k,
        "best_k": best_k,
        "used_components": cols
    }


# ==========================================================
# NEW: KMeans dengan log iterasi (untuk visualisasi di HTML)
# ==========================================================

def _kmeans_plus_plus_init(X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    """
    Inisialisasi centroid ala k-means++ (implementasi ringan) agar bisa dicatat per-iterasi.
    """
    rng = np.random.default_rng(int(random_state))
    n = X.shape[0]

    first_idx = int(rng.integers(0, n))
    centroids = [X[first_idx].copy()]

    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centroids)[None, :, :]) ** 2).sum(axis=2), axis=1)
        total = float(d2.sum())
        if total <= 0:
            next_idx = int(rng.integers(0, n))
        else:
            probs = d2 / total
            next_idx = int(rng.choice(n, p=probs))
        centroids.append(X[next_idx].copy())

    return np.array(centroids)


def _lloyd_kmeans_with_history(
    X: np.ndarray,
    init_centroids: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-4
) -> Dict[str, Any]:
    """
    Lloyd's algorithm KMeans + simpan history per iterasi:
    - labels_by_iteration
    - centroids_by_iteration
    - iterations_log (inertia, max shift, counts)
    """
    centroids = init_centroids.copy()
    k = centroids.shape[0]
    n = X.shape[0]

    iterations_log: List[Dict[str, Any]] = []
    labels_by_iteration: List[List[int]] = []
    centroids_by_iteration: List[List[List[float]]] = []

    prev_inertia: Optional[float] = None

    for it in range(1, int(max_iter) + 1):
        # assignment step
        d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)  # (n, k)
        labels = d2.argmin(axis=1)
        min_d2 = d2[np.arange(n), labels]
        inertia = float(min_d2.sum())

        # update step
        new_centroids = np.zeros_like(centroids)
        counts = np.bincount(labels, minlength=k)

        for j in range(k):
            if counts[j] > 0:
                new_centroids[j] = X[labels == j].mean(axis=0)
            else:
                # cluster kosong -> reseed ke titik paling jauh
                far_idx = int(np.argmax(min_d2))
                new_centroids[j] = X[far_idx]

        shifts = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1))
        max_shift = float(shifts.max()) if len(shifts) else 0.0

        iterations_log.append({
            "iter": int(it),
            "inertia": inertia,
            "max_centroid_shift": max_shift,
            "counts": {int(i): int(counts[i]) for i in range(k)}
        })
        labels_by_iteration.append(labels.tolist())
        centroids_by_iteration.append(new_centroids.tolist())

        centroids = new_centroids

        # stop condition
        if max_shift <= tol:
            break
        if prev_inertia is not None and abs(prev_inertia - inertia) <= 1e-12:
            break
        prev_inertia = inertia

    final_labels = labels_by_iteration[-1]
    final_centroids = np.array(centroids_by_iteration[-1], dtype=float)
    final_inertia = float(iterations_log[-1]["inertia"])

    return {
        "labels": final_labels,
        "centroids": final_centroids,
        "inertia": final_inertia,
        "n_iter": int(iterations_log[-1]["iter"]),
        "iterations_log": iterations_log,
        "labels_by_iteration": labels_by_iteration,
        "centroids_by_iteration": centroids_by_iteration,
    }


def run_kmeans(
    pca_output_path: str,
    components_count: int = 4,
    k: int = 3,
    init_method: str = "k-means++",
    init_rows_excel_1based: Optional[List[int]] = None,
    n_init: Optional[int] = None,
    max_iter: int = 300,
    random_state: int = 42,
    output_folder: Optional[str] = None,
    output_filename: str = "Hasil_KMeans.xlsx",
    tol: float = 1e-4,
    export_per_data_iterations: bool = True,
    export_max_iter_cols: int = 60
) -> Dict[str, Any]:
    """
    Jalankan KMeans pada PCA1..PCAk.

    init_method:
    - "k-means++" -> init k-means++ (implementasi internal) agar bisa simpan iterasi
    - "manual"    -> centroid awal dari baris tertentu (excel 1-based)

    Output Excel:
    - Data+Cluster (Cluster: C1..Ck)
    - Centroid_Akhir (Cluster: C1..Ck)
    - Jumlah_Anggota (Cluster: C1..Ck)
    - Iterasi
    - (opsional) Per_Data_Iterasi (cluster per iterasi per baris)
    """
    if k < 2:
        raise ValueError("k minimal 2 untuk clustering.")

    X_df, cols = load_pca_components(pca_output_path, components_count)
    X = X_df.values

    if init_method not in ("k-means++", "manual"):
        raise ValueError("init_method harus salah satu: k-means++, manual")

    if init_method == "manual":
        if not init_rows_excel_1based or len(init_rows_excel_1based) != k:
            raise ValueError(f"Untuk init manual, isi {k} nomor baris centroid awal (format Excel, 1-based).")

        init_rows_0 = [int(r) - 1 for r in init_rows_excel_1based]
        for r0 in init_rows_0:
            if r0 < 0 or r0 >= len(X_df):
                raise ValueError(f"Nomor baris centroid di luar range data: {r0+1}")

        init_rows_store = init_rows_excel_1based
        n_init_store = 1 if n_init is None else int(n_init)
    else:
        init_rows_store = None
        n_init_store = 10 if n_init is None else int(n_init)

    best: Optional[Dict[str, Any]] = None

    # pilih run terbaik berdasarkan inertia terendah
    for i in range(int(n_init_store)):
        rs_used = int(random_state) + i

        if init_method == "manual":
            init_centroids = X_df.iloc[[r - 1 for r in init_rows_excel_1based]].values
        else:
            init_centroids = _kmeans_plus_plus_init(X, k=int(k), random_state=rs_used)

        run = _lloyd_kmeans_with_history(
            X=X,
            init_centroids=init_centroids,
            max_iter=int(max_iter),
            tol=float(tol),
        )
        run["random_state_used"] = rs_used

        if best is None or float(run["inertia"]) < float(best["inertia"]):
            best = run

    if best is None:
        raise RuntimeError("KMeans gagal dijalankan (best is None).")

    # ==================================================
    # Label cluster: 0..k-1 -> C1..Ck
    # ==================================================
    cluster_labels = [f"C{i+1}" for i in range(int(k))]
    final_labels_int: List[int] = [int(x) for x in best["labels"]]
    final_labels_str: List[str] = [f"C{l+1}" for l in final_labels_int]

    df_out = X_df.copy()
    df_out.insert(0, "Row", np.arange(1, len(df_out) + 1))
    df_out["Cluster"] = final_labels_str

    centroid_akhir = pd.DataFrame(best["centroids"], columns=cols)
    centroid_akhir.insert(0, "Cluster", cluster_labels)

    counts_arr = np.bincount(np.array(final_labels_int), minlength=int(k))
    counts_by_cluster = {cluster_labels[i]: int(counts_arr[i]) for i in range(int(k))}
    jumlah_anggota_df = pd.DataFrame({
        "Cluster": cluster_labels,
        "Jumlah_Anggota": [counts_by_cluster[c] for c in cluster_labels]
    })

    # iter logs (ubah key counts ke C1..)
    iterations_log: List[Dict[str, Any]] = []
    for it in best["iterations_log"]:
        counts_map = {f"C{int(ci)+1}": int(cv) for ci, cv in it["counts"].items()}
        iterations_log.append({
            "iter": int(it["iter"]),
            "inertia": float(it["inertia"]),
            "max_centroid_shift": float(it["max_centroid_shift"]),
            "counts_by_cluster": counts_map
        })

    labels_by_iteration_str: List[List[str]] = [
        [f"C{int(l)+1}" for l in labels_iter]
        for labels_iter in best["labels_by_iteration"]
    ]
    centroids_by_iteration = best["centroids_by_iteration"]

    # save excel
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, output_filename)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_out.to_excel(writer, sheet_name="Data+Cluster", index=False)
            centroid_akhir.to_excel(writer, sheet_name="Centroid_Akhir", index=False)
            jumlah_anggota_df.to_excel(writer, sheet_name="Jumlah_Anggota", index=False)
            pd.DataFrame(iterations_log).to_excel(writer, sheet_name="Iterasi", index=False)

            if export_per_data_iterations:
                n_iters = len(labels_by_iteration_str)
                use_iters = min(n_iters, int(export_max_iter_cols))

                per_data = pd.DataFrame({
                    "Row": np.arange(1, len(X_df) + 1),
                    "Cluster_Final": final_labels_str
                })
                for t in range(use_iters):
                    per_data[f"Iter_{t+1}"] = labels_by_iteration_str[t]
                per_data.to_excel(writer, sheet_name="Per_Data_Iterasi", index=False)
    else:
        output_path = None

    return {
        "output_path": output_path,
        "k": int(k),
        "used_components": cols,
        "init_method": init_method,
        "init_rows": init_rows_store,
        "n_init": int(n_init_store),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "random_state": int(random_state),
        "random_state_used": int(best.get("random_state_used", random_state)),
        "n_iter": int(best["n_iter"]),
        "centroids_final": centroid_akhir.to_dict(orient="records"),
        "counts_by_cluster": counts_by_cluster,
        "iterations_log": iterations_log,
        "labels_by_iteration": labels_by_iteration_str,
        "centroids_by_iteration": centroids_by_iteration,
    }
