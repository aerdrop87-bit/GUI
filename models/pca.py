# ==========================================
# PCA SERVICE
# ==========================================
# Mengikuti alur Colab:
# - Baca data hasil preprocessing (kolom z_*)
# - Hitung matriks korelasi dan (opsional) kovarians
# - Eigen decomposition
# - Simpan: PCA_Scores, Eigen_Info, Loading_Matrix, Matriks_Korelasi, Matriks_Kovarians
#
# Catatan:
# - Preprocessing sudah menghasilkan kolom berawalan "z_"
# - Jika masih ada NaN, PCA akan gagal -> di sini dibuat pengecekan keras.

from __future__ import annotations

from typing import Dict, Any, List, Optional
import os
import numpy as np
import pandas as pd


def run_pca(
    preprocessed_path: str,
    output_folder: str,
    output_filename: str = "Hasil_PCA_9Komponen.xlsx",
    selected_components_count: int = 4,
    method: str = "correlation",
    include_covariance_sheet: bool = True
) -> Dict[str, Any]:
    """
    Jalankan PCA "manual" berbasis eigen-decomposition seperti di Colab ko.

    Parameters
    ----------
    preprocessed_path : str
        Path ke file excel output preprocessing (punya kolom z_*)
    output_folder : str
        Folder output untuk menyimpan file hasil PCA
    output_filename : str
        Nama file output PCA
    selected_components_count : int
        Jumlah komponen utama yang akan ditandai sebagai "dipakai untuk clustering" (default 4)
    method : str
        "correlation" (default) atau "covariance"
        - correlation: eigen dari matriks korelasi
        - covariance: eigen dari matriks kovarians
    include_covariance_sheet : bool
        Jika True, akan simpan sheet Matriks_Kovarians (selain Matriks_Korelasi)

    Returns
    -------
    Dict[str, Any]
        Metadata untuk dicatat ke database (PCARun)
    """
    if method not in ("correlation", "covariance"):
        raise ValueError("method harus salah satu: correlation, covariance")

    df = pd.read_excel(preprocessed_path)

    # Ambil kolom z_*
    kolom_pca: List[str] = [c for c in df.columns if isinstance(c, str) and c.startswith("z_")]
    if not kolom_pca:
        raise ValueError(
            "Tidak ditemukan kolom berawalan 'z_'. "
            "Pastikan preprocessing sudah menghasilkan kolom z_..."
        )

    X = df[kolom_pca].copy()

    # Pastikan numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        cols_nan = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "Masih ada NaN pada kolom z_ setelah preprocessing: "
            + ", ".join(cols_nan)
            + ". Jalankan preprocessing ulang / perbaiki missing value handling."
        )

    Xv = X.values
    var_names = kolom_pca

    # Matriks korelasi (sesuai Colab)
    R = np.corrcoef(Xv, rowvar=False)

    # Matriks kovarians (opsional)
    C = np.cov(Xv, rowvar=False)

    # Matriks utama untuk eigen decomposition
    M = R if method == "correlation" else C

    # Karena M simetris, gunakan eigh (lebih stabil)
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Urutkan dari terbesar
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real

    total = float(np.sum(eigenvalues))
    if total == 0:
        raise ValueError("Total eigenvalue = 0. Tidak bisa hitung proporsi variansi.")

    proporsi = (eigenvalues / total) * 100.0
    kumulatif = np.cumsum(proporsi)

    eigen_info = pd.DataFrame({
        "Komponen": [f"PCA{i+1}" for i in range(len(eigenvalues))],
        "Eigenvalue": np.round(eigenvalues, 6),
        "Proporsi_Variansi(%)": np.round(proporsi, 6),
        "Kumulatif(%)": np.round(kumulatif, 6),
    })
    
        # Urutkan dari terbesar
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real

    # ==============================
    # FIX SIGN (agar deterministik)
    # tiap kolom eigenvector: pastikan loading terbesar bernilai positif
    # ==============================
    for j in range(eigenvectors.shape[1]):
        col = eigenvectors[:, j]
        i_max = int(np.argmax(np.abs(col)))
        if col[i_max] < 0:
            eigenvectors[:, j] *= -1


    # Skor PCA (projection)
    scores = np.dot(Xv, eigenvectors)
    pca_scores = pd.DataFrame(
        scores,
        columns=[f"PCA{i+1}" for i in range(scores.shape[1])]
    )

    # Loading matrix (mengikuti Colab: eigenvectors langsung)
    loading_matrix = pd.DataFrame(
        eigenvectors,
        columns=[f"PCA{i+1}" for i in range(len(eigenvalues))],
        index=var_names
    )

    # Sheet khusus komponen terpilih (untuk clustering selanjutnya)
    k = int(selected_components_count)
    if k < 1:
        k = 1
    if k > pca_scores.shape[1]:
        k = pca_scores.shape[1]
    pca_selected = pca_scores[[f"PCA{i+1}" for i in range(k)]].copy()

    # Save
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pca_scores.to_excel(writer, sheet_name="PCA_Scores", index=False)
        eigen_info.to_excel(writer, sheet_name="Eigen_Info", index=False)
        loading_matrix.to_excel(writer, sheet_name="Loading_Matrix")

        pd.DataFrame(R, index=var_names, columns=var_names).to_excel(
            writer, sheet_name="Matriks_Korelasi"
        )

        if include_covariance_sheet:
            pd.DataFrame(C, index=var_names, columns=var_names).to_excel(
                writer, sheet_name="Matriks_Kovarians"
            )

        pca_selected.to_excel(writer, sheet_name=f"PCA_{k}", index=False)

    return {
        "output_path": output_path,
        "method": method,
        "n_components_total": int(len(eigenvalues)),
        "selected_components": [i + 1 for i in range(k)],  # [1..k]
        "eigenvalues": [float(x) for x in eigenvalues.tolist()],
        "explained_variance": [float(x) for x in proporsi.tolist()],
        "cumulative_variance": [float(x) for x in kumulatif.tolist()],
        "var_names": var_names,
        "selected_components_count": k,
    }
