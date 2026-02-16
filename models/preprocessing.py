# ==========================================
# PREPROCESSING SERVICE
# ==========================================
# Tujuan:
# - Replikasi alur Colab "standarisasi data" agar hasil downstream (PCA/KMeans/RF) konsisten.
#
# Mode:
# - mode="colab"  : STRICT meniru Colab -> tidak melakukan encoding otomatis & tidak imputasi (kecuali diminta).
#                   Jika ada nilai non-numerik / NaN, akan ERROR (supaya tidak diam-diam berbeda dari Colab).
# - mode="robust" : Lebih "tahan banting" -> bisa encode kategorikal + imputasi (median/mean) seperti versi sebelumnya.
#
# Output:
# - Sheet "Data" selalu ada (ini yang dibaca step berikutnya).
# - Jika debug_sheets=True, akan menambah sheet "Debug_Info", "Scaler_Params", "Debug_Stats".

from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Kolom fitur (sesuai Colab Anda)
FEATURE_COLUMNS: List[str] = [
    'goltarif',
    'wilayah',
    'lama berlangganan',
    'mawal',
    'makhir',
    'pemakaian',
    'minpemakaianair',
    'slip tagihan',
    'jumlah tagihan (Rp.)'
]

# Mapping khusus (hanya dipakai di mode="robust")
KNOWN_CATEGORY_MAPS = {
    "wilayah": {
        "Urban": 0,
        "Suburban": 1,
        "Rural": 2
    },
    "goltarif": {
        "2C": 0,
        "2D": 1,
        "4A": 2
    }
}


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df


def _encode_categorical(series: pd.Series, colname: str) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Encode kolom kategorikal menjadi angka.
    - Jika cocok dengan mapping khusus (KNOWN_CATEGORY_MAPS), pakai itu.
    - Jika tidak, buat mapping deterministik dari sorted unique (stringified).
    """
    s = series.astype("object")
    uniques = [u for u in s.dropna().unique().tolist()]

    if colname in KNOWN_CATEGORY_MAPS:
        m = KNOWN_CATEGORY_MAPS[colname]
        if all(u in m for u in uniques):
            return s.map(m), m

    uniques_sorted = sorted(uniques, key=lambda x: str(x))
    mapping = {u: i for i, u in enumerate(uniques_sorted)}
    return s.map(mapping), mapping


def _coerce_numeric_strict(X: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Colab tidak melakukan encoding/imputasi.
    Jadi jika ada nilai non-numerik, kita FAIL agar user sadar input berbeda dari Colab.
    """
    Xn = X.copy()
    problems: Dict[str, List[str]] = {}

    for col in feature_columns:
        # Coba konversi numeric (string angka -> angka). Kalau gagal, akan jadi NaN.
        before_na = int(Xn[col].isna().sum())
        Xn[col] = pd.to_numeric(Xn[col], errors="coerce")
        after_na = int(Xn[col].isna().sum())

        # NaN bertambah -> ada value yang tidak bisa dikonversi
        if after_na > before_na:
            bad_mask = Xn[col].isna()
            # ambil contoh nilai asli yang gagal (maks 10)
            raw_bad = X.loc[bad_mask, col].dropna().astype(str).unique().tolist()[:10]
            problems[col] = raw_bad

    if problems:
        msg_lines = ["Kolom berikut berisi nilai non-numerik / tidak bisa dikonversi (mode=colab strict):"]
        for col, examples in problems.items():
            ex = ", ".join(examples) if examples else "(contoh tidak tersedia)"
            msg_lines.append(f"- {col}: contoh -> {ex}")
        msg_lines.append("Solusi:")
        msg_lines.append("1) Pastikan file input yang dipakai sama dengan Colab (kolom sudah numerik).")
        msg_lines.append("2) Atau jalankan mode='robust' jika ingin encode/imputasi otomatis.")
        raise ValueError("\n".join(msg_lines))

    # Jika ada NaN asli dari Excel, colab akan error juga (StandardScaler tidak terima NaN)
    if Xn.isna().any().any():
        cols_nan = Xn.columns[Xn.isna().any()].tolist()
        raise ValueError(
            "Masih ada NaN pada kolom fitur (mode=colab). "
            "Di Colab, StandardScaler juga akan gagal jika ada NaN. "
            "Kolom yang masih NaN: " + ", ".join(cols_nan)
        )

    return Xn.astype(float)


def _impute(X: pd.DataFrame, feature_columns: List[str], strategy: str) -> pd.DataFrame:
    if strategy not in ("median", "mean", "none"):
        raise ValueError("imputation_strategy harus salah satu: median, mean, none")

    if strategy == "none":
        # fail-fast jika ada NaN
        if X.isna().any().any():
            cols_nan = X.columns[X.isna().any()].tolist()
            raise ValueError(
                "Terdapat NaN pada data fitur, sementara imputation_strategy='none'. "
                "Kolom yang NaN: " + ", ".join(cols_nan)
            )
        return X

    X2 = X.copy()
    for col in feature_columns:
        if X2[col].isna().any():
            if strategy == "median":
                fill_val = float(X2[col].median(skipna=True))
            else:
                fill_val = float(X2[col].mean(skipna=True))
            X2[col] = X2[col].fillna(fill_val)

    if X2.isna().any().any():
        cols_nan = X2.columns[X2.isna().any()].tolist()
        raise ValueError(
            "Masih ada NaN setelah imputasi. Kemungkinan ada kolom yang seluruhnya NaN / tidak valid: "
            + ", ".join(cols_nan)
        )
    return X2


def run_preprocessing(
    input_path: str,
    output_folder: str,
    output_filename: str = "Data_Standarisasi.xlsx",
    imputation_strategy: str = "none",
    mode: str = "colab",
    z_prefix: str = "z_",
    debug_sheets: bool = True,
) -> Dict[str, Any]:
    """
    Jalankan preprocessing (standarisasi).

    Parameter penting untuk *match Colab*:
    - mode="colab"
    - imputation_strategy="none" (kalau Colab Anda juga tidak imputasi)
    - order feature columns harus sama

    Returns dict:
    - output_path
    - feature_columns
    - imputation_strategy
    - scaler_params
    - n_rows, n_cols
    """

    if mode not in ("colab", "robust"):
        raise ValueError("mode harus salah satu: colab, robust")

    df = pd.read_excel(input_path)
    df = _strip_columns(df)

    # Validasi kolom
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Kolom berikut tidak ditemukan di dataset: " + ", ".join(missing))

    X = df[FEATURE_COLUMNS].copy()

    encoding_maps: Dict[str, Dict[str, int]] = {}

    if mode == "robust":
        # Encode kategorikal bila perlu
        for col in FEATURE_COLUMNS:
            if pd.api.types.is_numeric_dtype(X[col]):
                continue
            encoded, mapping = _encode_categorical(X[col], col)
            X[col] = encoded
            encoding_maps[col] = mapping

        # Coerce numeric
        for col in FEATURE_COLUMNS:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # Impute (median/mean/none)
        X = _impute(X, FEATURE_COLUMNS, imputation_strategy)

    else:
        # mode colab -> strict numeric
        X = _coerce_numeric_strict(X, FEATURE_COLUMNS)
        # imputation optional: jika user set median/mean, boleh, tapi default none
        X = _impute(X, FEATURE_COLUMNS, imputation_strategy)

    # StandardScaler (Z-score)
    scaler = StandardScaler()
    Z = scaler.fit_transform(X.values)

    # Kolom output: z_*
    z_cols = [f"{z_prefix}{k}" for k in FEATURE_COLUMNS]
    df_std = pd.DataFrame(Z, columns=z_cols)

    # Gabung: drop fitur asli -> tambah z_*
    df_final = pd.concat([df.drop(columns=FEATURE_COLUMNS), df_std], axis=1)

    # Save
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    if debug_sheets:
        # ringkasan
        dbg_info = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "dtype_raw": [str(df[c].dtype) for c in FEATURE_COLUMNS],
        })
        scaler_params_df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "mean": scaler.mean_.astype(float),
            "scale": scaler.scale_.astype(float),
        })
        stats = []
        for c in z_cols:
            s = df_std[c]
            stats.append({
                "col": c,
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)),  # ddof=0 agar konsisten dengan StandardScaler
                "min": float(s.min()),
                "max": float(s.max()),
            })
        dbg_stats = pd.DataFrame(stats)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_final.to_excel(writer, sheet_name="Data", index=False)
            dbg_info.to_excel(writer, sheet_name="Debug_Info", index=False)
            scaler_params_df.to_excel(writer, sheet_name="Scaler_Params", index=False)
            dbg_stats.to_excel(writer, sheet_name="Debug_Stats", index=False)
    else:
        df_final.to_excel(output_path, index=False)

    scaler_params = {
        "mean": [float(x) for x in scaler.mean_.tolist()],
        "scale": [float(x) for x in scaler.scale_.tolist()],
        "encoding_maps": encoding_maps,
        "feature_columns": FEATURE_COLUMNS,
        "mode": mode,
        "imputation_strategy": imputation_strategy,
        "z_prefix": z_prefix,
        "debug_sheets": bool(debug_sheets),
    }

    return {
        "output_path": output_path,
        "feature_columns": FEATURE_COLUMNS,
        "imputation_strategy": imputation_strategy,
        "scaler_params": scaler_params,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
    }
