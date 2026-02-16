# ==========================================
# RANDOM FOREST (MANUAL RULES) SERVICE
# ==========================================
# Mengikuti Colab (RF manual, bukan sklearn RandomForestClassifier):
# - Input: Data berisi PCA1..PCA4 + label aktual (hasil KMeans)
# - Terapkan 5 aturan "tree" berbasis ambang PCA
# - Voting mayoritas (mode) -> prediksi final
# - Evaluasi: classification_report, accuracy, macro-F1
#
# Kenapa hasil bisa beda dari Colab?
# 1) Label cluster KMeans (0/1/2) bisa tertukar urutannya dibanding "C1/C2/C3" di Colab.
# 2) Tanda (sign) komponen PCA bisa terbalik (+/-) karena eigenvector punya ambiguitas tanda.
#    (Secara matematis tetap benar, tapi rules berbasis ambang akan berubah kalau tanda terbalik.)
#
# Solusi:
# - Ada mode "auto alignment" yang mencari kombinasi terbaik:
#   * Permutasi mapping Cluster -> (C1,C2,C3)  (3! = 6)
#   * Flip tanda PCA1..PCA4                    (2^4 = 16)
#   Total 96 kombinasi, kecil dan cepat.
#   Dipilih kombinasi dengan akurasi paling tinggi di seluruh data (sebelum split).
#
# - Ada 2 mode split:
#   * "colab_like": shuffle + ambil N pertama (mirip membuat Data Training.xlsx / Data Testing.xlsx)
#   * "stratified": train_test_split(..., stratify=y) (lebih stabil distribusi label)

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product, permutations
from statistics import mode, StatisticsError
from typing import Any, Dict, List, Optional, Tuple

import re

import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


# ===== Default ambang dari Colab =====
DEFAULT_AMBANG: Dict[str, Dict[str, float]] = {
    "PCA1": {"low": -0.471222, "mid": 0.740738},
    "PCA2": {"low": -0.496107, "mid": 0.799846},
    "PCA3": {"low": -0.143652, "mid": 0.300361},
    "PCA4": {"low": -0.206872, "mid": 0.633076},
}


# ===== Helpers =====
def _as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def normalize_label_to_c(x) -> str:
    """
    Normalisasi label ke format 'C1','C2','C3' jika memungkinkan.
    - angka 0/1/2 -> C1/C2/C3
    - string '0'/'1' -> C1/C2
    - string 'C1'/'c1' -> C1
    """
    if x is None:
        return ""
    if isinstance(x, (int, float)) and not pd.isna(x):
        xi = int(round(float(x)))
        return f"C{xi + 1}"
    s = str(x).strip()
    if not s:
        return ""
    if s.isdigit():
        return f"C{int(s) + 1}"
    if s.upper().startswith("C"):
        tail = s[1:].strip()
        return f"C{tail}" if tail else "C"
    return s


def validate_ambang(ambang: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Validasi struktur ambang JSON agar sesuai dan cast ke float."""
    required = ["PCA1", "PCA2", "PCA3", "PCA4"]
    out: Dict[str, Dict[str, float]] = {}
    for k in required:
        if k not in ambang:
            raise ValueError(f"Ambang harus punya key '{k}'")
        if not isinstance(ambang[k], dict) or "low" not in ambang[k] or "mid" not in ambang[k]:
            raise ValueError(f"Ambang '{k}' harus punya subkey 'low' dan 'mid'")
        low = float(ambang[k]["low"])
        mid = float(ambang[k]["mid"])
        if low > mid:
            raise ValueError(f"Ambang '{k}' tidak valid: low ({low}) harus <= mid ({mid})")
        out[k] = {"low": low, "mid": mid}
    return out


def _ensure_pca_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError("Kolom PCA tidak lengkap: " + ", ".join(missing))
    for c in cols:
        df[c] = df[c].apply(_as_float)
    if df[cols].isna().any().any():
        bad = df[cols].columns[df[cols].isna().any()].tolist()
        raise ValueError("Kolom PCA berisi NaN / tidak numerik: " + ", ".join(bad))


# ===== Kategori PCA (sesuai Colab) =====
def kategori_pca(nilai: float, batas: Dict[str, float]) -> str:
    if nilai <= batas["low"]:
        return "Rendah"
    elif nilai <= batas["mid"]:
        return "Sedang"
    else:
        return "Tinggi"


# ===== 5 pohon (sesuai Colab) =====
def tree1(row, ambang):
    p1 = kategori_pca(row["PCA1"], ambang["PCA1"])
    p2 = kategori_pca(row["PCA2"], ambang["PCA2"])
    p4 = kategori_pca(row["PCA4"], ambang["PCA4"])
    if p1 == "Tinggi" and p2 == "Tinggi":
        return "C1"
    elif p1 == "Sedang" and p4 == "Rendah":
        return "C2"
    else:
        return "C3"


def tree2(row, ambang):
    p4 = kategori_pca(row["PCA4"], ambang["PCA4"])
    p1 = kategori_pca(row["PCA1"], ambang["PCA1"])
    p3 = kategori_pca(row["PCA3"], ambang["PCA3"])
    if p4 == "Tinggi" and p3 in ["Sedang", "Tinggi"]:
        return "C3"
    elif p4 == "Rendah" and p1 == "Sedang":
        return "C2"
    else:
        return "C1"


def tree3(row, ambang):
    p2 = kategori_pca(row["PCA2"], ambang["PCA2"])
    p1 = kategori_pca(row["PCA1"], ambang["PCA1"])
    p3 = kategori_pca(row["PCA3"], ambang["PCA3"])
    if p2 == "Tinggi" and p1 != "Rendah":
        return "C1"
    elif p2 == "Sedang" and p3 == "Sedang":
        return "C2"
    else:
        return "C3"


def tree4(row, ambang):
    p3 = kategori_pca(row["PCA3"], ambang["PCA3"])
    p4 = kategori_pca(row["PCA4"], ambang["PCA4"])
    if p3 == "Tinggi" and p4 == "Tinggi":
        return "C1"
    elif p3 == "Sedang" and p4 == "Rendah":
        return "C2"
    else:
        return "C3"


def tree5(row, ambang):
    p1 = kategori_pca(row["PCA1"], ambang["PCA1"])
    p2 = kategori_pca(row["PCA2"], ambang["PCA2"])
    p4 = kategori_pca(row["PCA4"], ambang["PCA4"])
    if p1 == "Tinggi" and p4 == "Tinggi":
        return "C1"
    elif p1 == "Sedang" and p2 == "Sedang":
        return "C2"
    else:
        return "C3"


# ===== Voting mayoritas (meniru statistics.mode di Colab) =====
def majority_vote_colab(values: List[str]) -> str:
    """
    Meniru perilaku mode() pada list.
    Jika terjadi tie (mis. C1=2, C2=2, C3=1), pilih label yang muncul paling awal di list.
    """
    if not values:
        return ""
    try:
        return mode(values)
    except StatisticsError:
        c = Counter(values)
        top = max(c.values())
        for v in values:
            if c[v] == top:
                return v
        return values[0]


def apply_rf_manual(df: pd.DataFrame, ambang: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Tambahkan kolom Tree1..Tree5 dan Voting Mayoritas."""
    cols = ["PCA1", "PCA2", "PCA3", "PCA4"]
    _ensure_pca_columns(df, cols)

    out = df.copy()
    out["Tree1"] = out.apply(lambda r: tree1(r, ambang), axis=1)
    out["Tree2"] = out.apply(lambda r: tree2(r, ambang), axis=1)
    out["Tree3"] = out.apply(lambda r: tree3(r, ambang), axis=1)
    out["Tree4"] = out.apply(lambda r: tree4(r, ambang), axis=1)
    out["Tree5"] = out.apply(lambda r: tree5(r, ambang), axis=1)
    out["Voting Mayoritas"] = out[["Tree1", "Tree2", "Tree3", "Tree4", "Tree5"]].apply(
        lambda x: majority_vote_colab(list(x.values)), axis=1
    )
    return out


def evaluate_predictions(
    df_with_pred: pd.DataFrame,
    actual_col: str = "Cluster Aktual",
    pred_col: str = "Voting Mayoritas",
    labels: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], float, float]:
    """Return (classification_report dict, accuracy, macro_f1)."""
    if actual_col not in df_with_pred.columns:
        raise ValueError(f"Kolom label aktual '{actual_col}' tidak ditemukan.")
    if pred_col not in df_with_pred.columns:
        raise ValueError(f"Kolom prediksi '{pred_col}' tidak ditemukan.")

    y_true = df_with_pred[actual_col].tolist()
    y_pred = df_with_pred[pred_col].tolist()

    if labels is None:
        labels = ["C1", "C2", "C3"]

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    # JSON-serializable
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, (int, float, str)) or o is None:
            return float(o) if isinstance(o, float) else o
        return str(o)

    return _clean(report), acc, macro


def load_excel_any_sheet(path: str, prefer_sheets: List[str]) -> pd.DataFrame:
    """Load Excel dari sheet yang cocok, fallback ke sheet pertama."""
    xls = pd.ExcelFile(path)
    for s in prefer_sheets:
        if s in xls.sheet_names:
            return pd.read_excel(path, sheet_name=s)
    return pd.read_excel(path, sheet_name=xls.sheet_names[0])


def build_rf_dataframe_from_kmeans_output(kmeans_output_path: str) -> pd.DataFrame:
    """
    Ambil data dari output KMeans:
    - Minimal harus ada PCA1..PCA4
    - Label aktual bisa bernama 'Cluster Aktual' atau 'Cluster'
    Output:
    - PCA1..PCA4
    - Cluster Raw (label asli dari file)
    """
    df = load_excel_any_sheet(
        kmeans_output_path,
        prefer_sheets=["Data+Cluster", "Data_Cluster", "Data", "Sheet1"],
    )

    if "Cluster Aktual" in df.columns:
        label_col = "Cluster Aktual"
    elif "Cluster" in df.columns:
        label_col = "Cluster"
    else:
        raise ValueError("Tidak ditemukan kolom label cluster ('Cluster' atau 'Cluster Aktual') pada file KMeans.")

    needed = ["PCA1", "PCA2", "PCA3", "PCA4"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Kolom {c} tidak ditemukan pada data KMeans output.")

    out = df[needed + [label_col]].copy()
    out.rename(columns={label_col: "Cluster Raw"}, inplace=True)
    return out


# ===== Alignment (mapping cluster + sign flip PCA) =====
@dataclass(frozen=True)
class AlignmentResult:
    mapping: Dict[Any, str]                 # raw_cluster_value -> "C1/C2/C3"
    sign: Dict[str, int]                    # "PCA1".. -> +1/-1
    score_accuracy: float                   # accuracy pada full data
    score_macro_f1: float                   # macro-F1 pada full data
    note: str                               # keterangan singkat


def _unique_sorted_cluster_ids(values: List[Any]) -> List[Any]:
    # coba urut numerik jika mungkin
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        cleaned.append(v)
    uniq = list(dict.fromkeys(cleaned))
    # jika semua bisa jadi int, urutkan
    try:
        ints = [int(round(float(x))) for x in uniq]
        # pastikan tidak ada bentrok
        if len(set(ints)) == len(ints):
            order = [x for _, x in sorted(zip(ints, uniq), key=lambda t: t[0])]
            return order
    except Exception:
        pass
    return sorted(uniq, key=lambda x: str(x))


def infer_best_alignment(
    df_raw: pd.DataFrame,
    ambang: Dict[str, Dict[str, float]],
    labels_target: List[str] = ["C1", "C2", "C3"],
) -> AlignmentResult:
    """
    Cari mapping Cluster->C* dan sign flip PCA1..PCA4 terbaik berdasarkan macro-F1 terhadap rule RF.

    Catatan:
    - Walaupun label dari KMeans sudah berbentuk 'C1/C2/C3', itu tetap bisa tertukar maknanya.
      Maka auto alignment tetap perlu mencoba permutasi mapping.
    """
    df = df_raw.copy()

    # pastikan PCA numeric
    cols = ["PCA1", "PCA2", "PCA3", "PCA4"]
    _ensure_pca_columns(df, cols)

    if "Cluster Raw" not in df.columns:
        raise ValueError("Kolom 'Cluster Raw' tidak ada. Gunakan build_rf_dataframe_from_kmeans_output().")

    raw_vals = df["Cluster Raw"].tolist()

    # ============================
    # sebelum
    # ============================
    # raw_str = [str(v).strip() for v in raw_vals if not pd.isna(v)]
    # looks_like_c = bool(raw_str) and all(re.match(r"^[cC]\s*\d+$", s) for s in raw_str)
    # if looks_like_c:
    #     mapping = {v: normalize_label_to_c(v) for v in _unique_sorted_cluster_ids(raw_vals)}
    #     sign = {c: 1 for c in cols}
    #     df2 = df.copy()
    #     df2["Cluster Aktual"] = [mapping[v] for v in df2["Cluster Raw"]]
    #     pred = apply_rf_manual(df2, ambang)["Voting Mayoritas"]
    #     acc = float(accuracy_score(df2["Cluster Aktual"], pred))
    #     return AlignmentResult(mapping=mapping, sign=sign, score_accuracy=acc, note="Label sudah C1/C2/C3 (tanpa remap & tanpa flip).")

    # ============================
    # sesudah
    # ============================
    # Tidak lagi melakukan shortcut ketika label sudah 'C1/C2/C3'.
    # Alasannya: label output KMeans bersifat arbitrer (bisa tertukar urutannya),
    # sehingga auto alignment tetap harus mencoba permutasi mapping.

    cluster_ids = _unique_sorted_cluster_ids(raw_vals)
    if len(cluster_ids) != 3:
        raise ValueError(
            f"Auto alignment butuh tepat 3 label cluster unik, ditemukan {len(cluster_ids)}: {cluster_ids}"
        )

    best: Optional[AlignmentResult] = None

    # precompute base df untuk speed
    base = df[cols].copy()

    for signs in product([1, -1], repeat=4):
        df_s = base.copy()
        sign_map = {col: int(s) for col, s in zip(cols, signs)}
        for col in cols:
            if sign_map[col] == -1:
                df_s[col] = -df_s[col]

        # apply rules once per sign
        tmp = df_s.copy()
        tmp["Cluster Raw"] = df["Cluster Raw"].values
        pred = apply_rf_manual(tmp, ambang)["Voting Mayoritas"].tolist()

        for perm in permutations(labels_target, 3):
            mapping = {cid: lbl for cid, lbl in zip(cluster_ids, perm)}
            y_true = [mapping[v] for v in raw_vals]
            acc = float(accuracy_score(y_true, pred))
            macro = float(f1_score(y_true, pred, labels=labels_target, average="macro", zero_division=0))

            # tie-break deterministik: macro-F1 desc, accuracy desc, jumlah flip asc, mapping string asc
            flips = sum(1 for s in signs if s == -1)
            mapping_key = "|".join([f"{str(k)}->{v}" for k, v in sorted(mapping.items(), key=lambda t: str(t[0]))])

            if best is None:
                best = AlignmentResult(
                    mapping=mapping,
                    sign=sign_map,
                    score_accuracy=acc,
                    score_macro_f1=macro,
                    note=mapping_key,
                )
            else:
                best_flips = sum(1 for v in best.sign.values() if v == -1)
                best_key = best.note
                if (macro > best.score_macro_f1) or (
                    macro == best.score_macro_f1 and acc > best.score_accuracy
                ) or (
                    macro == best.score_macro_f1 and acc == best.score_accuracy and flips < best_flips
                ) or (
                    macro == best.score_macro_f1 and acc == best.score_accuracy and flips == best_flips and mapping_key < best_key
                ):
                    best = AlignmentResult(
                        mapping=mapping,
                        sign=sign_map,
                        score_accuracy=acc,
                        score_macro_f1=macro,
                        note=mapping_key,
                    )

    assert best is not None
    return AlignmentResult(
        mapping=best.mapping,
        sign=best.sign,
        score_accuracy=best.score_accuracy,
        score_macro_f1=best.score_macro_f1,
        note=(
            "Auto alignment terbaik "
            f"(full-data macro-F1={best.score_macro_f1:.4f}, acc={best.score_accuracy:.4f}): "
            f"{best.note} | sign={best.sign}"
        ),
    )


def apply_alignment(df_raw: pd.DataFrame, alignment: AlignmentResult) -> pd.DataFrame:
    """Terapkan mapping & sign flip ke dataframe."""
    df = df_raw.copy()
    cols = ["PCA1", "PCA2", "PCA3", "PCA4"]
    _ensure_pca_columns(df, cols)

    # sign flip
    for c in cols:
        if alignment.sign.get(c, 1) == -1:
            df[c] = -df[c]

    # mapping label
    df["Cluster Aktual"] = df["Cluster Raw"].map(alignment.mapping)
    if df["Cluster Aktual"].isna().any():
        raise ValueError("Gagal mapping label cluster ke C1/C2/C3. Periksa mapping alignment.")
    return df


# ===== Split =====
def split_train_test_colab_like(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Meniru praktik Colab yang biasanya:
    - Data diacak dulu
    - Ambil 260 sebagai training (80% dari 325), sisanya testing
    """
    if not (0.05 < float(train_ratio) < 0.95):
        raise ValueError("train_ratio harus di antara 0.05 dan 0.95.")
    shuffled = df.sample(frac=1, random_state=int(random_state)).reset_index(drop=True)
    n_train = int(round(len(shuffled) * float(train_ratio)))
    n_train = max(1, min(n_train, len(shuffled) - 1))
    df_train = shuffled.iloc[:n_train].reset_index(drop=True)
    df_test = shuffled.iloc[n_train:].reset_index(drop=True)
    return df_train, df_test


def split_train_test_stratified(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split stratified agar distribusi label relatif stabil."""
    if "Cluster Aktual" not in df.columns:
        raise ValueError("Kolom 'Cluster Aktual' wajib ada untuk split stratified.")
    if not (0.05 < float(train_ratio) < 0.95):
        raise ValueError("train_ratio harus di antara 0.05 dan 0.95.")
    y = df["Cluster Aktual"]
    df_train, df_test = train_test_split(
        df,
        train_size=float(train_ratio),
        random_state=int(random_state),
        shuffle=True,
        stratify=y,
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


# ===== Save helpers =====
def save_dataframe_excel(df: pd.DataFrame, output_path: str, sheet_name: str = "Data") -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=sheet_name, index=False)
    return output_path


def save_evaluation_excel(
    df_pred: pd.DataFrame,
    report_dict: Dict[str, Any],
    output_path: str,
    meta: Optional[Dict[str, Any]] = None,
    title_sheet: str = "Prediksi",
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # rapikan kolom seperti Colab (precision, recall, f1-score, support)
    keep = [c for c in ["precision", "recall", "f1-score", "support"] if c in report_df.columns]
    report_view = report_df[keep] if keep else report_df

    with pd.ExcelWriter(output_path, engine="openpyxl") as w:
        df_pred.to_excel(w, sheet_name=title_sheet, index=False)
        report_view.to_excel(w, sheet_name="Evaluasi")
        if meta:
            pd.DataFrame([meta]).to_excel(w, sheet_name="Meta", index=False)
    return output_path


# ===== Main pipeline =====
def run_rf_manual_evaluation(
    kmeans_output_path: str,
    output_folder: str,
    ambang: Optional[Dict[str, Dict[str, float]]] = None,
    train_ratio: float = 0.8,
    random_state: int = 42,
    labels: Optional[List[str]] = None,
    split_method: str = "colab_like",        # "colab_like" | "stratified"
    alignment_mode: str = "auto",            # "auto" | "none"
) -> Dict[str, Any]:
    """
    End-to-end:
    - load data PCA+Cluster dari output KMeans
    - (opsional) auto-align: mapping cluster & sign flip PCA
    - split train/test
    - jalankan RF manual + evaluasi untuk train & test
    - simpan file train, test, report train, report test

    Return dict siap disimpan ke DB/controller.
    """
    if ambang is None:
        ambang = DEFAULT_AMBANG
    ambang = validate_ambang(ambang)

    if labels is None:
        labels = ["C1", "C2", "C3"]

    df_raw = build_rf_dataframe_from_kmeans_output(kmeans_output_path)

    # alignment
    if alignment_mode not in ("auto", "none"):
        raise ValueError("alignment_mode harus 'auto' atau 'none'")

    if alignment_mode == "auto":
        alignment = infer_best_alignment(df_raw, ambang, labels_target=labels)
    else:
        # tanpa auto: hanya coba normalize ke C* (kalau dari file sudah C1/C2/C3)
        mapping = {v: normalize_label_to_c(v) for v in _unique_sorted_cluster_ids(df_raw["Cluster Raw"].tolist())}
        alignment = AlignmentResult(
            mapping=mapping,
            sign={c: 1 for c in ["PCA1", "PCA2", "PCA3", "PCA4"]},
            score_accuracy=0.0,
            score_macro_f1=0.0,
            note="alignment_mode=none (tanpa auto mapping/sign flip).",
        )

    df_all = apply_alignment(df_raw, alignment)

    # split
    if split_method not in ("colab_like", "stratified"):
        raise ValueError("split_method harus 'colab_like' atau 'stratified'")

    if split_method == "colab_like":
        df_train, df_test = split_train_test_colab_like(df_all, train_ratio=train_ratio, random_state=random_state)
    else:
        df_train, df_test = split_train_test_stratified(df_all, train_ratio=train_ratio, random_state=random_state)

    # train eval
    pred_train = apply_rf_manual(df_train, ambang)
    report_train, acc_train, macro_train = evaluate_predictions(pred_train, labels=labels)

    # test eval
    pred_test = apply_rf_manual(df_test, ambang)
    report_test, acc_test, macro_test = evaluate_predictions(pred_test, labels=labels)

    os.makedirs(output_folder, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

    train_path = os.path.join(output_folder, f"Data_Train_{ts}.xlsx")
    test_path = os.path.join(output_folder, f"Data_Test_{ts}.xlsx")
    report_train_path = os.path.join(output_folder, f"RF_Evaluasi_Train_{ts}.xlsx")
    report_test_path = os.path.join(output_folder, f"RF_Evaluasi_Test_{ts}.xlsx")

    # simpan split (berisi Cluster Raw + Cluster Aktual, sudah align)
    save_dataframe_excel(df_train, train_path, sheet_name="Train")
    save_dataframe_excel(df_test, test_path, sheet_name="Test")

    meta = {
        "kmeans_output_path": kmeans_output_path,
        "train_ratio": float(train_ratio),
        "random_state": int(random_state),
        "split_method": split_method,
        "alignment_mode": alignment_mode,
        "alignment_note": alignment.note,
        "alignment_full_data_accuracy": float(alignment.score_accuracy),
        "alignment_full_data_macro_f1": float(alignment.score_macro_f1),
        "mapping": str(alignment.mapping),
        "sign": str(alignment.sign),
    }

    save_evaluation_excel(pred_train, report_train, report_train_path, meta=meta, title_sheet="Train_Prediksi")
    save_evaluation_excel(pred_test, report_test, report_test_path, meta=meta, title_sheet="Test_Prediksi")

    return {
        "train_path": train_path,
        "test_path": test_path,
        "alignment": {
            "mapping": alignment.mapping,
            "sign": alignment.sign,
            "full_data_accuracy": alignment.score_accuracy,
            "full_data_macro_f1": alignment.score_macro_f1,
            "note": alignment.note,
        },
        "train": {
            "report": report_train,
            "accuracy": acc_train,
            "macro_f1": macro_train,
            "output_path": report_train_path,
        },
        "test": {
            "report": report_test,
            "accuracy": acc_test,
            "macro_f1": macro_test,
            "output_path": report_test_path,
        },
        "ambang": ambang,
        "random_state": int(random_state),
        "train_ratio": float(train_ratio),
        "split_method": split_method,
        "alignment_mode": alignment_mode,
    }

# =========================
# BATAS
# ======================
# ==========================================
# RANDOM FOREST (MANUAL RULES) SERVICE
# ==========================================
# Mengikuti Colab (RF manual, bukan sklearn RandomForestClassifier):
# - Input: Data berisi PCA1..PCA4 + label aktual (hasil KMeans)
# - Terapkan 5 aturan "tree" berbasis ambang PCA
# - Voting mayoritas (mode) -> prediksi final
# - Evaluasi: classification_report, accuracy, macro-F1
#
# Kenapa hasil bisa beda dari Colab?
# 1) Label cluster KMeans (0/1/2) bisa tertukar urutannya dibanding "C1/C2/C3" di Colab.
# 2) Tanda (sign) komponen PCA bisa terbalik (+/-) karena eigenvector punya ambiguitas tanda.
#    (Secara matematis tetap benar, tapi rules berbasis ambang akan berubah kalau tanda terbalik.)
#
# Solusi:
# - Ada mode "auto alignment" yang mencari kombinasi terbaik:
#   * Permutasi mapping Cluster -> (C1,C2,C3)  (3! = 6)
#   * Flip tanda PCA1..PCA4                    (2^4 = 16)
#   Total 96 kombinasi, kecil dan cepat.
#   Dipilih kombinasi dengan akurasi paling tinggi di seluruh data (sebelum split).
#
# - Ada 2 mode split:
#   * "colab_like": shuffle + ambil N pertama (mirip membuat Data Training.xlsx / Data Testing.xlsx)
#   * "stratified": train_test_split(..., stratify=y) (lebih stabil distribusi label)

# from __future__ import annotations

# from collections import Counter
# from dataclasses import dataclass
# from itertools import product, permutations
# from statistics import mode, StatisticsError
# from typing import Any, Dict, List, Optional, Tuple

# import re

# import os
# import pandas as pd
# from sklearn.metrics import accuracy_score, classification_report, f1_score
# from sklearn.model_selection import train_test_split


# # ===== Default ambang dari Colab =====
# DEFAULT_AMBANG: Dict[str, Dict[str, float]] = {
#     "PCA1": {"low": -0.471222, "mid": 0.740738},
#     "PCA2": {"low": -0.496107, "mid": 0.799846},
#     "PCA3": {"low": -0.143652, "mid": 0.300361},
#     "PCA4": {"low": -0.206872, "mid": 0.633076},
# }


# # ===== Helpers =====
# def _as_float(x) -> float:
#     try:
#         return float(x)
#     except Exception:
#         return float("nan")


# def normalize_label_to_c(x) -> str:
#     """
#     Normalisasi label ke format 'C1','C2','C3' jika memungkinkan.
#     - angka 0/1/2 -> C1/C2/C3
#     - string '0'/'1' -> C1/C2
#     - string 'C1'/'c1' -> C1
#     """
#     if x is None:
#         return ""
#     if isinstance(x, (int, float)) and not pd.isna(x):
#         xi = int(round(float(x)))
#         return f"C{xi + 1}"
#     s = str(x).strip()
#     if not s:
#         return ""
#     if s.isdigit():
#         return f"C{int(s) + 1}"
#     if s.upper().startswith("C"):
#         tail = s[1:].strip()
#         return f"C{tail}" if tail else "C"
#     return s


# def validate_ambang(ambang: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
#     """Validasi struktur ambang JSON agar sesuai dan cast ke float."""
#     required = ["PCA1", "PCA2", "PCA3", "PCA4"]
#     out: Dict[str, Dict[str, float]] = {}
#     for k in required:
#         if k not in ambang:
#             raise ValueError(f"Ambang harus punya key '{k}'")
#         if not isinstance(ambang[k], dict) or "low" not in ambang[k] or "mid" not in ambang[k]:
#             raise ValueError(f"Ambang '{k}' harus punya subkey 'low' dan 'mid'")
#         low = float(ambang[k]["low"])
#         mid = float(ambang[k]["mid"])
#         if low > mid:
#             raise ValueError(f"Ambang '{k}' tidak valid: low ({low}) harus <= mid ({mid})")
#         out[k] = {"low": low, "mid": mid}
#     return out


# def _ensure_pca_columns(df: pd.DataFrame, cols: List[str]) -> None:
#     missing = [c for c in cols if c not in df.columns]
#     if missing:
#         raise ValueError("Kolom PCA tidak lengkap: " + ", ".join(missing))
#     for c in cols:
#         df[c] = df[c].apply(_as_float)
#     if df[cols].isna().any().any():
#         bad = df[cols].columns[df[cols].isna().any()].tolist()
#         raise ValueError("Kolom PCA berisi NaN / tidak numerik: " + ", ".join(bad))


# # ===== Kategori PCA (sesuai Colab) =====
# def kategori_pca(nilai: float, batas: Dict[str, float]) -> str:
#     if nilai <= batas["low"]:
#         return "Rendah"
#     elif nilai <= batas["mid"]:
#         return "Sedang"
#     else:
#         return "Tinggi"


# # ===== 5 pohon (sesuai Colab) =====
# def tree1(row, ambang):
#     p1 = kategori_pca(row["PCA1"], ambang["PCA1"])
#     p2 = kategori_pca(row["PCA2"], ambang["PCA2"])
#     p4 = kategori_pca(row["PCA4"], ambang["PCA4"])
#     if p1 == "Tinggi" and p2 == "Tinggi":
#         return "C1"
#     elif p1 == "Sedang" and p4 == "Rendah":
#         return "C2"
#     else:
#         return "C3"


# def tree2(row, ambang):
#     p4 = kategori_pca(row["PCA4"], ambang["PCA4"])
#     p1 = kategori_pca(row["PCA1"], ambang["PCA1"])
#     p3 = kategori_pca(row["PCA3"], ambang["PCA3"])
#     if p4 == "Tinggi" and p3 in ["Sedang", "Tinggi"]:
#         return "C3"
#     elif p4 == "Rendah" and p1 == "Sedang":
#         return "C2"
#     else:
#         return "C1"


# def tree3(row, ambang):
#     p2 = kategori_pca(row["PCA2"], ambang["PCA2"])
#     p1 = kategori_pca(row["PCA1"], ambang["PCA1"])
#     p3 = kategori_pca(row["PCA3"], ambang["PCA3"])
#     if p2 == "Tinggi" and p1 != "Rendah":
#         return "C1"
#     elif p2 == "Sedang" and p3 == "Sedang":
#         return "C2"
#     else:
#         return "C3"


# def tree4(row, ambang):
#     p3 = kategori_pca(row["PCA3"], ambang["PCA3"])
#     p4 = kategori_pca(row["PCA4"], ambang["PCA4"])
#     if p3 == "Tinggi" and p4 == "Tinggi":
#         return "C1"
#     elif p3 == "Sedang" and p4 == "Rendah":
#         return "C2"
#     else:
#         return "C3"


# def tree5(row, ambang):
#     p1 = kategori_pca(row["PCA1"], ambang["PCA1"])
#     p2 = kategori_pca(row["PCA2"], ambang["PCA2"])
#     p4 = kategori_pca(row["PCA4"], ambang["PCA4"])
#     if p1 == "Tinggi" and p4 == "Tinggi":
#         return "C1"
#     elif p1 == "Sedang" and p2 == "Sedang":
#         return "C2"
#     else:
#         return "C3"


# # ===== Voting mayoritas (meniru statistics.mode di Colab) =====
# def majority_vote_colab(values: List[str]) -> str:
#     """
#     Meniru perilaku mode() pada list.
#     Jika terjadi tie (mis. C1=2, C2=2, C3=1), pilih label yang muncul paling awal di list.
#     """
#     if not values:
#         return ""
#     try:
#         return mode(values)
#     except StatisticsError:
#         c = Counter(values)
#         top = max(c.values())
#         for v in values:
#             if c[v] == top:
#                 return v
#         return values[0]


# def apply_rf_manual(df: pd.DataFrame, ambang: Dict[str, Dict[str, float]]) -> pd.DataFrame:
#     """Tambahkan kolom Tree1..Tree5 dan Voting Mayoritas."""
#     cols = ["PCA1", "PCA2", "PCA3", "PCA4"]
#     _ensure_pca_columns(df, cols)

#     out = df.copy()
#     out["Tree1"] = out.apply(lambda r: tree1(r, ambang), axis=1)
#     out["Tree2"] = out.apply(lambda r: tree2(r, ambang), axis=1)
#     out["Tree3"] = out.apply(lambda r: tree3(r, ambang), axis=1)
#     out["Tree4"] = out.apply(lambda r: tree4(r, ambang), axis=1)
#     out["Tree5"] = out.apply(lambda r: tree5(r, ambang), axis=1)
#     out["Voting Mayoritas"] = out[["Tree1", "Tree2", "Tree3", "Tree4", "Tree5"]].apply(
#         lambda x: majority_vote_colab(list(x.values)), axis=1
#     )
#     return out


# def evaluate_predictions(
#     df_with_pred: pd.DataFrame,
#     actual_col: str = "Cluster Aktual",
#     pred_col: str = "Voting Mayoritas",
#     labels: Optional[List[str]] = None,
# ) -> Tuple[Dict[str, Any], float, float]:
#     """Return (classification_report dict, accuracy, macro_f1)."""
#     if actual_col not in df_with_pred.columns:
#         raise ValueError(f"Kolom label aktual '{actual_col}' tidak ditemukan.")
#     if pred_col not in df_with_pred.columns:
#         raise ValueError(f"Kolom prediksi '{pred_col}' tidak ditemukan.")

#     y_true = df_with_pred[actual_col].tolist()
#     y_pred = df_with_pred[pred_col].tolist()

#     if labels is None:
#         labels = ["C1", "C2", "C3"]

#     report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
#     acc = float(accuracy_score(y_true, y_pred))
#     macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

#     # JSON-serializable
#     def _clean(o):
#         if isinstance(o, dict):
#             return {k: _clean(v) for k, v in o.items()}
#         if isinstance(o, list):
#             return [_clean(v) for v in o]
#         if isinstance(o, (int, float, str)) or o is None:
#             return float(o) if isinstance(o, float) else o
#         return str(o)

#     return _clean(report), acc, macro


# def load_excel_any_sheet(path: str, prefer_sheets: List[str]) -> pd.DataFrame:
#     """Load Excel dari sheet yang cocok, fallback ke sheet pertama."""
#     xls = pd.ExcelFile(path)
#     for s in prefer_sheets:
#         if s in xls.sheet_names:
#             return pd.read_excel(path, sheet_name=s)
#     return pd.read_excel(path, sheet_name=xls.sheet_names[0])


# def build_rf_dataframe_from_kmeans_output(kmeans_output_path: str) -> pd.DataFrame:
#     """
#     Ambil data dari output KMeans:
#     - Minimal harus ada PCA1..PCA4
#     - Label aktual bisa bernama 'Cluster Aktual' atau 'Cluster'
#     Output:
#     - PCA1..PCA4
#     - Cluster Raw (label asli dari file)
#     """
#     df = load_excel_any_sheet(
#         kmeans_output_path,
#         prefer_sheets=["Data+Cluster", "Data_Cluster", "Data", "Sheet1"],
#     )

#     if "Cluster Aktual" in df.columns:
#         label_col = "Cluster Aktual"
#     elif "Cluster" in df.columns:
#         label_col = "Cluster"
#     else:
#         raise ValueError("Tidak ditemukan kolom label cluster ('Cluster' atau 'Cluster Aktual') pada file KMeans.")

#     needed = ["PCA1", "PCA2", "PCA3", "PCA4"]
#     for c in needed:
#         if c not in df.columns:
#             raise ValueError(f"Kolom {c} tidak ditemukan pada data KMeans output.")

#     out = df[needed + [label_col]].copy()
#     out.rename(columns={label_col: "Cluster Raw"}, inplace=True)
#     return out


# # ===== Alignment (mapping cluster + sign flip PCA) =====
# @dataclass(frozen=True)
# class AlignmentResult:
#     mapping: Dict[Any, str]                 # raw_cluster_value -> "C1/C2/C3"
#     sign: Dict[str, int]                    # "PCA1".. -> +1/-1
#     score_accuracy: float                   # accuracy pada full data
#     note: str                               # keterangan singkat


# def _unique_sorted_cluster_ids(values: List[Any]) -> List[Any]:
#     # coba urut numerik jika mungkin
#     cleaned = []
#     for v in values:
#         if pd.isna(v):
#             continue
#         cleaned.append(v)
#     uniq = list(dict.fromkeys(cleaned))
#     # jika semua bisa jadi int, urutkan
#     try:
#         ints = [int(round(float(x))) for x in uniq]
#         # pastikan tidak ada bentrok
#         if len(set(ints)) == len(ints):
#             order = [x for _, x in sorted(zip(ints, uniq), key=lambda t: t[0])]
#             return order
#     except Exception:
#         pass
#     return sorted(uniq, key=lambda x: str(x))


# def infer_best_alignment(
#     df_raw: pd.DataFrame,
#     ambang: Dict[str, Dict[str, float]],
#     labels_target: List[str] = ["C1", "C2", "C3"],
# ) -> AlignmentResult:
#     """
#     Cari mapping Cluster->C* dan sign flip PCA1..PCA4 terbaik berdasarkan akurasi terhadap rule RF.
#     """
#     df = df_raw.copy()

#     # pastikan PCA numeric
#     cols = ["PCA1", "PCA2", "PCA3", "PCA4"]
#     _ensure_pca_columns(df, cols)

#     if "Cluster Raw" not in df.columns:
#         raise ValueError("Kolom 'Cluster Raw' tidak ada. Gunakan build_rf_dataframe_from_kmeans_output().")

#     raw_vals = df["Cluster Raw"].tolist()

#     # Jika label memang sudah berbentuk C1/C2/C3 di file (string 'C1', dst),
#     # maka tidak perlu remap; tetap bisa saja ada sign flip, tapi biasanya file Colab sudah konsisten.
#     raw_str = [str(v).strip() for v in raw_vals if not pd.isna(v)]
#     looks_like_c = bool(raw_str) and all(re.match(r"^[cC]\s*\d+$", s) for s in raw_str)
#     if looks_like_c:
#         mapping = {v: normalize_label_to_c(v) for v in _unique_sorted_cluster_ids(raw_vals)}
#         sign = {c: 1 for c in cols}
#         df2 = df.copy()
#         df2["Cluster Aktual"] = [mapping[v] for v in df2["Cluster Raw"]]
#         pred = apply_rf_manual(df2, ambang)["Voting Mayoritas"]
#         acc = float(accuracy_score(df2["Cluster Aktual"], pred))
#         return AlignmentResult(mapping=mapping, sign=sign, score_accuracy=acc, note="Label sudah C1/C2/C3 (tanpa remap & tanpa flip).")

#     # Kalau tidak (umumnya label numeric 0/1/2 dari KMeans), kita cari kombinasi terbaik

#     cluster_ids = _unique_sorted_cluster_ids(raw_vals)
#     if len(cluster_ids) != 3:
#         raise ValueError(
#             f"Auto alignment butuh tepat 3 label cluster unik, ditemukan {len(cluster_ids)}: {cluster_ids}"
#         )

#     best: Optional[AlignmentResult] = None

#     # precompute base df untuk speed
#     base = df[cols].copy()

#     for signs in product([1, -1], repeat=4):
#         df_s = base.copy()
#         sign_map = {col: int(s) for col, s in zip(cols, signs)}
#         for col in cols:
#             if sign_map[col] == -1:
#                 df_s[col] = -df_s[col]

#         # apply rules once per sign
#         tmp = df_s.copy()
#         tmp["Cluster Raw"] = df["Cluster Raw"].values
#         pred = apply_rf_manual(tmp, ambang)["Voting Mayoritas"].tolist()

#         for perm in permutations(labels_target, 3):
#             mapping = {cid: lbl for cid, lbl in zip(cluster_ids, perm)}
#             y_true = [mapping[v] for v in raw_vals]
#             acc = float(accuracy_score(y_true, pred))

#             # tie-break deterministik: acc desc, jumlah flip asc, mapping string asc
#             flips = sum(1 for s in signs if s == -1)
#             mapping_key = "|".join([f"{str(k)}->{v}" for k, v in sorted(mapping.items(), key=lambda t: str(t[0]))])

#             if best is None:
#                 best = AlignmentResult(mapping=mapping, sign=sign_map, score_accuracy=acc, note=mapping_key)
#             else:
#                 best_flips = sum(1 for v in best.sign.values() if v == -1)
#                 best_key = best.note
#                 if (acc > best.score_accuracy) or (
#                     acc == best.score_accuracy and (flips < best_flips)
#                 ) or (
#                     acc == best.score_accuracy and flips == best_flips and mapping_key < best_key
#                 ):
#                     best = AlignmentResult(mapping=mapping, sign=sign_map, score_accuracy=acc, note=mapping_key)

#     assert best is not None
#     return AlignmentResult(
#         mapping=best.mapping,
#         sign=best.sign,
#         score_accuracy=best.score_accuracy,
#         note=f"Auto alignment terbaik (full-data acc={best.score_accuracy:.4f}): {best.note} | sign={best.sign}",
#     )


# def apply_alignment(df_raw: pd.DataFrame, alignment: AlignmentResult) -> pd.DataFrame:
#     """Terapkan mapping & sign flip ke dataframe."""
#     df = df_raw.copy()
#     cols = ["PCA1", "PCA2", "PCA3", "PCA4"]
#     _ensure_pca_columns(df, cols)

#     # sign flip
#     for c in cols:
#         if alignment.sign.get(c, 1) == -1:
#             df[c] = -df[c]

#     # mapping label
#     df["Cluster Aktual"] = df["Cluster Raw"].map(alignment.mapping)
#     if df["Cluster Aktual"].isna().any():
#         raise ValueError("Gagal mapping label cluster ke C1/C2/C3. Periksa mapping alignment.")
#     return df


# # ===== Split =====
# def split_train_test_colab_like(
#     df: pd.DataFrame,
#     train_ratio: float = 0.8,
#     random_state: int = 42,
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Meniru praktik Colab yang biasanya:
#     - Data diacak dulu
#     - Ambil 260 sebagai training (80% dari 325), sisanya testing
#     """
#     if not (0.05 < float(train_ratio) < 0.95):
#         raise ValueError("train_ratio harus di antara 0.05 dan 0.95.")
#     shuffled = df.sample(frac=1, random_state=int(random_state)).reset_index(drop=True)
#     n_train = int(round(len(shuffled) * float(train_ratio)))
#     n_train = max(1, min(n_train, len(shuffled) - 1))
#     df_train = shuffled.iloc[:n_train].reset_index(drop=True)
#     df_test = shuffled.iloc[n_train:].reset_index(drop=True)
#     return df_train, df_test


# def split_train_test_stratified(
#     df: pd.DataFrame,
#     train_ratio: float = 0.8,
#     random_state: int = 42,
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """Split stratified agar distribusi label relatif stabil."""
#     if "Cluster Aktual" not in df.columns:
#         raise ValueError("Kolom 'Cluster Aktual' wajib ada untuk split stratified.")
#     if not (0.05 < float(train_ratio) < 0.95):
#         raise ValueError("train_ratio harus di antara 0.05 dan 0.95.")
#     y = df["Cluster Aktual"]
#     df_train, df_test = train_test_split(
#         df,
#         train_size=float(train_ratio),
#         random_state=int(random_state),
#         shuffle=True,
#         stratify=y,
#     )
#     return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


# # ===== Save helpers =====
# def save_dataframe_excel(df: pd.DataFrame, output_path: str, sheet_name: str = "Data") -> str:
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with pd.ExcelWriter(output_path, engine="openpyxl") as w:
#         df.to_excel(w, sheet_name=sheet_name, index=False)
#     return output_path


# def save_evaluation_excel(
#     df_pred: pd.DataFrame,
#     report_dict: Dict[str, Any],
#     output_path: str,
#     meta: Optional[Dict[str, Any]] = None,
#     title_sheet: str = "Prediksi",
# ) -> str:
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     report_df = pd.DataFrame(report_dict).transpose()

#     # rapikan kolom seperti Colab (precision, recall, f1-score, support)
#     keep = [c for c in ["precision", "recall", "f1-score", "support"] if c in report_df.columns]
#     report_view = report_df[keep] if keep else report_df

#     with pd.ExcelWriter(output_path, engine="openpyxl") as w:
#         df_pred.to_excel(w, sheet_name=title_sheet, index=False)
#         report_view.to_excel(w, sheet_name="Evaluasi")
#         if meta:
#             pd.DataFrame([meta]).to_excel(w, sheet_name="Meta", index=False)
#     return output_path


# # ===== Main pipeline =====
# def run_rf_manual_evaluation(
#     kmeans_output_path: str,
#     output_folder: str,
#     ambang: Optional[Dict[str, Dict[str, float]]] = None,
#     train_ratio: float = 0.8,
#     random_state: int = 42,
#     labels: Optional[List[str]] = None,
#     split_method: str = "colab_like",        # "colab_like" | "stratified"
#     alignment_mode: str = "auto",            # "auto" | "none"
# ) -> Dict[str, Any]:
#     """
#     End-to-end:
#     - load data PCA+Cluster dari output KMeans
#     - (opsional) auto-align: mapping cluster & sign flip PCA
#     - split train/test
#     - jalankan RF manual + evaluasi untuk train & test
#     - simpan file train, test, report train, report test

#     Return dict siap disimpan ke DB/controller.
#     """
#     if ambang is None:
#         ambang = DEFAULT_AMBANG
#     ambang = validate_ambang(ambang)

#     if labels is None:
#         labels = ["C1", "C2", "C3"]

#     df_raw = build_rf_dataframe_from_kmeans_output(kmeans_output_path)

#     # alignment
#     if alignment_mode not in ("auto", "none"):
#         raise ValueError("alignment_mode harus 'auto' atau 'none'")

#     if alignment_mode == "auto":
#         alignment = infer_best_alignment(df_raw, ambang, labels_target=labels)
#     else:
#         # tanpa auto: hanya coba normalize ke C* (kalau dari file sudah C1/C2/C3)
#         mapping = {v: normalize_label_to_c(v) for v in _unique_sorted_cluster_ids(df_raw["Cluster Raw"].tolist())}
#         alignment = AlignmentResult(
#             mapping=mapping,
#             sign={c: 1 for c in ["PCA1", "PCA2", "PCA3", "PCA4"]},
#             score_accuracy=0.0,
#             note="alignment_mode=none (tanpa auto mapping/sign flip).",
#         )

#     df_all = apply_alignment(df_raw, alignment)

#     # split
#     if split_method not in ("colab_like", "stratified"):
#         raise ValueError("split_method harus 'colab_like' atau 'stratified'")

#     if split_method == "colab_like":
#         df_train, df_test = split_train_test_colab_like(df_all, train_ratio=train_ratio, random_state=random_state)
#     else:
#         df_train, df_test = split_train_test_stratified(df_all, train_ratio=train_ratio, random_state=random_state)

#     # train eval
#     pred_train = apply_rf_manual(df_train, ambang)
#     report_train, acc_train, macro_train = evaluate_predictions(pred_train, labels=labels)

#     # test eval
#     pred_test = apply_rf_manual(df_test, ambang)
#     report_test, acc_test, macro_test = evaluate_predictions(pred_test, labels=labels)

#     os.makedirs(output_folder, exist_ok=True)
#     ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

#     train_path = os.path.join(output_folder, f"Data_Train_{ts}.xlsx")
#     test_path = os.path.join(output_folder, f"Data_Test_{ts}.xlsx")
#     report_train_path = os.path.join(output_folder, f"RF_Evaluasi_Train_{ts}.xlsx")
#     report_test_path = os.path.join(output_folder, f"RF_Evaluasi_Test_{ts}.xlsx")

#     # simpan split (berisi Cluster Raw + Cluster Aktual, sudah align)
#     save_dataframe_excel(df_train, train_path, sheet_name="Train")
#     save_dataframe_excel(df_test, test_path, sheet_name="Test")

#     meta = {
#         "kmeans_output_path": kmeans_output_path,
#         "train_ratio": float(train_ratio),
#         "random_state": int(random_state),
#         "split_method": split_method,
#         "alignment_mode": alignment_mode,
#         "alignment_note": alignment.note,
#         "alignment_full_data_accuracy": float(alignment.score_accuracy),
# "alignment_full_data_macro_f1": float(alignment.score_macro_f1),
#         "mapping": str(alignment.mapping),
#         "sign": str(alignment.sign),
#     }

#     save_evaluation_excel(pred_train, report_train, report_train_path, meta=meta, title_sheet="Train_Prediksi")
#     save_evaluation_excel(pred_test, report_test, report_test_path, meta=meta, title_sheet="Test_Prediksi")

#     return {
#         "train_path": train_path,
#         "test_path": test_path,
#         "alignment": {
#             "mapping": alignment.mapping,
#             "sign": alignment.sign,
#             "full_data_accuracy": alignment.score_accuracy,
# "full_data_macro_f1": alignment.score_macro_f1,
#             "note": alignment.note,
#         },
#         "train": {
#             "report": report_train,
#             "accuracy": acc_train,
#             "macro_f1": macro_train,
#             "output_path": report_train_path,
#         },
#         "test": {
#             "report": report_test,
#             "accuracy": acc_test,
#             "macro_f1": macro_test,
#             "output_path": report_test_path,
#         },
#         "ambang": ambang,
#         "random_state": int(random_state),
#         "train_ratio": float(train_ratio),
#         "split_method": split_method,
#         "alignment_mode": alignment_mode,
#     }
# =========================
# BATAS
# ======================

# ==========================================
# RANDOM FOREST (MANUAL RULES) SERVICE
# ==========================================
# Mengikuti Colab Anda:
# - Input: Data dengan kolom PCA1..PCA4 + label aktual ("Cluster Aktual" atau "Cluster")
# - Buat 5 "tree" berbasis aturan ambang PCA
# - Voting mayoritas -> prediksi final
# - Evaluasi classification_report, accuracy, macro f1
#
# Catatan:
# Ini bukan RandomForestClassifier sklearn, tapi "RF manual" (rule-based voting) sesuai skrip Colab.

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, Any, List, Tuple, Optional
# from collections import Counter
# import json
# import os

# import pandas as pd
# from sklearn.metrics import classification_report, accuracy_score, f1_score
# from sklearn.model_selection import train_test_split


# DEFAULT_AMBANG: Dict[str, Dict[str, float]] = {
#     "PCA1": {"low": -0.471222, "mid": 0.740738},
#     "PCA2": {"low": -0.496107, "mid": 0.799846},
#     "PCA3": {"low": -0.143652, "mid": 0.300361},
#     "PCA4": {"low": -0.206872, "mid": 0.633076},
# }


# def _as_float(x) -> float:
#     try:
#         return float(x)
#     except Exception:
#         return float("nan")


# def normalize_label(x) -> str:
#     """Normalisasi label aktual/prediksi ke format 'C1','C2','C3'."""
#     if x is None:
#         return ""
#     # angka numpy/int/float
#     if isinstance(x, (int, float)) and not pd.isna(x):
#         # asumsi cluster kmeans 0..n-1
#         if int(x) == x:
#             return f"C{int(x) + 1}"
#         return f"C{int(round(x)) + 1}"
#     s = str(x).strip()
#     if s == "":
#         return ""
#     # jika "0","1","2"
#     if s.isdigit():
#         return f"C{int(s) + 1}"
#     # jika sudah "C1"
#     if s.upper().startswith("C"):
#         return "C" + s[1:].strip()
#     return s


# def validate_ambang(ambang: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
#     """Validasi ambang JSON agar strukturnya sesuai dan tipe float."""
#     required = ["PCA1", "PCA2", "PCA3", "PCA4"]
#     out: Dict[str, Dict[str, float]] = {}
#     for k in required:
#         if k not in ambang:
#             raise ValueError(f"Ambang harus punya key '{k}'")
#         if not isinstance(ambang[k], dict) or "low" not in ambang[k] or "mid" not in ambang[k]:
#             raise ValueError(f"Ambang '{k}' harus punya subkey 'low' dan 'mid'")
#         out[k] = {"low": float(ambang[k]["low"]), "mid": float(ambang[k]["mid"])}
#     return out


# def kategori_pca(nilai: float, batas: Dict[str, float]) -> str:
#     if nilai <= batas["low"]:
#         return "Rendah"
#     elif nilai <= batas["mid"]:
#         return "Sedang"
#     else:
#         return "Tinggi"


# # === 5 pohon (sesuai Colab) ===
# def tree1(row, ambang):
#     p1 = kategori_pca(row["PCA1"], ambang["PCA1"])
#     p2 = kategori_pca(row["PCA2"], ambang["PCA2"])
#     p4 = kategori_pca(row["PCA4"], ambang["PCA4"])
#     if p1 == "Tinggi" and p2 == "Tinggi":
#         return "C1"
#     elif p1 == "Sedang" and p4 == "Rendah":
#         return "C2"
#     else:
#         return "C3"


# def tree2(row, ambang):
#     p4 = kategori_pca(row["PCA4"], ambang["PCA4"])
#     p1 = kategori_pca(row["PCA1"], ambang["PCA1"])
#     p3 = kategori_pca(row["PCA3"], ambang["PCA3"])
#     if p4 == "Tinggi" and p3 in ["Sedang", "Tinggi"]:
#         return "C3"
#     elif p4 == "Rendah" and p1 == "Sedang":
#         return "C2"
#     else:
#         return "C1"


# def tree3(row, ambang):
#     p2 = kategori_pca(row["PCA2"], ambang["PCA2"])
#     p1 = kategori_pca(row["PCA1"], ambang["PCA1"])
#     p3 = kategori_pca(row["PCA3"], ambang["PCA3"])
#     if p2 == "Tinggi" and p1 != "Rendah":
#         return "C1"
#     elif p2 == "Sedang" and p3 == "Sedang":
#         return "C2"
#     else:
#         return "C3"


# def tree4(row, ambang):
#     p3 = kategori_pca(row["PCA3"], ambang["PCA3"])
#     p4 = kategori_pca(row["PCA4"], ambang["PCA4"])
#     if p3 == "Tinggi" and p4 == "Tinggi":
#         return "C1"
#     elif p3 == "Sedang" and p4 == "Rendah":
#         return "C2"
#     else:
#         return "C3"


# def tree5(row, ambang):
#     p1 = kategori_pca(row["PCA1"], ambang["PCA1"])
#     p2 = kategori_pca(row["PCA2"], ambang["PCA2"])
#     p4 = kategori_pca(row["PCA4"], ambang["PCA4"])
#     if p1 == "Tinggi" and p4 == "Tinggi":
#         return "C1"
#     elif p1 == "Sedang" and p2 == "Sedang":
#         return "C2"
#     else:
#         return "C3"


# def majority_vote(values: List[str]) -> str:
#     """Voting mayoritas yang stabil (tie-break deterministik)."""
#     c = Counter(values)
#     if not c:
#         return ""
#     most_common = c.most_common()
#     top_count = most_common[0][1]
#     tied = sorted([v for v, cnt in most_common if cnt == top_count])
#     return tied[0]  # deterministik


# def apply_rf_manual(df: pd.DataFrame, ambang: Dict[str, Dict[str, float]]) -> pd.DataFrame:
#     """Tambahkan kolom Tree1..Tree5 dan Voting Mayoritas."""
#     for col in ["PCA1", "PCA2", "PCA3", "PCA4"]:
#         if col not in df.columns:
#             raise ValueError(f"Kolom {col} tidak ditemukan pada data.")
#         df[col] = df[col].apply(_as_float)

#     out = df.copy()
#     out["Tree1"] = out.apply(lambda r: tree1(r, ambang), axis=1)
#     out["Tree2"] = out.apply(lambda r: tree2(r, ambang), axis=1)
#     out["Tree3"] = out.apply(lambda r: tree3(r, ambang), axis=1)
#     out["Tree4"] = out.apply(lambda r: tree4(r, ambang), axis=1)
#     out["Tree5"] = out.apply(lambda r: tree5(r, ambang), axis=1)
#     out["Voting Mayoritas"] = out[["Tree1", "Tree2", "Tree3", "Tree4", "Tree5"]].apply(
#         lambda x: majority_vote(list(x.values)), axis=1
#     )
#     return out


# def evaluate_predictions(
#     df_with_pred: pd.DataFrame,
#     actual_col: str = "Cluster Aktual",
#     pred_col: str = "Voting Mayoritas",
#     labels: Optional[List[str]] = None,
# ) -> Tuple[Dict[str, Any], float, float]:
#     """Return (classification_report dict, accuracy, macro_f1)."""
#     if actual_col not in df_with_pred.columns:
#         raise ValueError(f"Kolom label aktual '{actual_col}' tidak ditemukan.")
#     if pred_col not in df_with_pred.columns:
#         raise ValueError(f"Kolom prediksi '{pred_col}' tidak ditemukan.")

#     y_true = df_with_pred[actual_col].apply(normalize_label).tolist()
#     y_pred = df_with_pred[pred_col].apply(normalize_label).tolist()

#     if labels is None:
#         labels = sorted(list(set([x for x in y_true + y_pred if x])))

#     report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
#     acc = float(accuracy_score(y_true, y_pred))
#     macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

#     # pastikan JSON-serializable
#     def _clean(o):
#         if isinstance(o, dict):
#             return {k: _clean(v) for k, v in o.items()}
#         if isinstance(o, list):
#             return [_clean(v) for v in o]
#         if isinstance(o, (int, float, str)) or o is None:
#             try:
#                 if isinstance(o, float):
#                     return float(o)
#                 return o
#             except Exception:
#                 return str(o)
#         return str(o)

#     return _clean(report), acc, macro


# def load_excel_any_sheet(path: str, prefer_sheets: List[str]) -> pd.DataFrame:
#     """Load excel dari sheet yang cocok, fallback ke sheet pertama."""
#     xls = pd.ExcelFile(path)
#     for s in prefer_sheets:
#         if s in xls.sheet_names:
#             return pd.read_excel(path, sheet_name=s)
#     return pd.read_excel(path, sheet_name=xls.sheet_names[0])


# def build_rf_dataframe_from_kmeans_output(kmeans_output_path: str) -> pd.DataFrame:
#     """
#     Ambil data dari output KMeans:
#     - minimal harus ada PCA1..PCA4
#     - label aktual bisa bernama 'Cluster Aktual' atau 'Cluster'
#     Hasil akan distandardisasi jadi kolom 'Cluster Aktual'.
#     """
#     df = load_excel_any_sheet(
#         kmeans_output_path,
#         prefer_sheets=["Data+Cluster", "Data_Cluster", "Data", "Sheet1"],
#     )

#     # label aktual
#     if "Cluster Aktual" in df.columns:
#         label_col = "Cluster Aktual"
#     elif "Cluster" in df.columns:
#         label_col = "Cluster"
#     else:
#         raise ValueError("Tidak ditemukan kolom label cluster ('Cluster' atau 'Cluster Aktual') pada file KMeans.")

#     needed = ["PCA1", "PCA2", "PCA3", "PCA4"]
#     for c in needed:
#         if c not in df.columns:
#             raise ValueError(f"Kolom {c} tidak ditemukan pada data KMeans output.")

#     out = df[needed + [label_col]].copy()
#     out.rename(columns={label_col: "Cluster Aktual"}, inplace=True)
#     out["Cluster Aktual"] = out["Cluster Aktual"].apply(normalize_label)

#     return out


# def split_train_test(
#     df: pd.DataFrame,
#     train_ratio: float = 0.7,
#     random_state: int = 42,
#     stratify: bool = True,
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """Split data latih & uji (default stratify by label)."""
#     if "Cluster Aktual" not in df.columns:
#         raise ValueError("Kolom 'Cluster Aktual' wajib ada untuk split stratified.")
#     y = df["Cluster Aktual"]
#     strat = y if stratify else None
#     df_train, df_test = train_test_split(
#         df,
#         train_size=float(train_ratio),
#         random_state=int(random_state),
#         shuffle=True,
#         stratify=strat,
#     )
#     return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


# def save_dataframe_excel(df: pd.DataFrame, output_path: str, sheet_name: str = "Data") -> str:
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with pd.ExcelWriter(output_path, engine="openpyxl") as w:
#         df.to_excel(w, sheet_name=sheet_name, index=False)
#     return output_path


# def save_evaluation_excel(
#     df_pred: pd.DataFrame,
#     report_dict: Dict[str, Any],
#     output_path: str,
#     title_sheet: str = "Prediksi",
# ) -> str:
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     report_df = pd.DataFrame(report_dict).transpose()

#     # rapikan kolom bila ada
#     for c in ["precision", "recall", "f1-score", "support"]:
#         if c in report_df.columns:
#             pass

#     with pd.ExcelWriter(output_path, engine="openpyxl") as w:
#         df_pred.to_excel(w, sheet_name=title_sheet, index=False)
#         report_df.to_excel(w, sheet_name="Classification_Report")
#     return output_path


# def run_rf_manual_evaluation(
#     kmeans_output_path: str,
#     output_folder: str,
#     ambang: Dict[str, Dict[str, float]] | None = None,
#     train_ratio: float = 0.7,
#     random_state: int = 42,
#     labels: Optional[List[str]] = None,
# ) -> Dict[str, Any]:
#     """
#     End-to-end:
#     - load data PCA+Cluster dari output KMeans
#     - split train/test
#     - jalankan RF manual + evaluasi untuk train & test
#     - simpan file train, test, report train, report test
#     """
#     if ambang is None:
#         ambang = DEFAULT_AMBANG
#     ambang = validate_ambang(ambang)

#     df_all = build_rf_dataframe_from_kmeans_output(kmeans_output_path)

#     df_train, df_test = split_train_test(df_all, train_ratio=train_ratio, random_state=random_state, stratify=True)

#     # train evaluation
#     pred_train = apply_rf_manual(df_train, ambang)
#     report_train, acc_train, macro_train = evaluate_predictions(pred_train, labels=labels)

#     # test evaluation
#     pred_test = apply_rf_manual(df_test, ambang)
#     report_test, acc_test, macro_test = evaluate_predictions(pred_test, labels=labels)

#     os.makedirs(output_folder, exist_ok=True)
#     ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

#     train_path = os.path.join(output_folder, f"Data_Train_{ts}.xlsx")
#     test_path = os.path.join(output_folder, f"Data_Test_{ts}.xlsx")
#     report_train_path = os.path.join(output_folder, f"RF_Report_Train_{ts}.xlsx")
#     report_test_path = os.path.join(output_folder, f"RF_Report_Test_{ts}.xlsx")

#     save_dataframe_excel(df_train, train_path, sheet_name="Train")
#     save_dataframe_excel(df_test, test_path, sheet_name="Test")
#     save_evaluation_excel(pred_train, report_train, report_train_path, title_sheet="Train_Prediksi")
#     save_evaluation_excel(pred_test, report_test, report_test_path, title_sheet="Test_Prediksi")

#     return {
#         "train_path": train_path,
#         "test_path": test_path,
#         "train": {
#             "report": report_train,
#             "accuracy": acc_train,
#             "macro_f1": macro_train,
#             "output_path": report_train_path,
#         },
#         "test": {
#             "report": report_test,
#             "accuracy": acc_test,
#             "macro_f1": macro_test,
#             "output_path": report_test_path,
#         },
#         "ambang": ambang,
#         "random_state": int(random_state),
#         "train_ratio": float(train_ratio),
#     }
