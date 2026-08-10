"""
E0 — Data module sadar-metadata (CUBS combined)
================================================
Menghasilkan fondasi untuk SEMUA eksperimen berikutnya:
  - data/master_index.csv   : 1 baris/citra, lengkap dengan center/SNR/morf/CF/mask
  - data/splits_5fold.json  : 5-fold CV patient-disjoint (untuk E1a/E1b)
  - data/splits_loco.json   : leave-one-center-out (untuk E3' — kontribusi bintang K2)
  - results/E0_sanity_*.png : overlay mask A1 hasil rasterisasi (verifikasi visual)

GT clinical = anotator A1 (GOLD STANDARD), dirasterisasi dari profil LI/MA.
CATATAN PENTING: folder `masks/` bawaan dataset dibuat dari A1' (bukan A1) — TIDAK dipakai
sebagai GT di sini. Kita rasterisasi ulang dari SEGMENTATIONS/Manual-A1.

Murni lokal (tanpa GPU). Jalankan: python E0_build_dataset_index.py
"""
from pathlib import Path
import json, os, sys
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# ---------------------------------------------------------------- paths
REPO = Path(__file__).resolve().parents[1]
DS   = Path(os.environ.get("CUBS_DATA_ROOT", REPO / "external_data"))
CLIN = DS / "cubs_clinical"
TECH = DS / "cubs_technical"
OUT  = REPO
(OUT / "data").mkdir(parents=True, exist_ok=True)
(OUT / "results").mkdir(parents=True, exist_ok=True)

GT_ANNOT = "Manual-A1"          # gold standard
ANNOT_MAP = {"A1": "Manual-A1", "A1p": "Manual-A1petik",
             "A2": "Manual-A2", "A3": "Manual-A3"}

# ---------------------------------------------------------------- helpers
def read_cf(cf_path: Path):
    try:
        return float(cf_path.read_text().strip().split()[0])
    except Exception:
        return np.nan

def load_profile(txt: Path):
    """Baca file profil batas (koordinat 'x y' per baris) -> array (N,2)."""
    pts = []
    for line in txt.read_text().splitlines():
        s = line.split()
        if len(s) >= 2:
            pts.append((float(s[0]), float(s[1])))
    return np.asarray(pts, dtype=np.float32)

def rasterize_imc(li: np.ndarray, ma: np.ndarray, hw):
    """Mask IMC = area antara batas LI (atas) dan MA (bawah).
    Poligon = LI kiri->kanan lalu MA kanan->kiri, di-fill."""
    H, W = hw
    li = li[np.argsort(li[:, 0])]
    ma = ma[np.argsort(ma[:, 0])][::-1]
    poly = np.vstack([li, ma]).astype(np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [poly], 1)
    return m

# ---------------------------------------------------------------- 1) clinical centers
clin_db = pd.read_csv(CLIN / "ClinicalDatabase-CUBS.csv", sep=None, engine="python")
clin_db.columns = [c.strip() for c in clin_db.columns]
c0 = clin_db.columns[0]
clin_db["center_raw"] = clin_db[c0].ffill()
center_of = {}   # clin_XXXX -> center
for _, r in clin_db.iterrows():
    pid = str(r["Patient ID"]).strip()
    raw = str(r["center_raw"])
    center_of[pid] = "Cyprus" if "Cyprus" in raw else ("Pisa" if "Pisa" in raw else "Unknown")

# ---------------------------------------------------------------- 2) technical center/SNR/morph
tdf = pd.read_excel(TECH / "CUBS-tech-SNR-geometry.xlsx", header=1, usecols="A:D")
tdf.columns = ["center", "image", "morph", "snr"]
tdf["center"] = tdf["center"].ffill()
tdf["image"]  = tdf["image"].astype(str).str.strip()
tech_meta = {r["image"]: (r["center"], r["morph"],
                          (np.nan if r["snr"] == -9999 else r["snr"]))
             for _, r in tdf.iterrows()}

# ---------------------------------------------------------------- 3) build rows
rows = []

def relative_data_path(path: Path) -> str:
    """Store a portable path relative to CUBS_DATA_ROOT."""
    return path.relative_to(DS).as_posix()

# clinical  (Pisa klinis -> "Pisa_clin" agar terpisah dari Pisa teknikal)
for img in sorted((CLIN / "IMAGES").glob("*.tiff")):
    iid = img.stem                       # clin_0001_L
    pid, side = iid.rsplit("_", 1)       # clin_0001 , L
    center = center_of.get(pid, "Unknown")
    if center == "Pisa":
        center = "Pisa_clin"
    seg = CLIN / "SEGMENTATIONS" / GT_ANNOT
    li, ma, rect = seg / f"{iid}-LI.txt", seg / f"{iid}-MA.txt", seg / f"{iid}_rect.txt"
    rows.append(dict(
        image_id=iid, release="clinical", center=center, city=center.split("_")[0],
        patient_id=pid, side=side, snr=np.nan, morph=np.nan,
        cf_mm_per_px=read_cf(CLIN / "CF" / f"{iid}_CF.txt"),
        img_path=relative_data_path(img), li_path=relative_data_path(li),
        ma_path=relative_data_path(ma), rect_path=relative_data_path(rect),
        mask_source="A1_rasterized",
        gt_available=li.exists() and ma.exists()))

# technical  (GT = A1 dirasterisasi dari LIMA-Profiles/Manual-A1, KONSISTEN dgn klinis;
#             Pisa teknikal -> "Pisa_tech" utk uji kota-sama-akuisisi-beda)
for img in sorted((TECH / "images").glob("*.tiff")):
    iid = img.stem                       # tech_001
    center, morph, snr = tech_meta.get(iid, ("Unknown", np.nan, np.nan))
    if center == "Pisa":
        center = "Pisa_tech"
    seg = TECH / "LIMA-Profiles" / GT_ANNOT
    li, ma, rect = seg / f"{iid}-LI.txt", seg / f"{iid}-MA.txt", seg / f"{iid}_rect.txt"
    rows.append(dict(
        image_id=iid, release="technical", center=center, city=center.split("_")[0],
        patient_id=iid, side="NA", snr=snr, morph=morph,
        cf_mm_per_px=read_cf(TECH / "CF" / f"{iid}_CF.txt"),
        img_path=relative_data_path(img), li_path=relative_data_path(li),
        ma_path=relative_data_path(ma), rect_path=relative_data_path(rect),
        mask_source="A1_rasterized",
        gt_available=li.exists() and ma.exists()))

master = pd.DataFrame(rows)
master.to_csv(OUT / "data" / "master_index.csv", index=False)

# ---------------------------------------------------------------- 4) splits
rng = np.random.default_rng(42)

# held-out TEST (15% pasien, stratified per pusat) + 5-fold CV patient-disjoint pada sisanya.
# Unit split = patient_id (klinis: 1 pasien = 2 citra L/R; teknikal: 1 citra = 1 unit).
split = {"test": [], "folds": {f"fold{i+1}": [] for i in range(5)}}
for center, grp in master.groupby("center"):
    pts = grp["patient_id"].unique().tolist()
    pts = [pts[i] for i in rng.permutation(len(pts))]
    n_test = max(1, round(0.15 * len(pts)))
    split["test"] += pts[:n_test]
    for j, p in enumerate(pts[n_test:]):
        split["folds"][f"fold{j % 5 + 1}"].append(p)
json.dump(split, open(OUT / "data" / "splits_5fold.json", "w"), indent=1)

# ringkas jumlah CITRA per fold/test (pasien -> citra)
pid2n = master.groupby("patient_id").size().to_dict()
img_test  = sum(pid2n[p] for p in split["test"])
img_folds = {k: sum(pid2n[p] for p in v) for k, v in split["folds"].items()}

# LOCO — leave-one-center-out (test = 1 center, train = sisanya)
centers = sorted(master["center"].unique())
loco = {}
for c in centers:
    if c == "Unknown":
        continue
    test_ids  = master.loc[master.center == c, "image_id"].tolist()
    train_ids = master.loc[master.center != c, "image_id"].tolist()
    loco[c] = dict(n_test=len(test_ids), n_train=len(train_ids))
json.dump(loco, open(OUT / "data" / "splits_loco.json", "w"), indent=1)

# ---------------------------------------------------------------- 5) sanity masks (clinical + technical A1)
sample = pd.concat([
    master[(master.release == "clinical")  & master.gt_available].head(2),
    master[(master.release == "technical") & master.gt_available].head(2)])
for _, r in sample.iterrows():
    img = np.array(Image.open(DS / r["img_path"]).convert("L"))
    li  = load_profile(DS / r["li_path"])
    ma  = load_profile(DS / r["ma_path"])
    m   = rasterize_imc(li, ma, img.shape)
    ov  = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ov[m == 1] = (0.5 * ov[m == 1] + np.array([0, 0, 128])).astype(np.uint8)
    cv2.imwrite(str(OUT / "results" / f"E0_sanity_{r['image_id']}.png"), ov)

# ---------------------------------------------------------------- 6) report
print("=== master_index ===", len(master), "citra")
print(master.groupby(["release", "center"]).size())
print("\nGT tersedia:", int(master.gt_available.sum()), "/", len(master))
print("CF valid   :", int(master.cf_mm_per_px.notna().sum()), "/", len(master))
print("CF range   : %.3f - %.3f mm/px" %
      (master.cf_mm_per_px.min(), master.cf_mm_per_px.max()))
print("\nHeld-out test:", len(split["test"]), "pasien /", img_test, "citra")
print("5-fold (citra/fold):", img_folds)
print("LOCO centers (citra test):", {k: v["n_test"] for k, v in loco.items()})
print("\nSanity overlay:", len(sample), "-> results/E0_sanity_*.png")
