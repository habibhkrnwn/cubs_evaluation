"""
R1 — E9 DIPERBAIKI: satu prosedur pengukuran untuk semua (temuan fatal M7)
==========================================================================
Masalah pada E9 lama:
  - CIMT model  diukur dari MASK BINER  (row_last - row_first, grid 256x256)
  - CIMT anotator diukur dari PROFIL NATIVE (interp MA_y - LI_y, resolusi penuh)
  Methods 3.5 mengklaim keduanya identik. Tidak. Bias "unbiased" -0.0013 mm adalah
  pembatalan bias model (+0.055) oleh bias rasterisasi referensi (-0.057).

Perbaikan di sini:
  Rasterisasi A1, A1', A2, A3 dengan PIPELINE YANG SAMA PERSIS seperti GT training:
    profil LI/MA  ->  fillPoly (resolusi penuh)
                  ->  crop_roi_with_padding memakai rect A1 (v_pad=1.0, h_pad=0.1)
                  ->  zero-pad ke persegi sisi `sq`
                  ->  resize 256x256 INTER_NEAREST, threshold >127
                  ->  measure_imt_from_mask: per kolom (row_last-row_first)*sq/256*CF
  ROI dari rect A1 untuk SEMUA anotator: itu wajib, karena prediksi model hidup di
  frame koordinat itu. ROI adalah artefak praproses, bukan anotasi.

Juga dilaporkan varian native (prosedur lama) supaya besar artefaknya terdokumentasi.
Perbandingan observer dipindahkan ke TEST SET 401 citra (data yang tidak dipakai
untuk early stopping / seleksi checkpoint), dengan val pool 2275 sebagai lampiran.

Output: results/R1_observer_matched.csv, R1_observer_matched.md
Murni lokal, tanpa GPU.
"""
from pathlib import Path
import json, os, glob
import numpy as np
import pandas as pd
import cv2

REPO = Path(__file__).resolve().parents[1]
DS   = Path(os.environ.get("CUBS_DATA_ROOT", REPO/"external_data"))
CLIN = DS/"cubs_clinical"
TECH = DS/"cubs_technical"
E2   = Path(os.environ.get("CUBS_EXPERIMENT_ROOT", REPO))
RES  = REPO/"results"
RUN  = Path(os.environ.get("CUBS_PROPOSED_RUN", E2/"external_results"/"proposed"))
RES.mkdir(exist_ok=True, parents=True)

IMG_SIZE = 256
ANN = {"A1": "Manual-A1", "A1p": "Manual-A1petik", "A2": "Manual-A2", "A3": "Manual-A3"}
rng = np.random.default_rng(42)

# ---------------------------------------------------------------- pipeline GT (identik E1a)
def load_profile(txt: Path):
    if not txt.exists():
        return None
    pts = []
    for line in txt.read_text().splitlines():
        s = line.split()
        if len(s) >= 2:
            try:
                pts.append((float(s[0]), float(s[1])))
            except ValueError:
                pass
    return np.asarray(pts, np.float32) if len(pts) >= 2 else None

def rasterize_imc(li, ma, hw):
    H, W = hw
    li = li[np.argsort(li[:, 0])]
    ma = ma[np.argsort(ma[:, 0])][::-1]
    poly = np.vstack([li, ma]).astype(np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [poly], 1)
    return m

def read_rect(rect_path: Path):
    try:
        v = rect_path.read_text().split()
        return float(v[0]), float(v[1]), float(v[2]), float(v[3])
    except Exception:
        return None

def crop_roi_with_padding(image, roi_x, roi_y, roi_w, roi_h, v_pad=1.0, h_pad=0.1):
    h, w = image.shape[:2]
    pad_v = roi_h*v_pad
    y0 = int(max(0, roi_y - pad_v/2)); y1 = int(min(h, roi_y + roi_h + pad_v/2))
    pad_h = roi_w*h_pad
    x0 = int(max(0, roi_x - pad_h));   x1 = int(min(w, roi_x + roi_w + pad_h))
    cropped = image[y0:y1, x0:x1]
    ch, cw = cropped.shape[:2]
    sq = max(ch, cw, 1)
    square = np.zeros((sq, sq), dtype=image.dtype)
    yo, xo = (sq - ch)//2, (sq - cw)//2
    square[yo:yo+ch, xo:xo+cw] = cropped
    return square, float(sq)

def measure_imt_from_mask(binary_mask, cf, roi_square_side, img_size=IMG_SIZE):
    """Prosedur pengukuran resmi paper (Sec. 3.5)."""
    scale = roi_square_side/float(img_size)
    b = binary_mask > 0.5
    th = []
    for col in range(b.shape[1]):
        rows = np.where(b[:, col])[0]
        if len(rows) >= 2:
            th.append((rows[-1] - rows[0])*scale*cf)
    return float(np.mean(th)) if th else np.nan

def native_cimt(li, ma, cf):
    """Prosedur LAMA (profil kontinu) — hanya untuk mengukur besar artefaknya."""
    if li is None or ma is None or not np.isfinite(cf):
        return np.nan
    li = li[np.argsort(li[:, 0])]; ma = ma[np.argsort(ma[:, 0])]
    x0 = max(li[:, 0].min(), ma[:, 0].min()); x1 = min(li[:, 0].max(), ma[:, 0].max())
    if x1 <= x0:
        return np.nan
    xs = np.arange(np.ceil(x0), np.floor(x1) + 1)
    if len(xs) < 3:
        return np.nan
    th = np.interp(xs, ma[:, 0], ma[:, 1]) - np.interp(xs, li[:, 0], li[:, 1])
    th = th[th > 0]
    return float(np.mean(th)*cf) if len(th) else np.nan

def seg_dir(release, annot):
    return (CLIN/"SEGMENTATIONS"/annot) if release == "clinical" else (TECH/"LIMA-Profiles"/annot)

def img_path(release, iid):
    return (CLIN/"IMAGES"/f"{iid}.tiff") if release == "clinical" else (TECH/"images"/f"{iid}.tiff")

# ---------------------------------------------------------------- 1) ukur semua anotator, dua prosedur
master = pd.read_csv(E2/"data"/"master_index.csv")
rows = []
for i, r in enumerate(master.itertuples()):
    iid, rel, cf = r.image_id, r.release, r.cf_mm_per_px
    im = cv2.imread(str(img_path(rel, iid)), cv2.IMREAD_GRAYSCALE)
    if im is None or not np.isfinite(cf):
        continue
    rect = read_rect(seg_dir(rel, "Manual-A1")/f"{iid}_rect.txt")   # ROI = rect A1, untuk SEMUA
    rec = dict(image_id=iid, release=rel, center=r.center, snr=r.snr, morph=r.morph, cf=cf)
    for k, folder in ANN.items():
        d = seg_dir(rel, folder)
        li, ma = load_profile(d/f"{iid}-LI.txt"), load_profile(d/f"{iid}-MA.txt")
        rec[f"nat_{k}"] = native_cimt(li, ma, cf)
        if li is None or ma is None:
            rec[f"msk_{k}"] = np.nan
            continue
        full = rasterize_imc(li, ma, im.shape)*255
        if rect is not None:
            m, sq = crop_roi_with_padding(full, *rect)
        else:
            sq = float(max(im.shape))
            m = full
        m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        rec[f"msk_{k}"] = measure_imt_from_mask((m > 127).astype(np.float32), cf, sq)
        if k == "A1":
            rec["roi_sq_recomputed"] = sq
    rows.append(rec)
    if (i+1) % 400 == 0:
        print(f"  ... {i+1}/{len(master)}")
obs = pd.DataFrame(rows)
print(f"[R1] anotator terukur: {len(obs)} citra")

# ---------------------------------------------------------------- 2) prediksi model: val pool & test set
def load_pred(split):
    fs = sorted(glob.glob(str(RUN/f"{split}_fold*_per_image.csv")))
    df = pd.concat([pd.read_csv(f).assign(fold=j+1) for j, f in enumerate(fs)], ignore_index=True)
    return df.rename(columns={"name": "image_id"})

val = load_pred("val")                       # tiap citra 1x (out-of-fold), n=2275
test = load_pred("test")                     # tiap citra 5x (5 model fold), n=401 unik

# KONVENSI AGREGASI (penting — sumber inkonsistensi antar-tabel).
# Test set dinilai oleh KELIMA model fold. Ada dua cara meringkasnya, dan hasilnya beda:
#   (a) rata-ratakan ERROR per citra lintas fold  -> MAE 0.1200  <- konvensi Tabel 1 & 3
#   (b) rata-ratakan PREDIKSI dulu, baru error    -> MAE 0.1181
# Bias identik (+0.0639) di keduanya. Kita pakai (a) supaya Tabel 5 melaporkan besaran
# yang SAMA dengan baris test Tabel 1 — kalau tidak, dua tabel akan melaporkan error
# model terhadap A1 dengan dua angka berbeda pada citra yang sama, dan itulah persis
# jenis inkonsistensi yang membuat versi sebelumnya bisa ditembak reviewer.
# Ensemble rata-rata-probabilitas (0.1191) adalah objek KETIGA; ia hanya dipakai di
# Sec. 4.6 dan diberi label eksplisit di sana.
test_avg = (test.groupby("image_id")
                .agg(cimt_pred_mm=("cimt_pred_mm", "mean"),   # utk plot Bland-Altman
                     err_foldmean=("imt_abs_err_mm", "mean"),  # utk MAE (konvensi (a))
                     cimt_gt_mm=("cimt_gt_mm", "first"),
                     dice=("dice", "mean"),
                     roi_square_side=("roi_square_side", "first"),
                     cf=("cf", "first")).reset_index())
print(f"[R1] val pool={val.image_id.nunique()}  test={test_avg.image_id.nunique()}")

# ---------------------------------------------------------------- 3) SANITY: reproduksi cimt_gt_mm
chk = obs.merge(val[["image_id", "cimt_gt_mm"]], on="image_id")
d = (chk.msk_A1 - chk.cimt_gt_mm).abs()
print(f"[SANITY] |msk_A1 - cimt_gt_mm|  median={d.median():.6f}  p95={d.quantile(.95):.6f}  max={d.max():.6f}")
print(f"[SANITY] korelasi = {np.corrcoef(chk.msk_A1, chk.cimt_gt_mm)[0,1]:.6f}")

# ---------------------------------------------------------------- 4) statistik pasangan + bootstrap
def boot_ci(x, fn, B=5000, seed=42):
    r = np.random.default_rng(seed)
    n = len(x)
    s = [fn(x[r.integers(0, n, n)]) for _ in range(B)]
    return np.percentile(s, [2.5, 97.5])

def pair_stats(a, b, df, label, mae_col=None):
    """mae_col: kalau diberikan, MAE diambil dari kolom itu (konvensi agregasi (a))
    alih-alih dari |a - b|. Bias/LoA/r tetap dari selisih, yang tidak terpengaruh."""
    cols = [a, b] + ([mae_col] if mae_col else [])
    dd = df[cols].dropna()
    if len(dd) < 5:
        return None
    diff = (dd[a] - dd[b]).to_numpy()
    absdiff = dd[mae_col].to_numpy() if mae_col else np.abs(diff)
    mae_ci = boot_ci(absdiff, np.mean)
    bias_ci = boot_ci(diff, np.mean)
    return dict(pair=label, n=len(dd),
                MAE=absdiff.mean(), MAE_lo=mae_ci[0], MAE_hi=mae_ci[1],
                bias=diff.mean(), bias_lo=bias_ci[0], bias_hi=bias_ci[1],
                LoA=1.96*diff.std(ddof=1),
                r=np.corrcoef(dd[a], dd[b])[0, 1])

out = {}
for split, pred, tag in [("TEST (401, bersih)", test_avg, "test"),
                         ("VAL pool (2275, data seleksi)", val, "val")]:
    cols = ["image_id", "cimt_pred_mm", "cimt_gt_mm", "dice"]
    if "err_foldmean" in pred:
        cols.append("err_foldmean")
    m = obs.merge(pred[cols], on="image_id", how="inner")
    ecol = "err_foldmean" if "err_foldmean" in m else None
    S = []
    # --- prosedur MASK (benar; satu prosedur untuk semua)
    for a, b, lab, ec in [("cimt_pred_mm", "msk_A1", "MODEL - A1        [mask/mask]  <-- BARU", ecol),
                          ("msk_A1p", "msk_A1",      "A1' - A1  (intra) [mask/mask]", None),
                          ("msk_A2",  "msk_A1",      "A2  - A1  (inter) [mask/mask]", None),
                          ("msk_A3",  "msk_A1",      "A3  - A1  (inter) [mask/mask]", None),
                          ("msk_A3",  "msk_A2",      "A3  - A2  (inter) [mask/mask]", None)]:
        s = pair_stats(a, b, m, lab, mae_col=ec)
        if s: S.append(s)
    # --- prosedur NATIVE (lama; untuk dokumentasi artefak)
    for a, b, lab in [("cimt_pred_mm", "nat_A1", "MODEL - A1        [mask/NATIVE] <-- LAMA (paper)"),
                      ("nat_A1p", "nat_A1",      "A1' - A1  (intra) [native]"),
                      ("nat_A2",  "nat_A1",      "A2  - A1  (inter) [native]"),
                      ("nat_A3",  "nat_A1",      "A3  - A1  (inter) [native]"),
                      ("nat_A3",  "nat_A2",      "A3  - A2  (inter) [native]"),
                      ("msk_A1",  "nat_A1",      "A1-raster - A1-native  (ARTEFAK MURNI, tanpa model)")]:
        s = pair_stats(a, b, m, lab)
        if s: S.append(s)
    out[split] = (pd.DataFrame(S), m)

# ---------------------------------------------------------------- 5) uji berpasangan model vs inter-observer
from scipy.stats import wilcoxon
def model_vs_inter(m, mkey="msk", ann="A2"):
    ec = "err_foldmean" if "err_foldmean" in m else None
    need = ["cimt_pred_mm", f"{mkey}_A1", f"{mkey}_{ann}"] + ([ec] if ec else [])
    dd = m[need].dropna()
    e_model = (dd[ec].to_numpy() if ec
               else (dd.cimt_pred_mm - dd[f"{mkey}_A1"]).abs().to_numpy())
    e_inter = (dd[f"{mkey}_{ann}"] - dd[f"{mkey}_A1"]).abs().to_numpy()
    d = e_inter - e_model
    ci = boot_ci(d, np.mean)
    p = wilcoxon(e_inter, e_model).pvalue
    return dict(n=len(dd), delta=d.mean(), lo=ci[0], hi=ci[1], p=p,
                ratio=e_model.mean()/e_inter.mean())

# ---------------------------------------------------------------- 6) tulis
L = ["# R1 — Perbandingan observer dengan SATU prosedur pengukuran",
     "",
     "Memperbaiki temuan fatal M7: E9 lama membandingkan CIMT model (mask biner) dengan",
     "CIMT anotator (profil native). Di sini **semua** anotator dirasterisasi dan diukur",
     "dengan pipeline yang persis sama seperti GT pelatihan A1 (fillPoly -> crop rect-A1",
     "-> 256x256 NEAREST -> per-kolom row_last-row_first x sq/256 x CF).",
     "",
     f"Sanity check reproduksi `cimt_gt_mm` dari CSV run: median |selisih| = {d.median():.6f} mm, "
     f"max = {d.max():.6f} mm, r = {np.corrcoef(chk.msk_A1, chk.cimt_gt_mm)[0,1]:.6f}",
     ""]
for split, (S, m) in out.items():
    L += [f"## {split}", "",
          "| Perbandingan | n | MAE (mm) [95% CI] | Bias (mm) [95% CI] | LoA ±1.96σ | r |",
          "|---|---|---|---|---|---|"]
    for _, s in S.iterrows():
        L.append("| %s | %d | **%.4f** [%.4f, %.4f] | %+.4f [%+.4f, %+.4f] | ±%.4f | %.3f |" % (
            s.pair, s.n, s.MAE, s.MAE_lo, s.MAE_hi, s.bias, s.bias_lo, s.bias_hi, s.LoA, s.r))
    st = model_vs_inter(m)
    L += ["", "Model vs inter-observer (A2-A1), prosedur mask, berpasangan per citra:",
          f"- Δ = {st['delta']:.4f} mm [95% CI {st['lo']:.4f}, {st['hi']:.4f}], Wilcoxon p = {st['p']:.3g}, n = {st['n']}",
          f"- rasio MAE model / MAE inter-observer = **{st['ratio']:.2f}×**", ""]
(RES/"R1_observer_matched.md").write_text("\n".join(L), encoding="utf-8")
obs.to_csv(RES/"R1_observer_matched.csv", index=False)

for split, (S, m) in out.items():
    print(f"\n===== {split} =====")
    print(S[["pair", "n", "MAE", "bias", "LoA", "r"]].round(4).to_string(index=False))
    print("model_vs_inter:", {k: round(v, 4) if isinstance(v, float) else v for k, v in model_vs_inter(m).items()})
print("\n-> results/R1_observer_matched.{md,csv}")
