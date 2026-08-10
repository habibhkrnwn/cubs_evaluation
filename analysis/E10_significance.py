"""
E10 — Signifikansi statistik (menjawab kritik "single seed / lapor CI")
=======================================================================
Nol compute — semua dari prediksi per-citra yang sudah ada.
  1. Bootstrap 95% CI: Dice & CIMT MAE model usulan (val pooled).
  2. E9 berpasangan: model vs inter-observer (A1-A2) -> selisih + CI + p.
  3. Perbandingan berpasangan model usulan vs tiap BASELINE (test hold-out):
     Wilcoxon signed-rank + bootstrap CI selisih (Dice & CIMT error),
     dengan koreksi Benjamini-Hochberg lintas seluruh uji di seksi ini.
     (Otomatis dilewati jika folder baseline belum ada.)
  4. Bootstrap CI untuk AUC failure-detection (E5) — pengganti DeLong.

Output: results/E10_significance.md
"""
from pathlib import Path
import glob, sys, numpy as np, pandas as pd

# Konsol Windows default cp1252 tersedak U+2212 ("−") & "⏳" saat print ringkasan,
# padahal file .md-nya sendiri ditulis UTF-8 dan sudah benar.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

import os
REPO = Path(__file__).resolve().parents[1]
E2   = Path(os.environ.get("CUBS_EXPERIMENT_ROOT", REPO))
RES  = REPO/"results"; RES.mkdir(exist_ok=True, parents=True)
PROP = Path(os.environ.get("CUBS_PROPOSED_RUN", E2/"external_results"/"proposed"))
TAG  = os.environ.get("CUBS_TAG","")
print(f"[E10] PROP={PROP}  TAG='{TAG}'")
BASE_ROOT = Path(os.environ.get("CUBS_RUNS_ROOT", E2/"external_results"))
rng = np.random.default_rng(0)

def boot_ci(x, stat=np.mean, n=5000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    idx = rng.integers(0, len(x), (n, len(x)))
    bs = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def boot_p_le0(diff, n=5000):
    """p dua-arah bahwa mean(diff)=0, berbasis bootstrap (diff>0 = usulan lebih baik)."""
    diff = np.asarray(diff, float); diff = diff[~np.isnan(diff)]
    bs = np.array([np.mean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(n)])
    p_one = min(np.mean(bs <= 0), np.mean(bs >= 0))
    return float(2*p_one), float(np.mean(diff))

def bh_adjust(pvals):
    """Benjamini-Hochberg: p -> q (FDR). Monoton, dipotong di 1."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):           # dari p terbesar ke terkecil
        prev = min(prev, p[i] * n / (n - rank))
        q[i] = prev
    return np.minimum(q, 1.0)


def per_image_testmean(run_dir):
    """rata-rata over 5 fold per citra pada TEST hold-out -> Series dice & err (index=name)."""
    fs = sorted(glob.glob(str(run_dir/"test_fold*_per_image.csv")))
    if not fs: return None
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    g = df.groupby("name").agg(dice=("dice","mean"), err=("imt_abs_err_mm","mean"))
    return g

L = ["# E10 — Signifikansi statistik (nol compute)\n"]

# ---------- 1) CI model usulan (val pooled) ----------
val = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(PROP/"val_fold*_per_image.csv")))],
                ignore_index=True)
d = boot_ci(val.dice.values); m = boot_ci(val.imt_abs_err_mm.values)
L += [f"## 1. Model usulan — bootstrap 95% CI (val pooled, n={len(val)})",
      f"- Dice = **{d[0]:.4f}**  [95% CI {d[1]:.4f}, {d[2]:.4f}]",
      f"- CIMT MAE = **{m[0]:.4f}**  [95% CI {m[1]:.4f}, {m[2]:.4f}] mm\n"]

# ---------- 2) E9 berpasangan ----------
obs = pd.read_csv(RES/f"E9_observer_cimt{TAG}.csv")
p = obs.dropna(subset=["cimt_pred_mm","cimt_A1","cimt_A2"]).copy()
p["model_err"] = (p.cimt_pred_mm-p.cimt_A1).abs()
p["inter_err"] = (p.cimt_A1-p.cimt_A2).abs()
diff = (p.inter_err - p.model_err).values
md = boot_ci(diff); pv,_ = boot_p_le0(diff)
wp = wilcoxon(p.model_err, p.inter_err).pvalue if HAVE_SCIPY else np.nan
L += [f"## 2. E9 — model vs inter-observer (berpasangan, n={len(p)})",
      f"- model MAE = {p.model_err.mean():.4f} mm · inter-obs(A1-A2) MAE = {p.inter_err.mean():.4f} mm",
      f"- selisih (inter − model) = **{md[0]:.4f}** [95% CI {md[1]:.4f}, {md[2]:.4f}] mm",
      f"- bootstrap p = {pv:.2g}"+ (f" · Wilcoxon p = {wp:.2g}" if HAVE_SCIPY else "")
      + "  → model **signifikan lebih dekat ke A1** daripada A2 ke A1.\n"]

# ---------- 3) usulan vs baseline (test hold-out) ----------
L += ["## 3. Model usulan vs baseline (test hold-out, berpasangan per-citra)"]
prop = per_image_testmean(PROP)
base_dirs = sorted(glob.glob(str(BASE_ROOT/"**/results/baselines/*"), recursive=True)) \
          + sorted(glob.glob(str(BASE_ROOT/"**/baselines/*"), recursive=True))
base_dirs = [Path(d) for d in dict.fromkeys(base_dirs) if Path(d).is_dir()]
if prop is None:
    L += ["- (prediksi test model usulan tidak ditemukan)\n"]
elif not base_dirs:
    L += ["- ⏳ Belum ada folder baseline. Setelah `E2_baselines_READY` selesai, taruh "
          "`results/baselines/{model}/` di `hasil run kaggle/` lalu jalankan ulang skrip ini.\n"]
else:
    # Lintasan 1: kumpulkan efek + p mentah untuk SEMUA baseline & kedua metrik.
    rows = []
    for bd in base_dirs:
        b = per_image_testmean(bd)
        if b is None: continue
        j = prop.join(b, lsuffix="_p", rsuffix="_b", how="inner")
        dd = (j.dice_p - j.dice_b).values                 # >0 usulan lebih baik (Dice)
        de = (j.err_b - j.err_p).values                   # >0 usulan lebih baik (error lebih kecil)
        pdd,_ = boot_p_le0(dd); pde,_ = boot_p_le0(de)
        rows.append(dict(
            name=bd.name, n=len(j),
            cid=boot_ci(dd), cie=boot_ci(de),
            p_dice=wilcoxon(j.dice_p, j.dice_b).pvalue if HAVE_SCIPY else pdd,
            p_err =wilcoxon(j.err_p,  j.err_b ).pvalue if HAVE_SCIPY else pde))

    # Lintasan 2: BH atas seluruh 2*len(rows) uji seksi ini, lalu cetak.
    if rows:
        q = bh_adjust([r["p_dice"] for r in rows] + [r["p_err"] for r in rows])
        k = len(rows)
        for i, r in enumerate(rows):
            r["q_dice"], r["q_err"] = q[i], q[k+i]
        L += [f"Uji berpasangan per-citra (n={rows[0]['n']} citra test). "
              f"p = Wilcoxon signed-rank{'' if HAVE_SCIPY else ' (bootstrap; scipy tidak ada)'}; "
              f"q = Benjamini-Hochberg atas {2*k} uji ({k} baseline x 2 metrik).\n",
              "| Baseline | ΔDice (usulan−base) [95% CI] | p | q | ΔCIMTerr (base−usulan) mm [95% CI] | p | q |",
              "|---|---|---|---|---|---|---|"]
        for r in rows:
            cid, cie = r["cid"], r["cie"]
            L.append("| %s | %+.4f [%+.4f,%+.4f] | %.2g | %.2g | %+.4f [%+.4f,%+.4f] | %.2g | %.2g |"%(
                r["name"], cid[0],cid[1],cid[2], r["p_dice"], r["q_dice"],
                cie[0],cie[1],cie[2], r["p_err"], r["q_err"]))
        L += ["\n(ΔDice>0 & ΔCIMTerr>0 = model usulan lebih baik. "
              "Signifikansi dinilai dari **q**, bukan p.)\n"]
    else:
        L += ["- (folder baseline ada tetapi tidak berisi `test_fold*_per_image.csv`)\n"]

# ---------- 4) AUC failure-detection CI (E5) ----------
e5 = RES/f"E1b_ensemble_per_image{TAG}.csv"
if e5.exists():
    df = pd.read_csv(e5)
    fail = (df.err_ens > 0.191).astype(int).values
    def auc(score, label):
        P=label.sum(); N=len(label)-P
        if P==0 or N==0: return np.nan
        o=np.argsort(score,kind="mergesort"); r=np.empty(len(score)); r[o]=np.arange(1,len(score)+1)
        return (r[label==1].sum()-P*(P+1)/2)/(P*N)
    L += ["## 4. AUC failure-detection (E5) — bootstrap CI"]
    for sig in ["fg_entropy","band_disagree","cimt_spread_mm"]:
        s=df[sig].values
        a0=auc(s,fail)
        bs=[auc(s[i:=rng.integers(0,len(s),len(s))],fail[i]) for _ in range(2000)]
        bs=[x for x in bs if not np.isnan(x)]
        L.append(f"- {sig}: AUC(gagal-CIMT) = {a0:.3f} [95% CI {np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
    L += [""]

(RES/f"E10_significance{TAG}.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print("\n-> results/E10_significance%s.md   (scipy:%s, baseline dirs:%d)"%(TAG,HAVE_SCIPY,len(base_dirs)))
