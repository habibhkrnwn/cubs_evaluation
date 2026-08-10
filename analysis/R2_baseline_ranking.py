"""
R2 — Peringkat arsitektur: Spearman rho (M1) + standar bukti yang konsisten (M2, M3)
====================================================================================
1. M1: hitung Spearman rho antara peringkat-Dice dan peringkat-CIMT-MAE lintas arsitektur.
       Ini SATU-SATUNYA statistik yang menguji tesis "Dice adalah proxy peringkat yang lemah".
       + CI bootstrap atas CITRA (bukan atas model), + sensitivitas terhadap run U-Mamba mana.
2. M3: inventaris SEMUA run U-Mamba -> sebaran run-to-run; bandingkan dengan jarak antar-model.
3. M2: uji berpasangan Wilcoxon+BH pada test set, standar tunggal untuk semua klaim.

Output: results/R2_baseline_ranking.md
"""
from pathlib import Path
import glob, itertools, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
sys.stdout.reconfigure(encoding="utf-8")

def bh(pvals):
    """Benjamini-Hochberg FDR: kembalikan q-value (monotone step-up)."""
    p = np.asarray(pvals, float); n = len(p)
    o = np.argsort(p); q = np.empty(n)
    prev = 1.0
    for rank in range(n-1, -1, -1):
        i = o[rank]
        prev = min(prev, p[i]*n/(rank+1))
        q[i] = prev
    return q

REPO = Path(__file__).resolve().parents[1]
E2   = Path(os.environ.get("CUBS_EXPERIMENT_ROOT", REPO))
RES  = REPO/"results"
KG   = Path(os.environ.get("CUBS_RUNS_ROOT", E2/"external_results"))
BC   = KG/"baseline comparison"
RES.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------- inventaris run
RUNS = {
    "BC-VMamba (+Lcimt, s42)": KG/"result e1c/results/bcvmamba_a1/bc_vmamba_cubs_lcimt_s42",
    "BC-VMamba (region, s42)": KG/"result1/results/bcvmamba_a1/bc_vmamba_cubs_a1",
    "BC-VMamba (+Lcimt, s1)":  KG/"result e1c seed1/results/bcvmamba_a1/bc_vmamba_cubs_lcimt",
    "U-Net":            BC/"result(7)/results/baselines/unet",
    "Attention U-Net":  BC/"results(6)/results/baselines/attention_unet",
    "SegFormer":        BC/"results(5)/results/baselines/segformer",
    "Swin-UNet":        BC/"results(4)/results/baselines/swin_unet",
    "UNeXt":            BC/"results(5)/results/baselines/unext",
    # results(2) dan results(3) byte-identical -> SATU run. Jadi U-Mamba = 2 run berbeda.
    "U-Mamba [run 1]":  BC/"results(2)/results/baselines/umamba",
    "U-Mamba [run 2]":  BC/"results(4)/results/baselines/umamba",
    # Run 5-fold LENGKAP (2026-07-14). Menggantikan run parsial lama:
    #   ResUNet 2/5 fold di results(6), UNet++ 1/5 fold di result(7) -> JANGAN dipakai lagi.
    "UNet++":           BC/"unetpp_5fold/results/baselines/unetpp",
    "ResU-Net":         BC/"resunet/results/baselines/resunet",
    "TransUNet":        BC/"transunet/results/baselines/transunet",
}

def fold_metrics(run):
    """Sumber angka Tabel 2 manuskrip: *_fold_metrics.csv (metrik agregat per fold)."""
    fs = glob.glob(str(run/"*_fold_metrics.csv"))
    if not fs:
        return None
    d = pd.read_csv(fs[0])
    return dict(dice=d.dice.mean(), dice_sd=d.dice.std(),
                err=d.mae_cimt_mm.mean(), err_sd=d.mae_cimt_mm.std(),
                params=d.params_m.mean() if "params_m" in d else np.nan,
                gflops=d.gflops.mean() if "gflops" in d else np.nan,
                latency=d.latency_ms.mean() if "latency_ms" in d else np.nan)

def load(run, split):
    fs = sorted(glob.glob(str(run/f"{split}_fold*_per_image.csv")))
    if not fs:
        return None
    df = pd.concat([pd.read_csv(f).assign(fold=i+1) for i, f in enumerate(fs)], ignore_index=True)
    return df.rename(columns={"name": "image_id"})

def per_image_test(run):
    """Test set: 5 model fold menilai 401 citra yang sama -> rata-rata per citra."""
    t = load(run, "test")
    if t is None or t.image_id.nunique() < 300:
        return None
    return (t.groupby("image_id")
             .agg(dice=("dice", "mean"), err=("imt_abs_err_mm", "mean")).reset_index())

def cv_summary(run):
    """Metrik resmi dari fold_metrics.csv (sumber Tabel 2); n_folds dari CSV per-citra."""
    fm = fold_metrics(run)
    if fm is None:
        return None
    v = load(run, "val")
    fm["n_folds"] = v.fold.nunique() if v is not None else 5
    fm["n_img"] = v.image_id.nunique() if v is not None else 0
    return fm

L = ["# R2 — Peringkat arsitektur, dianalisis ulang", "",
     "Angka Dice/CIMT diambil dari `*_fold_metrics.csv` — **sumber yang sama dengan Tabel 2 manuskrip**,",
     "sehingga analisis ini tidak memperkenalkan angka baru, hanya statistik baru di atas angka lama.", "",
     "## 0. Inventaris run yang benar-benar ada di disk", "",
     "| Run | folds | citra val | Dice (CV) | CIMT MAE (CV, mm) |", "|---|---|---|---|---|"]
inv = {}
for name, run in RUNS.items():
    s = cv_summary(run)
    inv[name] = s
    if s is None:
        L.append(f"| {name} | — | — | *tidak ada fold_metrics* | — |")
    else:
        L.append(f"| {name} | {s['n_folds']} | {s['n_img']} | {s['dice']:.4f} ± {s['dice_sd']:.4f} | "
                 f"{s['err']:.4f} ± {s['err_sd']:.4f} |")

# ---------------------------------------------------------------- M3: sebaran run-to-run U-Mamba
um = {k: v for k, v in inv.items() if k.startswith("U-Mamba") and v}
errs = np.array([v["err"] for v in um.values()])
dices = np.array([v["dice"] for v in um.values()])
gap12 = abs(inv["Swin-UNet"]["err"] - inv["BC-VMamba (region, s42)"]["err"])
L += ["", "## 1. M3 — Sebaran run-to-run (U-Mamba), seed nominal sama", "",
      f"- CIMT MAE: {', '.join(f'{e:.4f}' for e in errs)}  ->  sebaran **{errs.max()-errs.min():.4f} mm**",
      f"- Dice:     {', '.join(f'{d:.4f}' for d in dices)}  ->  sebaran **{dices.max()-dices.min():.4f}**",
      f"- Manuskrip Tabel 2 memakai run yang **lebih buruk** ({errs.max():.4f} mm).",
      "",
      f"**Jarak juara-1 ke juara-2 di Tabel 2** (BC-VMamba region {inv['BC-VMamba (region, s42)']['err']:.4f} vs "
      f"Swin-UNet {inv['Swin-UNet']['err']:.4f}) = **{gap12:.4f} mm** = "
      f"**{gap12/(errs.max()-errs.min()):.2f}x** sebaran run-to-run yang terukur di baseline yang sama.",
      "",
      "> Setiap sel Tabel 2 adalah SATU run. Pada skala yang lebih halus dari noise run-to-run yang",
      "> sudah terbukti ada, peringkat single-run tidak dapat dipertahankan.", ""]

# ---------------------------------------------------------------- M1: Spearman rho antar arsitektur
# Set arsitektur untuk peringkat: SEMUA region-only (loss dikunci), U-Mamba dipilih per-varian.
ARCH_BASE = ["U-Net", "Attention U-Net", "SegFormer", "Swin-UNet",
             "UNet++", "ResU-Net", "TransUNet", "BC-VMamba (region, s42)"]
UM_VARIANTS = [k for k in um]

def rho_for(um_choice, include_unext=False):
    names = ARCH_BASE + [um_choice] + (["UNeXt"] if include_unext else [])
    d = np.array([inv[n]["dice"] for n in names])
    e = np.array([inv[n]["err"] for n in names])
    rho, p = spearmanr(-d, e)     # -Dice (besar=buruk) vs error (besar=buruk): rho>0 = peringkat sepakat
    return names, d, e, rho, p

L += ["## 2. M1 — Spearman rho: apakah peringkat-Dice sepakat dengan peringkat-CIMT?", "",
      "Konvensi: rho dihitung antara (−Dice) dan CIMT MAE, keduanya \"besar = buruk\".",
      "rho = +1 berarti kedua metrik meranking arsitektur **identik** (Dice = proxy sempurna);",
      "rho = 0 berarti peringkatnya tak berhubungan (Dice = proxy tak berguna).", "",
      "| Run U-Mamba dipakai | k arsitektur | Spearman rho | p |", "|---|---|---|---|"]
rhos = {}
for umc in UM_VARIANTS:
    names, d, e, rho, p = rho_for(umc)
    rhos[umc] = rho
    L.append(f"| {umc} | {len(names)} | **{rho:+.3f}** | {p:.3f} |")
for umc in UM_VARIANTS:
    names, d, e, rho, p = rho_for(umc, include_unext=True)
    L.append(f"| {umc} + UNeXt | {len(names)} | {rho:+.3f} | {p:.3f} |")

# bootstrap CI atas CITRA (bukan atas model): resample citra test, hitung ulang mean per model, lalu rho
tests = {}
for n in ARCH_BASE + UM_VARIANTS:
    t = per_image_test(RUNS[n])
    if t is not None:
        tests[n] = t.set_index("image_id")
common = sorted(set.intersection(*[set(t.index) for t in tests.values()]))
L += ["", f"Bootstrap CI atas citra test (n = {len(common)} citra yang dimiliki semua model, 5.000 resample):", ""]
rng = np.random.default_rng(42)
for umc in UM_VARIANTS:
    names = ARCH_BASE + [umc]
    D = np.column_stack([tests[n].loc[common, "dice"].to_numpy() for n in names])
    E = np.column_stack([tests[n].loc[common, "err"].to_numpy() for n in names])
    boot = []
    for _ in range(5000):
        idx = rng.integers(0, len(common), len(common))
        boot.append(spearmanr(-D[idx].mean(0), E[idx].mean(0)).statistic)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    names_, d_, e_, rho_, p_ = rho_for(umc)
    L.append(f"- U-Mamba {umc.split('[')[-1].rstrip(']')}: rho = **{rho_:+.3f}**, 95% CI [{lo:+.3f}, {hi:+.3f}]")
K_ARCH = len(ARCH_BASE) + 1
pmin = min(rho_for(u)[4] for u in UM_VARIANTS)
L += ["",
      f"**rho berayun {min(rhos.values()):+.2f} .. {max(rhos.values()):+.2f} hanya dengan mengganti run U-Mamba mana yang dipakai.** "
      f"k = {K_ARCH} arsitektur; p terkecil di antara kedua pilihan run U-Mamba = {pmin:.3f} "
      f"({'signifikan' if pmin < 0.05 else 'TIDAK signifikan'} pada alfa=0.05). "
      "Klaim yang boleh ditarik hanya sejauh p mengizinkan.", ""]

# tabel peringkat eksplisit
names, d, e, rho, p = rho_for(UM_VARIANTS[0])
rk = pd.DataFrame(dict(model=names, dice=d, err=e))
rk["rank_dice"] = rk.dice.rank(ascending=False).astype(int)
rk["rank_err"] = rk.err.rank(ascending=True).astype(int)
rk = rk.sort_values("rank_err")
L += ["Peringkat (region-only, U-Mamba run A):", "",
      "| Model | Dice | peringkat Dice | CIMT MAE | peringkat CIMT |", "|---|---|---|---|---|"]
for _, r in rk.iterrows():
    L.append(f"| {r.model} | {r.dice:.4f} | {r.rank_dice} | {r.err:.4f} | {r.rank_err} |")

# ---------------------------------------------------------------- M2: uji berpasangan, satu standar
PROP = "BC-VMamba (+Lcimt, s42)"
tests[PROP] = per_image_test(RUNS[PROP]).set_index("image_id")
tests["BC-VMamba (region, s42)"] = per_image_test(RUNS["BC-VMamba (region, s42)"]).set_index("image_id")
common = sorted(set.intersection(*[set(t.index) for t in tests.values()]))
L += ["", "## 3. M2 — Uji berpasangan pada test set, standar tunggal", "",
      f"n = {len(common)} citra. Δ positif = model usulan lebih baik. q = Wilcoxon + Benjamini-Hochberg.", "",
      "| Pembanding | ΔDice | q(Dice) | ΔCIMT err (mm) | q(CIMT) | putusan endpoint |",
      "|---|---|---|---|---|---|"]
comps = [n for n in tests if n != PROP and not n.startswith("BC-VMamba (+")]
raw = []
for n in comps:
    a, b = tests[PROP].loc[common], tests[n].loc[common]
    dd = (a.dice - b.dice).to_numpy(); de = (b.err - a.err).to_numpy()
    raw.append((n, dd.mean(), wilcoxon(a.dice, b.dice).pvalue, de.mean(), wilcoxon(a.err, b.err).pvalue))
ps = [r[2] for r in raw] + [r[4] for r in raw]
qs = bh(ps)
k = len(raw)
for i, (n, dd, _, de, _) in enumerate(raw):
    qd, qe = qs[i], qs[k+i]
    verdict = "**lebih baik**" if (qe < 0.05 and de > 0) else ("**lebih buruk**" if (qe < 0.05 and de < 0) else "*seri*")
    L.append(f"| {n} | {dd:+.4f} | {qd:.2g} | {de:+.4f} | {qe:.2g} | {verdict} |")
L += ["",
      "> Standar bukti yang konsisten: jika selisih 0.0472 mm terhadap Swin-UNet ditolak sebagai",
      "> tidak nyata (q > 0.05), maka klaim *'lowest CIMT error of the tested models'* tidak boleh",
      "> dibuat dari rata-rata. Klaim yang didukung: **seri dengan Swin-UNet pada endpoint, dengan",
      "> ~1/3.5 parameter**, dan lebih baik dari sisanya.", ""]

(RES/"R2_baseline_ranking.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print("\n-> results/R2_baseline_ranking.md")
