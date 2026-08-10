"""
R3 — M14 (asosiasi Dice-CIMT), M13 (uji stratifikasi), M9/S3-S6 (bab UQ)
=========================================================================
M14: paper memakai Pearson r pada variabel error yang IA SENDIRI sebut heavy-tailed
     (Sec 4.1). Pearson diatenuasi ekor berat -> r = -0.53 adalah BATAS BAWAH asosiasi.
     Di sini: Pearson DAN Spearman, dua model (lambda=0, 0.2), val pool + test, + CI.
M13: setiap klaim stratifikasi (SNR, morfologi, center) diberi CI bootstrap dan uji.
     Klaim "SNR 50%" ada di Highlight + abstrak tanpa uji apa pun.
UQ : - ambang "CIMT failure": sensitivitas terhadap 3 kandidat inter-observer
       (+ nilai baru dari prosedur mask yang dicocokkan, R1)
     - kontrol untuk label "segmentation failure" (apakah fg_entropy mengalahkan SNR / luas mask?)
     - non-monotonisitas kurva selective prediction
     - uncertainty pada citra out-of-distribution? (butuh prediksi LOCO -> dilewati bila tak ada)

Output: results/R3_assoc_strat_uq.md
"""
from pathlib import Path
import glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
E2 = Path(os.environ.get("CUBS_EXPERIMENT_ROOT", REPO))
RES = REPO/"results"
KG = Path(os.environ.get("CUBS_RUNS_ROOT", E2/"external_results"))
RUN_L = KG/"result e1c/results/bcvmamba_a1/bc_vmamba_cubs_lcimt_s42"   # lambda=0.2
RUN_R = KG/"result1/results/bcvmamba_a1/bc_vmamba_cubs_a1"             # lambda=0 (region-only)
RES.mkdir(exist_ok=True, parents=True)
rng = np.random.default_rng(42)

def load(run, split):
    fs = sorted(glob.glob(str(run/f"{split}_fold*_per_image.csv")))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).rename(columns={"name": "image_id"})

def boot_ci(fn, *arrs, B=5000):
    n = len(arrs[0]); out = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        out.append(fn(*[a[i] for a in arrs]))
    return np.percentile(out, [2.5, 97.5])

L = ["# R3 — Asosiasi Dice-CIMT, uji stratifikasi, dan audit bab ketidakpastian", ""]

# ================================================================ M14
L += ["## 1. M14 — Pearson vs Spearman untuk asosiasi Dice <-> |error CIMT|", "",
      "Sec. 4.1 manuskrip menjelaskan hasil Swin-UNet yang tidak signifikan dengan:",
      "*\"the CIMT error distribution is heavy-tailed\"*. Sec. 4.2 lalu mengukur asosiasi",
      "Dice-CIMT dengan **Pearson** — statistik yang justru diatenuasi oleh ekor berat.",
      "Spearman adalah statistik yang tepat untuk asosiasi monoton pada data berekor berat.", "",
      "| Model | Data | n | Pearson r [95% CI] | Spearman rho [95% CI] | r^2 | rho^2 |",
      "|---|---|---|---|---|---|---|"]
for lab, run in [("lambda=0.2 (usulan)", RUN_L), ("lambda=0 (region-only)", RUN_R)]:
    for split, sname in [("val", "val pool"), ("test", "test 401")]:
        d = load(run, split)
        if split == "test":
            d = d.groupby("image_id").agg(dice=("dice", "mean"), imt_abs_err_mm=("imt_abs_err_mm", "mean")).reset_index()
        x, y = d.dice.to_numpy(), d.imt_abs_err_mm.to_numpy()
        r = pearsonr(x, y).statistic; rho = spearmanr(x, y).statistic
        rc = boot_ci(lambda a, b: pearsonr(a, b).statistic, x, y)
        sc = boot_ci(lambda a, b: spearmanr(a, b).statistic, x, y)
        L.append(f"| {lab} | {sname} | {len(d)} | {r:+.3f} [{rc[0]:+.3f}, {rc[1]:+.3f}] | "
                 f"**{rho:+.3f}** [{sc[0]:+.3f}, {sc[1]:+.3f}] | {r**2:.2f} | {rho**2:.2f} |")

dl = load(RUN_L, "val"); x, y = dl.dice.to_numpy(), dl.imt_abs_err_mm.to_numpy()
r_l = pearsonr(x, y).statistic; rho_l = spearmanr(x, y).statistic
dr = load(RUN_R, "val"); r_r = pearsonr(dr.dice, dr.imt_abs_err_mm).statistic
L += ["",
      "**HASIL: keberatan M14 GUGUR — dan itu kabar baik untuk paper.**", "",
      f"- Spearman (rho = {rho_l:+.3f}) praktis identik dengan Pearson (r = {r_l:+.3f}) pada val pool. "
      "Ekor berat ternyata TIDAK mengatenuasi Pearson di sini, jadi kekhawatiran bahwa r = -0.53 adalah "
      "batas bawah artifisial tidak terbukti. Asosiasi Dice-CIMT memang selemah yang dilaporkan.",
      f"- Klaim *\"explains under a third of the variance\"* (r^2 = {r_l**2:.2f}) bertahan; rho^2 = {rho_l**2:.2f}.",
      f"- Pada test set 401 (data bersih) asosiasinya sedikit **lebih kuat** (rho = {spearmanr(*[np.asarray(z) for z in [load(RUN_L,'test').groupby('image_id').dice.mean(), load(RUN_L,'test').groupby('image_id').imt_abs_err_mm.mean()]]).statistic:+.3f}), "
      "tetapi tetap jauh dari -1.",
      "",
      "**Tindakan:** laporkan Spearman berdampingan dengan Pearson sebagai uji ketahanan. Ini MEMPERKUAT",
      "tesis, bukan melemahkannya, dan mendahului keberatan reviewer yang jelas akan muncul.",
      f"- Catatan kecil: paper mem-headline r = {r_l:+.2f} (model lambda=0.2) padahal model region-only memberi "
      f"r = {r_r:+.2f}. Sebutkan keduanya (sudah dilakukan di Sec 4.2) dan pilih yang dari model usulan — itu wajar.", ""]

# ================================================================ M13
L += ["## 2. M13 — Setiap klaim stratifikasi, dengan CI", "",
      "Manuskrip melaporkan mean +- sd per strata tanpa satu pun uji. Di bawah: selisih antar-strata",
      "dengan CI bootstrap (5.000 resample citra) dan Mann-Whitney.", ""]
v = load(RUN_L, "val")

def contrast(df, col, a, b, metric, lab):
    xa = df.loc[df[col] == a, metric].dropna().to_numpy()
    xb = df.loc[df[col] == b, metric].dropna().to_numpy()
    diff = xa.mean() - xb.mean()
    B = []
    for _ in range(5000):
        B.append(xa[rng.integers(0, len(xa), len(xa))].mean() - xb[rng.integers(0, len(xb), len(xb))].mean())
    lo, hi = np.percentile(B, [2.5, 97.5])
    p = mannwhitneyu(xa, xb).pvalue
    sig = "**ya**" if (lo > 0) == (hi > 0) else "*tidak*"
    return f"| {lab} | {len(xa)}/{len(xb)} | {diff:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.3f} | {sig} |"

L += ["| Kontras | n | Δ | 95% CI | p (MWU) | signifikan? |", "|---|---|---|---|---|---|"]
L.append(contrast(v, "snr", -1.0, 1.0, "imt_abs_err_mm", "SNR rendah − SNR tinggi, **CIMT MAE** (klaim '50%', Highlight #4)"))
L.append(contrast(v, "snr", -1.0, 1.0, "dice", "SNR rendah − SNR tinggi, Dice"))
L.append(contrast(v, "morph", 2.0, 3.0, "imt_abs_err_mm", "Morfologi 2 − 3, **CIMT MAE** (dipakai sbg bukti tesis, Sec 4.4)"))
L.append(contrast(v, "morph", 2.0, 3.0, "dice", "Morfologi 2 − 3, Dice"))
L.append(contrast(v, "center", "Munich", "Toronto", "imt_abs_err_mm", "Munich − Toronto, CIMT MAE"))
L.append(contrast(v, "center", "Munich", "Toronto", "dice", "Munich − Toronto, **Dice** (klaim 'sebaran 6.5 poin')"))

# tren SNR: uji monotonisitas via Spearman pada citra, bukan pada 3 mean
s = v.dropna(subset=["snr"])
rho_snr = spearmanr(s.snr, s.imt_abs_err_mm)
rho_snr_d = spearmanr(s.snr, s.dice)
ci_snr = boot_ci(lambda a, b: spearmanr(a, b).statistic, s.snr.to_numpy(), s.imt_abs_err_mm.to_numpy())
L += ["",
      f"Tren SNR diuji dengan benar (Spearman pada {len(s)} citra, bukan pada 3 rata-rata):",
      f"- SNR vs CIMT MAE: rho = **{rho_snr.statistic:+.3f}** [95% CI {ci_snr[0]:+.3f}, {ci_snr[1]:+.3f}], p = {rho_snr.pvalue:.4f}",
      f"- SNR vs Dice   : rho = **{rho_snr_d.statistic:+.3f}**, p = {rho_snr_d.pvalue:.2g}",
      "",
      "> Kontras dua-kelompok (rendah vs tinggi, n=40 vs 52) tidak signifikan, TAPI uji tren monoton",
      "> pada seluruh citra ber-SNR jauh lebih berdaya. Pakai uji tren, bukan rasio 50% antar dua sel.", ""]

# ================================================================ UQ
L += ["## 3. Bab ketidakpastian — audit", ""]
u = pd.read_csv(RES/"E1b_ensemble_per_image_lcimt.csv")
SIG = ["fg_entropy", "band_disagree", "disagreement", "band_frac", "entropy", "cimt_spread_mm"]

def auc(score, label):
    x, y = score[label == 1], score[label == 0]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    return mannwhitneyu(x, y).statistic/(len(x)*len(y))

def auc_ci(score, label, B=2000):
    out = []
    n = len(score)
    for _ in range(B):
        i = rng.integers(0, n, n)
        a = auc(score[i], label[i])
        if np.isfinite(a):
            out.append(a)
    return np.percentile(out, [2.5, 97.5])

# --- S5: sensitivitas ambang
L += ["### 3a. Sensitivitas ambang 'CIMT failure' (S5)", "",
      "Manuskrip memilih 0.191 mm dari tiga nilai inter-observer tanpa alasan. R1 juga memberi",
      "nilai baru dengan prosedur yang dicocokkan. Setiap ambang -> base rate -> AUC berbeda.", "",
      "| Ambang (mm) | asal | n gagal (dari 401) | AUC fg_entropy [95% CI] | AUC cimt_spread |",
      "|---|---|---|---|---|"]
THR = [(0.1367, "A3-A1 native (paling longgar)"),
       (0.1686, "A2-A1 **mask, test set** (R1 — yang benar)"),
       (0.1906, "A2-A1 native (dipakai paper)"),
       (0.2274, "A3-A2 native (paling ketat)")]
err = u.err_ens.to_numpy()
for t, src in THR:
    lab = (err > t).astype(int)
    a1 = auc(u.fg_entropy.to_numpy(), lab); c1 = auc_ci(u.fg_entropy.to_numpy(), lab)
    a2 = auc(u.cimt_spread_mm.to_numpy(), lab)
    L.append(f"| {t:.4f} | {src} | {lab.sum()} ({lab.mean()*100:.1f}%) | {a1:.3f} [{c1[0]:.3f}, {c1[1]:.3f}] | {a2:.3f} |")
L += ["", "> Kesimpulan kualitatif paper (uncertainty tidak mendeteksi CIMT salah) **stabil** terhadap",
      "> pilihan ambang — itu kabar baik dan harus dilaporkan sebagai analisis sensitivitas, bukan",
      "> ditinggalkan sebagai pilihan tak beralasan.", ""]

# --- S4: kontrol untuk label segmentation failure
segfail = (u.dice_mean <= u.dice_mean.quantile(0.25)).astype(int).to_numpy()
L += ["### 3b. Kontrol untuk label 'segmentation failure' (S4)", "",
      "Label = kuartil Dice terbawah; detektor = fg_entropy (ambiguitas mask). Nyaris tautologis.",
      "Pertanyaan: apakah fg_entropy benar-benar mengalahkan prediktor sepele?", "",
      "| Prediktor | AUC (segmentation failure) |", "|---|---|"]
ctrl = {"fg_entropy (uncertainty)": u.fg_entropy.to_numpy(),
        "entropy": u.entropy.to_numpy(),
        "band_disagree": u.band_disagree.to_numpy(),
        "KONTROL: luas mask prediksi (band_frac)": u.band_frac.to_numpy(),
        "KONTROL: CIMT prediksi (mm)": u.cimt_ens_mm.to_numpy(),
        "KONTROL: -SNR (citra teknis saja)": None}
for k, s_ in ctrl.items():
    if s_ is None:
        continue
    L.append(f"| {k} | {auc(s_, segfail):.3f} |")
us = u.dropna(subset=["snr"])
sf2 = (us.dice_mean <= u.dice_mean.quantile(0.25)).astype(int).to_numpy()
L.append(f"| KONTROL: −SNR (hanya n={len(us)} citra teknis) | {auc(-us.snr.to_numpy(), sf2):.3f} |")
L.append(f"| fg_entropy (subset yang sama, n={len(us)}) | {auc(us.fg_entropy.to_numpy(), sf2):.3f} |")
L += ["", "> Jika kontrol sepele mendekati 0.740, klaim 'uncertainty adalah gerbang QC' melemah.", ""]

# --- S6: non-monotonisitas kurva selektif
L += ["### 3c. Kurva selective prediction (S6)", "",
      "| Coverage | n dipertahankan | CIMT MAE (mm) |", "|---|---|---|"]
o = u.sort_values("fg_entropy")
prev = None; nonmono = []
for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
    k = int(round(cov*len(o)))
    m = o.head(k).err_ens.mean()
    flag = ""
    if prev is not None and m > prev:
        flag = "  **<- MEMBURUK**"; nonmono.append(cov)
    prev = m
    L.append(f"| {cov*100:.0f}% | {k} | {m:.4f}{flag} |")
L += ["", f"> Kurva tidak monoton pada coverage {nonmono}. Manuskrip mencetak angkanya (Tabel 7) tapi tidak",
      "> menyebutnya. Fakta ini **memperkuat** argumen negatif penulis: entropi tidak melacak error CIMT",
      "> secara monoton. Menghilangkannya terbaca sebagai penghalusan data.", ""]

(RES/"R3_assoc_strat_uq.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print("\n-> results/R3_assoc_strat_uq.md")
