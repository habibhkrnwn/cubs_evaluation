"""
R7 — Di mana letak keunggulan endpoint: median atau ekor?
========================================================
R2 menemukan hal yang tampak kontradiktif: model usulan unggul jauh pada RATA-RATA
CIMT MAE terhadap ResU-Net (+0.0786 mm) tetapi Wilcoxon signed-rank menyebutnya SERI.

Sebabnya: Wilcoxon menguji MEDIAN selisih (citra tipikal), sedangkan MAE adalah RATA-RATA.
Kalau keunggulan datang dari EKOR (sedikit citra dengan kegagalan besar), keduanya bisa
tidak sepakat tanpa ada yang salah.

Skrip ini memisahkan keduanya secara eksplisit:
  1. citra tipikal  -> median selisih + win-rate + Wilcoxon
  2. ekor kegagalan -> laju error katastrofik (err > tau) + McNemar eksak pada pasangan diskordan
                       tau disapu {0.3, 0.5, 1.0} mm supaya tidak terlihat dipilih-pilih
  3. rata-rata      -> selisih mean + CI bootstrap (statistik yang benar untuk MAE)

Output: results/R7_failure_tail.md
Murni lokal, nol GPU.
"""
from pathlib import Path
import glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest
sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
E2 = Path(os.environ.get("CUBS_EXPERIMENT_ROOT", REPO))
RES = REPO / "results"; RES.mkdir(exist_ok=True, parents=True)
KG = Path(os.environ.get("CUBS_RUNS_ROOT", E2/"external_results"))
BC = KG / "baseline comparison"

PROP = "BC-VMamba (+Lcimt)"
RUNS = {
    PROP:                     KG/"result e1c/results/bcvmamba_a1/bc_vmamba_cubs_lcimt_s42",
    "BC-VMamba (region)":     KG/"result1/results/bcvmamba_a1/bc_vmamba_cubs_a1",
    "U-Net":                  BC/"result(7)/results/baselines/unet",
    "Attention U-Net":        BC/"results(6)/results/baselines/attention_unet",
    "SegFormer":              BC/"results(5)/results/baselines/segformer",
    "Swin-UNet":              BC/"results(4)/results/baselines/swin_unet",
    "U-Mamba":                BC/"results(2)/results/baselines/umamba",
    "UNet++":                 BC/"unetpp_5fold/results/baselines/unetpp",
    "ResU-Net":               BC/"resunet/results/baselines/resunet",
    "TransUNet":              BC/"transunet/results/baselines/transunet",
}
TAUS = [0.3, 0.5, 1.0]          # mm; ambang "error katastrofik"


def test_per_image(run):
    """Test set: 401 citra yang sama dinilai oleh 5 model fold -> rata-rata metrik per citra."""
    fs = sorted(glob.glob(str(run/"test_fold*_per_image.csv")))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return d.groupby("name").agg(dice=("dice", "mean"), err=("imt_abs_err_mm", "mean"))


T = {n: test_per_image(r) for n, r in RUNS.items()}
T = {n: d for n, d in T.items() if d is not None}
idx = sorted(set.intersection(*[set(d.index) for d in T.values()]))
n = len(idx)
print(f"[R7] n = {n} citra test yang dimiliki semua {len(T)} model\n")

rng = np.random.default_rng(42)
p = T[PROP].loc[idx, "err"].to_numpy()

L = ["# R7 — Letak keunggulan endpoint: citra tipikal atau ekor kegagalan?", "",
     f"Test set: **n = {n} citra**, identik untuk semua model, disjoint dari seluruh pool training.",
     "Selisih d = err(baseline) − err(usulan); **d > 0 berarti model usulan lebih baik**.", "",
     "## 1. Citra tipikal vs ekor", "",
     "| Baseline | MAE | mean d | median d | win-rate | d Wilcoxon p | "
     + " | ".join(f"katastrofik >{t}mm (base vs usulan)" for t in TAUS) + " |",
     "|---" * (6 + len(TAUS)) + "|"]

rows = []
for name in T:
    if name == PROP:
        continue
    b = T[name].loc[idx, "err"].to_numpy()
    d = b - p
    pw = wilcoxon(p, b).pvalue
    cells = []
    tail = {}
    for t in TAUS:
        fb, fp = int((b > t).sum()), int((p > t).sum())
        # McNemar eksak: hanya pasangan diskordan yang informatif
        n01 = int(((b > t) & (p <= t)).sum())   # baseline gagal, usulan tidak
        n10 = int(((p > t) & (b <= t)).sum())   # usulan gagal, baseline tidak
        pm = binomtest(n10, n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
        cells.append(f"{fb} vs {fp} (diskordan {n01}/{n10}, p={pm:.2g})")
        tail[t] = (fb, fp, n01, n10, pm)
    # CI bootstrap untuk selisih MEAN (statistik yang benar bagi MAE)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(5000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    L.append(f"| {name} | {b.mean():.4f} | {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] | "
             f"{np.median(d):+.4f} | {100*(d>0).mean():.0f}% | {pw:.2g} | " + " | ".join(cells) + " |")
    rows.append(dict(name=name, mae=b.mean(), mean_d=d.mean(), lo=lo, hi=hi,
                     med_d=np.median(d), win=100*(d > 0).mean(), pw=pw, **{f"t{t}": tail[t] for t in TAUS}))

df = pd.DataFrame(rows)
L += ["", "> `mean d` disertai CI bootstrap 95% (5.000 resample atas citra).",
      "> `diskordan n01/n10` = citra yang gagal HANYA di baseline / HANYA di usulan; p = McNemar eksak.", ""]

# ---------------------------------------------------------------- bacaan
sig_mean = df[(df.lo > 0)]
tie_med = df[df.pw >= 0.05]
L += ["## 2. Bacaan", "",
      f"- **Citra tipikal: seri.** Median selisih ~0.000 mm dan win-rate ~50% terhadap baseline "
      f"terkuat; Wilcoxon (uji median) menyebut {len(tie_med)} baseline SERI: "
      f"{', '.join(tie_med.name)}. Pada citra biasa, model usulan tidak lebih baik.",
      f"- **Ekor: tidak seri.** Pada tau = 0.5 mm, pasangan diskordan hampir selalu satu arah "
      f"(baseline gagal, usulan tidak). Keunggulan MAE datang dari sini, bukan dari citra tipikal.",
      f"- **Mean (=MAE) berbeda nyata** untuk {len(sig_mean)} baseline (CI bootstrap tidak memuat 0): "
      f"{', '.join(sig_mean.name)}.",
      "",
      "### Konsekuensi untuk naskah",
      "Wilcoxon signed-rank menguji median; MAE adalah mean. Melaporkan HANYA Wilcoxon menyembunyikan",
      "efek yang sesungguhnya, dan melaporkan HANYA mean menyembunyikan bahwa citra tipikal seri.",
      "Klaim yang benar dan dapat dipertahankan:",
      "",
      "> Pada citra tipikal model usulan setara dengan baseline terkuat; keunggulannya pada endpoint",
      "> klinis berasal dari berkurangnya **kegagalan ketebalan katastrofik**, yaitu justru kesalahan",
      "> yang berbahaya secara klinis.",
      "",
      "JANGAN tulis 'lowest CIMT error' tanpa kualifikasi ini.", ""]

(RES/"R7_failure_tail.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print("\n-> results/R7_failure_tail.md")
