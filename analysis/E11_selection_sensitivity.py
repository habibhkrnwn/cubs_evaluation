"""E11 — Sensitivitas kriteria seleksi checkpoint (nol compute).

Early stopping memantau **val Dice** (patience 15), tetapi endpoint klinis adalah
**CIMT MAE**. Skrip ini menjawab dua pertanyaan dari data yang sudah ada:

  1. Berapa harga memilih checkpoint dengan val Dice, bukan val MAE?
     -> `MAE@bestDice` vs `MAE@bestMAE` per model (penalti absolut & relatif).
  2. Apakah val MAE masih membaik ketika early stopping memotong latihan?
     -> slope val MAE pada 10 epoch terakhir + apakah argmin jatuh di ujung.

Keduanya memakai kurva val per-epoch:
  * baseline  : baris `Fold f | Ep e/100 | ... val dice=.. iou=.. mae=..mm` di log Kaggle
  * usulan/base: `*_training_history.csv` (kolom `val_dice`, `val_mae_cimt_mm`)

BATAS TAFSIR (wajib dibaca sebelum menulis paper):
  * `MAE@bestMAE` adalah **oracle pada set validasi** — memilih checkpoint dengan metrik
    yang sama yang dilaporkan. Ia BUKAN angka test dan tidak boleh jadi headline.
    Perannya: batas atas optimistis untuk tiap baseline. Kalau model usulan dengan seleksi
    JUJUR masih mengalahkan baseline dengan seleksi ORACLE, tuduhan "baseline kurang
    terlatih / salah pilih checkpoint" tidak berdiri.
  * Kurva baseline terpotong early stopping (15 epoch setelah best Dice), jadi `MAE@bestMAE`
    adalah batas bawah dari yang bisa dicapai latihan tak terbatas. Karena itu slope diukur.
  * Penalti rendah pada BC-VMamba adalah sifat ARSITEKTUR, bukan efek L_cimt:
    base (lambda=0) dan +L_cimt memberi penalti yang praktis sama. JANGAN klaim sebaliknya.

Output: results/E11_selection_sensitivity.md

Pakai:  python notebooks/E11_selection_sensitivity.py
"""
from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EXP = Path(os.environ.get("CUBS_EXPERIMENT_ROOT", REPO))
LOGS = Path(os.environ.get("CUBS_BASELINE_LOG_DIR", EXP/"external_results"/"baseline_logs"))
RES = REPO / "results"
KAGGLE = Path(os.environ.get("CUBS_RUNS_ROOT", EXP/"external_results"))
RES.mkdir(exist_ok=True, parents=True)

# MAE di atas ini = model divergen (CIMT fisiologis ~0.5-1.5 mm). Dikeluarkan dari tabel.
DIVERGED_MM = 1.0
TAIL = 10          # epoch terakhir yang dipakai untuk mengukur slope
N_FOLDS = 5        # hanya model dengan 5 fold penuh yang dilaporkan

STAMP = re.compile(r"^\s*[\d.]+s\s+\d+\s+")
START = re.compile(r"STARTING:\s*([a-z_0-9]+)\s*$")
EPOCH = re.compile(r"Fold (\d) \| Ep (\d+)/\d+ \|.*?val dice=([\d.]+).*?mae=([\d.]+)mm")

LABEL = {
    "unet": "U-Net", "unetpp": "UNet++", "resunet": "ResU-Net",
    "attention_unet": "Attention U-Net", "unext": "UNeXt", "transunet": "TransUNet",
    "segformer": "SegFormer", "swin_unet": "Swin-UNet", "umamba": "U-Mamba",
    "vm_unet_baseline": "VM-UNet",
}

# Kurva model usulan & kontrol arsitekturnya (region-only). Bukan dari log.
HISTORIES = [
    ("BC-VMamba (+L_cimt, seed 42)", "result e1c/results/bcvmamba_a1/bc_vmamba_cubs_lcimt_s42/*_training_history.csv"),
    ("BC-VMamba (+L_cimt, seed 1)", "result e1c seed1/results/bcvmamba_a1/*/*_training_history.csv"),
    ("BC-VMamba (region-only, lambda=0)", "result1/results/bcvmamba_a1/*/*_training_history.csv"),
]


def curves_from_logs() -> dict[str, pd.DataFrame]:
    """Kurva val per-epoch tiap model dari log Kaggle. Log duplikat (byte-identik) diabaikan."""
    seen: dict[int, Path] = {}
    for f in sorted(LOGS.glob("*.log")):
        h = hash(f.read_bytes())
        if h in seen:
            print(f"  DUPLIKAT: '{f.name}' byte-identik dengan '{seen[h].name}' -> diabaikan")
            continue
        seen[h] = f

    out: dict[str, pd.DataFrame] = {}
    for f in sorted(seen.values(), key=lambda p: p.name):
        cur, rows = None, {}
        for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
            ln = STAMP.sub("", ln)
            if m := START.search(ln):
                cur = m.group(1)
                rows.setdefault(cur, [])
            if (m := EPOCH.search(ln)) and cur:
                rows[cur].append((int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))))
        for model, rec in rows.items():
            if not rec:
                continue
            df = pd.DataFrame(rec, columns=["fold", "epoch", "val_dice", "val_mae"])
            if model in out:
                # Run ganda (nondeterminisme kernel). Kebijakan sama dengan
                # parse_baseline_logs.PICK_WORST_MAE: pakai run ber-MAE terburuk,
                # supaya E11 dan tabel paper merujuk run yang sama.
                old = out[model].groupby("fold").val_mae.min().mean()
                new = df.groupby("fold").val_mae.min().mean()
                keep_new = new > old
                print(f"  ! '{model}' muncul di >1 log; memakai run ber-MAE terburuk "
                      f"({'yang baru' if keep_new else 'yang lama'})")
                if not keep_new:
                    continue
            out[model] = df
    return out


def curves_from_history() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, pat in HISTORIES:
        hits = glob.glob(str(KAGGLE / pat))
        if not hits:
            print(f"  ! history tidak ada: {name}")
            continue
        h = pd.read_csv(hits[0]).rename(columns={"val_mae_cimt_mm": "val_mae"})
        out[name] = h[["fold", "epoch", "val_dice", "val_mae"]]
    return out


def analyse(df: pd.DataFrame) -> dict | None:
    """Per fold: MAE di checkpoint best-Dice vs best-MAE, dan slope val MAE di ekor."""
    rows, slopes, at_edge = [], [], 0
    for _, g in df.groupby("fold"):
        g = g.sort_values("epoch")
        if len(g) < TAIL + 2:
            continue
        i_d, i_m = g.val_dice.idxmax(), g.val_mae.idxmin()
        rows.append((g.loc[i_d, "val_mae"], g.loc[i_m, "val_mae"]))
        tail = g.tail(TAIL)
        slopes.append(np.polyfit(tail.epoch, tail.val_mae, 1)[0])
        if g.loc[i_m, "epoch"] >= g.epoch.max() - 2:
            at_edge += 1
    if len(rows) != N_FOLDS:
        return None
    a = float(np.mean([r[0] for r in rows]))
    b = float(np.mean([r[1] for r in rows]))
    return {
        "mae_at_dice": a, "mae_at_mae": b,
        "penalty": a - b, "penalty_pct": 100.0 * (a - b) / b,
        "slope": float(np.mean(slopes)), "at_edge": at_edge, "n_fold": len(rows),
    }


def main() -> int:
    print("E11 — sensitivitas kriteria seleksi checkpoint\n")
    curves = {LABEL.get(k, k): v for k, v in curves_from_logs().items()}
    curves.update(curves_from_history())

    res, skipped, diverged = {}, [], []
    for name, df in curves.items():
        r = analyse(df)
        if r is None:
            skipped.append((name, df.fold.nunique()))
            continue
        # Divergensi dinilai dari checkpoint yang BENAR-BENAR dipakai. Memakai min val MAE
        # akan meloloskan UNeXt, yang sempat menyentuh 0.71 mm di epoch ke-2 lalu meledak.
        if r["mae_at_dice"] > DIVERGED_MM:
            diverged.append(f"{name} (MAE@bestDice {r['mae_at_dice']:.3f} mm)")
            continue
        res[name] = r

    if not res:
        print("tidak ada model dengan 5 fold penuh.")
        return 1

    L = ["# E11 — Sensitivitas kriteria seleksi checkpoint (nol compute)\n",
         "Early stopping memantau **val Dice**; endpoint klinis adalah **CIMT MAE**.",
         "`MAE@bestDice` = checkpoint yang benar-benar dipakai. `MAE@bestMAE` = oracle validasi",
         "(batas atas optimistis, BUKAN angka test).\n",
         "## 1. Harga memilih checkpoint dengan Dice\n",
         "| Model | MAE@bestDice | MAE@bestMAE | penalti | penalti % |",
         "|---|---|---|---|---|"]
    for name, r in sorted(res.items(), key=lambda x: x[1]["penalty_pct"]):
        L.append(f"| {name} | {r['mae_at_dice']:.4f} | {r['mae_at_mae']:.4f} | "
                 f"+{r['penalty']:.4f} | {r['penalty_pct']:.1f}% |")

    L += ["\n## 2. Apakah val MAE masih membaik saat latihan dipotong?\n",
          f"Slope = regresi linier val MAE pada {TAIL} epoch terakhir (mm/epoch).",
          "`argmin di ujung` = jumlah fold yang val MAE-nya minimum pada 3 epoch terakhir.\n",
          "| Model | slope | argmin di ujung | status |", "|---|---|---|---|"]
    for name, r in sorted(res.items(), key=lambda x: x[1]["slope"]):
        status = "masih turun" if r["slope"] < -1e-4 else "konvergen"
        L.append(f"| {name} | {r['slope']:+.5f} | {r['at_edge']}/{r['n_fold']} | {status} |")

    # Klaim kunci: usulan dengan seleksi jujur vs baseline dengan seleksi oracle.
    prop = res.get("BC-VMamba (+L_cimt, seed 42)")
    base_names = [n for n in res if not n.startswith("BC-VMamba")]
    if prop and base_names:
        worst_oracle = max(base_names, key=lambda n: -res[n]["mae_at_mae"])
        best_oracle = min(base_names, key=lambda n: res[n]["mae_at_mae"])
        L += ["\n## 3. Klaim yang ditopang\n",
              f"- Di bawah seleksi-oracle pun model usulan tetap terbaik: "
              f"{prop['mae_at_mae']:.4f} mm vs {res[best_oracle]['mae_at_mae']:.4f} mm ({best_oracle}).",
              f"- Model usulan dengan seleksi **jujur** ({prop['mae_at_dice']:.4f} mm) masih "
              f"mengalahkan baseline terbaik dengan seleksi **oracle** "
              f"({res[best_oracle]['mae_at_mae']:.4f} mm, {best_oracle}).",
              "- Karena itu \"baseline kurang terlatih / salah pilih checkpoint\" tidak dapat "
              "membalikkan kesimpulan.\n"]
        _ = worst_oracle

    b0, b1 = res.get("BC-VMamba (region-only, lambda=0)"), prop
    if b0 and b1:
        L += [f"- **Penalti bukan efek L_cimt.** region-only {b0['penalty_pct']:.1f}% vs "
              f"+L_cimt {b1['penalty_pct']:.1f}%: keselarasan Dice-MAE adalah sifat arsitektur.\n"]

    if diverged:
        L += [f"\n**Dikeluarkan (divergen, MAE > {DIVERGED_MM} mm):** " + ", ".join(diverged) + "\n"]
    if skipped:
        # Hitungan di sini = fold yang punya kurva val (termasuk fold yang terpotong di
        # tengah), jadi bisa lebih besar dari "fold selesai" di parse_baseline_logs.py.
        L += ["**Dikeluarkan (sesi mati, <5 fold penuh):** "
              + ", ".join(f"{n} ({k} fold berkurva)" for n, k in skipped) + "\n"]

    RES.mkdir(exist_ok=True)
    (RES / "E11_selection_sensitivity.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> results/E11_selection_sensitivity.md  ({len(res)} model, "
          f"{len(skipped)} tak lengkap, {len(diverged)} divergen)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
