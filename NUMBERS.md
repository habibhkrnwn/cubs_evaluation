# NUMBERS.md — sumber kebenaran tunggal untuk paper_cbm

Semua angka di manuskrip HARUS berasal dari file ini. Semuanya dari **pipeline E0**
(`Experiment baru 2/`). Angka dari `Experiment baru 1/`, exp28, exp29 **tidak boleh dipakai**.

**Semua analisis (E8/E9/E1b/E5/E10) dihitung ulang pada MODEL USULAN**
(`bc_vmamba_cubs_lcimt_s42`, λ=0.2), bukan pada E1a base. Output ber-sufiks `_lcimt`
di `Experiment baru 2/results/`. Cara mereproduksi:

```bash
cd "Experiment baru 2/notebooks"
export CUBS_RUN="hasil run kaggle/result e1c/results/bcvmamba_a1/bc_vmamba_cubs_lcimt_s42"
export CUBS_TAG="_lcimt"
python E8_stratification.py && python E9_observer_variability.py \
  && python E1b_E5_ensemble_uq.py && python E10_significance.py
```

Terakhir diperbarui: 2026-07-09.

---

## 0. Protokol (berlaku untuk semua angka di bawah)

- Dataset: CUBS gabungan, 2.676 citra (klinis 2.176 + teknikal 500).
- Ground truth: anotator **A1** (gold standard), dirasterisasi dari profil LI/MA.
- Split: patient-disjoint, seed 42. **Held-out test = 401 citra.** Sisanya
  (2.275 citra) untuk 5-fold cross-validation.
- Input 256×256, ROI crop dinding arteri, normalisasi statistik region non-nol.
- Optimizer AdamW, lr 1e-4, wd 1e-4, batch 8, maks 100 epoch, warmup 5 epoch,
  CosineAnnealing (eta_min 1e-6), early stopping patience 15.
- **Loss dasar = 0.5·Dice + 0.5·CE(bobot kelas [1, 3]).**
  TIDAK ADA boundary loss, TIDAK ADA speckle loss, TIDAK ADA IMTGradeHead.
  (Boundary/speckle hanya ada di `Experiment baru 1` — pipeline lain, jangan dikutip.)
- L_cimt merata-rata **semua kolom** (termasuk kolom tanpa GT), dinormalisasi tinggi H.
- CIMT: per kolom, `(row_last − row_first) × scale × CF`, `scale = roi_square_side / 256`.
- Uji statistik: Wilcoxon signed-rank berpasangan; bootstrap 5.000 resample untuk 95% CI.

## 1. Model usulan — BC-VMamba + L_cimt (λ = 0.2), seed 42

Sumber: `hasil run kaggle/result e1c/.../bc_vmamba_cubs_lcimt_s42/bc_vmamba_cubs_a1_fold_metrics.csv`

| Metrik | Nilai (5-fold, mean ± std) |
|---|---|
| Dice | 0.8491 ± 0.0021 |
| IoU | 0.7385 |
| CIMT MAE | **0.1199 ± 0.0046 mm** |
| Parameter | 9,629 M |
| GFLOPs | 22,318 |
| Latensi | ~12 ms/citra (T4) |

Per-fold CIMT MAE: 0.1227 / 0.1219 / 0.1166 / 0.1126 / 0.1256 mm.

Val pooled (n = 2,275, tiap citra diprediksi fold hold-out-nya): Dice **0.8452**, MAE **0.1199** mm.
Test hold-out (n=401, rata-rata 5 fold): Dice 0.8454, MAE 0.1200 mm → tidak overfit.

## 2. Ablasi loss (arsitektur identik; hanya λ yang berubah)

| Varian | Seed | Dice (5-fold) | CIMT MAE (mm) |
|---|---|---|---|
| λ = 0 (base) | 42 | 0.8485 ± 0.0013 | 0.1240 ± 0.0031 |
| λ = 0.2 | 42 | 0.8491 ± 0.0021 | 0.1199 ± 0.0046 |
| λ = 0.2 | 1 | 0.8476 ± 0.0019 | 0.1265 ± 0.0061 |

Sumber base: `hasil run kaggle/result1/.../bc_vmamba_cubs_a1/`
Sumber seed 1: `hasil run kaggle/result e1c seed1/.../bc_vmamba_cubs_lcimt/`

### Uji berpasangan seed 42 (test hold-out, n = 401, rata-rata 5 fold)

- CIMT MAE: 0.1241 → 0.1200 mm; selisih **0.0040 mm [95% CI 0.0014, 0.0065]**;
  Wilcoxon **p = 1.22 × 10⁻⁸**; 62,6% citra membaik.
- Dice: 0.8442 → 0.8454; selisih +0.0011; Wilcoxon p = 2,0 × 10⁻³. Bukan trade-off.

### ATURAN PENULISAN L_cimt (wajib)

Varians antar-seed (~0,007 mm) **sebanding** dengan efek L_cimt (~0,004 mm).
Kalimat yang boleh: *"L_cimt memperbaiki CIMT MAE secara konsisten dalam perbandingan
berpasangan pada seed yang sama, tanpa menurunkan Dice; namun besar efeknya berada dalam
rentang varians antar-seed, sehingga kami melaporkan kedua seed."*
Kalimat yang DILARANG: "L_cimt menurunkan MAE" tanpa kualifikasi; "konsisten di semua seed".

## 3. E8 — stratifikasi robustness (val pooled, n = 2.275, model usulan)

Overall: Dice 0.8452 · CIMT MAE 0.1199 mm. Sumber: `results/E8_stratification_lcimt.md`

| Pusat | n | Dice | MAE (mm) |
|---|---|---|---|
| Munich | 85 | 0.8266 ± 0.0644 | 0.1375 ± 0.1317 |
| Pisa_tech | 85 | 0.8399 ± 0.0675 | 0.1048 ± 0.0923 |
| Porto | 85 | 0.8420 ± 0.0663 | 0.1181 ± 0.0852 |
| Cyprus | 1180 | 0.8434 ± 0.0637 | 0.1177 ± 0.1077 |
| Torino | 85 | 0.8440 ± 0.0547 | 0.1291 ± 0.1260 |
| Pisa_clin | 670 | 0.8460 ± 0.0572 | 0.1257 ± 0.0987 |
| Toronto | 85 | 0.8912 ± 0.0361 | 0.0954 ± 0.0645 |

Gap Dice antar-pusat = **0,0646** (Munich terendah, Toronto tertinggi).

| SNR | n | Dice | MAE (mm) |
|---|---|---|---|
| −1 (buruk) | 40 | 0.8013 ± 0.0862 | 0.1432 ± 0.1447 |
| 0 (sedang) | 248 | 0.8391 ± 0.0601 | 0.1245 ± 0.1090 |
| +1 (baik) | 52 | 0.8616 ± 0.0448 | 0.0960 ± 0.0838 |

| Morfologi | n | Dice | MAE (mm) |
|---|---|---|---|
| 1 | 102 | 0.8322 ± 0.0696 | 0.1335 ± 0.1052 |
| 2 | 90 | 0.8528 ± 0.0604 | 0.1272 ± 0.1352 |
| 3 | 170 | 0.8537 ± 0.0609 | 0.1036 ± 0.0873 |
| 4 | 26 | 0.8610 ± 0.0538 | 0.1142 ± 0.0887 |
| 5 | 37 | 0.8530 ± 0.0570 | 0.1101 ± 0.0857 |

Catatan: semua pusat ikut training → **in-distribution**. Bukan bukti generalisasi.
Metadata SNR hanya ada untuk empat situs teknikal (Munich/Pisa_tech/Porto/Torino);
Toronto tidak punya SNR, dan pusat klinis tidak punya sama sekali.

## 4. R1 — validitas klinis vs variabilitas observer (CIMT, mm) — **DIREVISI 2026-07-12**

Sumber: `results/R1_observer_matched.md` (menggantikan `E9_summary_lcimt.md`)

**Mengapa direvisi.** E9 lama membandingkan CIMT model (diukur dari **mask biner**) dengan CIMT
anotator (diukur dari **profil LI/MA native**). Methods §3.5 mengklaim keduanya identik — tidak.
Bias "unbiased" −0.0013 mm adalah pembatalan bias model (+0.055) oleh bias rasterisasi
referensi (−0.057). R1 merasterisasi A1′/A2/A3 dengan pipeline yang **persis sama** seperti GT
pelatihan A1, lalu mengukur semuanya dengan prosedur mask.

Sanity check: rasterisasi ulang A1 mereproduksi `cimt_gt_mm` dari CSV run **secara eksak**
(max |selisih| = 0.000000 mm, r = 1.000000). Prosedur terbukti identik dengan pelatihan.

### ANGKA UTAMA — test set hold-out (n = 401, data BERSIH, bukan data seleksi checkpoint)

| Perbandingan | n | MAE | Bias | LoA (±1.96σ) | r |
|---|---|---|---|---|---|
| **Model − A1** | 401 | **0.1200** | **+0.0639** | 0.2708 | 0.740 |
| Intra-observer A1′ − A1 | 401 | **0.1319** | +0.0486 | 0.3232 | 0.684 |
| Inter-observer A2 − A1 | 401 | **0.1686** | +0.0660 | 0.4084 | 0.595 |
| Inter-observer A3 − A1 | 118 | 0.1386 | −0.0585 | 0.3348 | 0.559 |
| Inter-observer A3 − A2 | 118 | 0.2002 | −0.1354 | 0.4258 | 0.557 |

- Rasio model / inter-observer (A2−A1) = **0,71×** (dulu keliru dilaporkan 0,58×).
- Selisih berpasangan (inter − model) = **0.0486 mm [95% CI 0.0336, 0.0633]**, Wilcoxon p < 10⁻¹⁰.
- **Model MELEBIHKAN ketebalan +0.0639 mm secara sistematis.** Ini BUKAN unbiased.

### KONVENSI AGREGASI (wajib — sumber inkonsistensi antar-tabel)

Test set dinilai oleh **kelima** model fold. Ada TIGA cara meringkasnya, dan angkanya beda:

| Cara | MAE | Dipakai di |
|---|---|---|
| (a) rata-ratakan **error** per citra lintas fold | **0.1200** | **Tabel 1 (baris test), Tabel 3, Tabel 5** ← default |
| (b) rata-ratakan **prediksi** dulu, baru error | 0.1181 | tidak dipakai di naskah |
| (c) ensemble rata-rata **probabilitas** (satu mask) | 0.1191 | **hanya Sec. 4.6**, diberi label eksplisit |

Bias (+0.0639), LoA, dan r **identik** di (a) dan (b) — hanya MAE yang bergeser.

**ATURAN:** Tabel 5 WAJIB memakai (a) supaya melaporkan besaran yang SAMA dengan baris
test Tabel 1. Kalau tidak, dua tabel akan melaporkan error model terhadap A1 dengan dua
angka berbeda pada citra yang sama — editor akan melihatnya dalam satu menit.

### Val pool (n = 2.275) — lampiran saja (data yang dipakai seleksi checkpoint)

| Perbandingan | n | MAE | Bias |
|---|---|---|---|
| Model − A1 | 2275 | 0.1199 | +0.0554 |
| A1′ − A1 (intra) | 2275 | 0.1452 | +0.0595 |
| A2 − A1 (inter) | 2274 | 0.1776 | +0.0547 |

### Besar artefak prosedur (dihitung, tanpa model sama sekali)

| A1-raster − A1-native | n | MAE | Bias | r |
|---|---|---|---|---|
| test 401 | 401 | 0.0567 | −0.0560 | 0.989 |
| val 2275 | 2275 | 0.0575 | −0.0567 | 0.991 |

Merasterisasi delineasi A1 menggeser CIMT-nya −0.056 mm. Itu **48% dari error headline lama
(0.1103 mm)**, murni artefak. Angka lama 0.1103 / −0.0013 **DILARANG dipakai**.

### ATURAN PENULISAN R1 (wajib)

1. Semua angka observer diambil dari **test set 401**, bukan val pool.
2. Kata **"unbiased"** DILARANG. Model over-read +0.064 mm — laporkan sebagai temuan.
3. Bias +0.064 mm adalah **shared offset** yang dihipotesiskan Diskusi §5.3 untuk menjelaskan
   kebutaan uncertainty berbasis varians. Sambungkan keduanya secara eksplisit.
4. Model dilatih pada A1. Klaim yang boleh: *"mereproduksi pembaca gold-standard dalam batas
   repeatability intra-observer-nya"* (0.1200 < 0.1319 ✓). Klaim yang DILARANG: "melampaui ahli
   manusia", "lebih akurat daripada radiolog".
5. Klaim di poin 4 hanya berlaku **in-distribution**. Pada pusat tak-terlihat (bagian 12,
   poin 3) error model MELAMPAUI pita inter-observer di dua dari tiga sumber. Setiap kutipan
   angka 0.1200 WAJIB membawa syarat itu.

## 5. E1b / E5 / R3 — ensemble & UQ (test hold-out, n = 401)

Sumber: `results/E1b_E5_summary_lcimt.md` + `results/R3_assoc_strat_uq.md` §3

**ISTILAH (wajib).** Lima checkpoint cross-validation BUKAN *deep ensemble* (Lakshminarayanan:
inisialisasi acak independen, data sama). Ini *cross-validation ensemble* / bagging: diversitas
digerakkan oleh **data**, bukan inisialisasi. Pakai istilah **"cross-validation ensemble"**.

- Single model (rata-rata 5 fold): CIMT MAE 0.1200 mm.
- CV ensemble (rata-rata probabilitas 5 fold): **0.1191 mm**.
- Gagal-Dice: kuartil Dice terbawah → 100/401.

| Sinyal UQ | AUC gagal-Dice | AUC gagal-CIMT (ambang 0.191) |
|---|---|---|
| fg_entropy | **0.740** | 0.598 [0.523, 0.672] |
| band_disagree | 0.635 | 0.551 [0.471, 0.626] |
| disagreement | 0.600 | 0.515 |
| band_frac | 0.545 | 0.473 |
| entropy | 0.529 | 0.464 |
| cimt_spread_mm | 0.506 | 0.461 [0.389, 0.533] |

### 5a. Sensitivitas ambang gagal-CIMT (BARU — menjawab "mengapa 0.191?")

| Ambang | asal | n gagal / 401 | AUC fg_entropy | AUC cimt_spread |
|---|---|---|---|---|
| 0.1367 | A3−A1 native | 137 (34,2%) | 0.571 [0.510, 0.632] | 0.477 |
| **0.1686** | **A2−A1 mask, test (R1)** | 98 (24,4%) | **0.580** [0.515, 0.641] | 0.488 |
| 0.1906 | A2−A1 native (dipakai versi lama) | 70 (17,5%) | 0.598 [0.521, 0.669] | 0.461 |
| 0.2274 | A3−A2 native | 43 (10,7%) | 0.623 [0.537, 0.705] | 0.523 |

**Kesimpulan stabil di keempat ambang: AUC 0.57–0.62, semuanya lemah.** Pakai ambang 0.1686
(konsisten dengan prosedur R1) sebagai utama, lampirkan tabel ini sebagai analisis sensitivitas.

### 5b. Kontrol untuk label "gagal-Dice" (BARU — menjawab tuduhan tautologi)

| Prediktor | AUC gagal-Dice |
|---|---|
| **fg_entropy (uncertainty)** | **0.745** |
| band_disagree | 0.640 |
| KONTROL: luas mask prediksi (band_frac) | 0.549 |
| KONTROL: entropy prediktif penuh | 0.534 |
| KONTROL: CIMT prediksi (mm) | 0.423 |
| KONTROL: −SNR (n=60 citra teknis) | 0.499 |

fg_entropy **mengalahkan setiap kontrol sepele dengan selisih besar** → klaim "gerbang QC" bertahan.

### 5c. Selective prediction (fg_entropy) — TIDAK MONOTON, wajib disebut

| Coverage | 100% | 90% | 80% | 70% | 60% | 50% |
|---|---|---|---|---|---|---|
| CIMT MAE (mm) | 0.1191 | 0.1125 | **0.1126 ↑** | 0.1066 | **0.1071 ↑** | 0.1060 |

Membuang 10% citra paling tidak pasti berikutnya justru sedikit **memperburuk** MAE (di 80% dan
60%). Fakta ini **memperkuat** argumen negatif: entropi tidak melacak error CIMT secara monoton.
Menghilangkannya terbaca sebagai penghalusan data. **WAJIB dilaporkan.**

## 5d. Panel kualitatif F4 (dua kasus yang membalik peringkat)

Sumber: `figures/scripts/make_f4_qualitative.py`, persentil error CIMT ke-70 dan ke-95
pada test hold-out. Angka-angka ini sebelumnya **tidak pernah tercatat** di NUMBERS.md
karena `check_numbers.py` lama hanya melacak angka 4-desimal (sudah ditambal).

| Panel | Dice | CIMT err (mm) |
|---|---|---|
| persentil 70 | 0.8330 | 0.1550 |
| persentil 95 | **0.8480** (lebih TINGGI) | **0.2630** (1,70× lebih buruk) |

**ATURAN PENULISAN:** rasionya **1,70×**, bukan "dua kali lipat". Frasa *"twice"*,
*"nearly twice"*, *"doubled"* **DILARANG** — ketiganya pernah dipakai untuk satu rasio
yang sama dan tak satu pun akurat.

## 6. Insight pengikat — asosiasi Dice ↔ error CIMT

Sumber: `results/R3_assoc_strat_uq.md` §1

| Model | Data | n | Pearson r [95% CI] | Spearman ρ [95% CI] |
|---|---|---|---|---|
| **λ=0.2 (usulan)** | **val pool** | **2275** | **−0.528** [−0.595, −0.467] | **−0.526** [−0.559, −0.492] |
| λ=0.2 (usulan) | test 401 | 401 | −0.500 [−0.652, −0.410] | −0.567 [−0.638, −0.489] |
| λ=0 (region-only) | val pool | 2275 | −0.572 [−0.618, −0.526] | −0.543 [−0.574, −0.510] |

**Spearman ≈ Pearson** → asosiasi memang selemah yang dilaporkan; ekor berat TIDAK
mengatenuasinya. Klaim "Dice menjelaskan < ⅓ varians endpoint" (r² = 0.28) bertahan.
**Laporkan ρ berdampingan dengan r** untuk mendahului keberatan reviewer.

## 7. Bootstrap CI model (E10, val pooled n = 2.275)

Sumber: `results/E10_significance_lcimt.md`

- Dice 0.8452 [0.8426, 0.8476]
- CIMT MAE 0.1199 [0.1156, 0.1243] mm

## 8. Peta dataset

| Rilis | Pusat | Pasien | Citra | Catatan |
|---|---|---|---|---|
| CUBS 2021 (klinis) | Cyprus (Nicolaides) | 694 | 1388 | 2003–2007, ada follow-up CVD |
| CUBS 2021 (klinis) | Pisa (Ghiadoni) | 394 | 788 | 2011–2014 |
| CUBS 2022 (teknikal) | Munich | — | 100 | banyak SNR rendah |
| CUBS 2022 (teknikal) | Pisa | — | 100 | |
| CUBS 2022 (teknikal) | Porto | — | 100 | banyak SNR tinggi |
| CUBS 2022 (teknikal) | Torino | — | 100 | |
| CUBS 2022 (teknikal) | Toronto | — | 100 | SNR tidak tersedia |

Calibration factor per citra: 0,038–0,267 mm/piksel (notasi titik: 0.038–0.267).

Anotator: A1 (L.G., Torino, >10 th) = gold standard; A1′ = ulangan A1 setelah 1 bulan;
A2 (G.V., >25 th); A3 (M.G. Cyprus / L.G. Pisa, >25 th).

### ATURAN PENULISAN DATASET (wajib)

Tulis **"two clinical centers and five technical acquisition sites"**.
DILARANG menulis "seven centers" / "7 centers" telanjang.

## 9. Angka dari literatur (bukan hasil kita — jangan bandingkan langsung)

Protokolnya berbeda (split, subset, GT), jadi ini hanya konteks Related Work.

| Sumber | Angka | Catatan |
|---|---|---|
| Hassen Mohammed 2023 | Dice 0.801 (CCTrans, 104,6 M param) | subset klinis 2.176 citra |
| Sarmun 2024 | Dice 0.8203, IMT MAE 0.166 mm | subset teknikal |
| Jeong 2025 | Dice 0.8051–0.8216 (10 varian U-Net) | plateau yang dikutip |
| Meiburger 2021 | inter-analyst 0.160 ± 0.140 mm | JANGAN dipakai; kita hitung sendiri (R1) |

## 9b. Literatur loss function (WAJIB disitasi — sebelumnya NOL sitasi loss di refs.bib)

L_cimt = jumlah softmax per kolom vs jumlah mask per kolom = **soft-cardinality / size-constraint
loss**, didekomposisi per-kolom. Ini punya leluhur yang harus diakui:

| Sumber | Relevansi |
|---|---|
| Kervadec et al., MedIA 2019 (*Constrained-CNN losses*) | penalti diferensiabel pada **jumlah softmax** = mekanisme identik, tapi global |
| Kervadec et al., MIDL 2019 (*Boundary loss*) | loss sadar-geometri untuk struktur tak seimbang |
| Shit et al., CVPR 2021 (*clDice*) | loss sadar-topologi untuk **struktur tubular tipis** — kasus terdekat |
| Karimi & Salcudean, TMI 2020 (*Hausdorff loss*) | mengoptimalkan metrik jarak, bukan overlap |

**Novelty yang boleh diklaim:** dekomposisi **per-kolom** + fakta bahwa jumlah kolom itu *adalah*
endpoint klinis (bukan sekadar regularizer geometri). BUKAN "ide mengoptimalkan jumlah softmax".

## 10. Log penyelesaian eksperimen yang semula direncanakan

| Kode | Isi | Status |
|---|---|---|
| E2 | Uji berpasangan (Wilcoxon + BH) vs baseline | **SELESAI** 2026-07-12 → bagian 11c |
| E3 | LOCO: leave-Cyprus, leave-Pisa_clin, leave-all-technical | **SELESAI** 2026-07-12 → bagian 12 |

Tidak ada eksperimen terjadwal yang masih menjadi placeholder di manuskrip final.

**E4 (XAI faithfulness) DICORET** (keputusan 2026-07-10): nol dari 11 paper CBM melakukannya,
sinyal UQ untuk CIMT lemah (AUC 0.598), dan ia butuh GPU — gradient hook harus menembus
backbone Mamba, jadi label `[lokal]` di `04_EKSPERIMEN.md` keliru.

---

## 11. Perbandingan terkontrol E2 (protokol identik, 5-fold, seed 42)

Direkonstruksi dari log Kaggle oleh `Experiment baru 2/notebooks/parse_baseline_logs.py`
ke `Experiment baru 2/hasil run kaggle/baseline comparison/results/baselines/`.
Mean ± std lintas 5 fold. Model usulan = `bc_vmamba_cubs_lcimt_s42`.

**DIPERLUAS 2026-07-14:** ResU-Net, TransUNet, UNet++ selesai 5 fold (dulu parsial/tak ada).
Roster naik dari 5 → **8 baseline sehat** + UNeXt (divergen). Sumber: folder
`baseline comparison/{resunet,transunet,unetpp_5fold}/`.

| Model | Dice | CIMT MAE (mm) | Params (M) | GFLOPs | best_epoch (mean) |
|---|---|---|---|---|---|
| **BC-VMamba (usulan)** | **0.8491 ± 0.0021** | **0.1199 ± 0.0046** | 9.63 | 22.32 | 84.2 |
| Swin-UNet | 0.7982 ± 0.0151 | 0.1332 ± 0.0226 | 34.17 | 18.11 | 84.8 |
| UNet++ | 0.8219 ± 0.0095 | 0.1460 ± 0.0115 | 36.63 | 276.05 | 65.0 |
| TransUNet | 0.8292 ± 0.0035 | 0.1481 ± 0.0075 | 4.02 | 3.43 | 90.6 |
| SegFormer | 0.8267 ± 0.0031 | 0.1513 ± 0.0134 | 13.67 | 6.57 | 84.6 |
| Attention U-Net | 0.8281 ± 0.0071 | 0.1641 ± 0.0310 | 31.57 | 112.63 | 71.0 |
| U-Mamba | 0.8424 ± 0.0040 | 0.1692 ± 0.0283 | 10.41 | 28.86 | 55.8 |
| ResU-Net | 0.8312 ± 0.0137 | 0.1890 ± 0.0552 | 32.44 | 114.71 | 61.6 |
| U-Net | 0.8055 ± 0.0130 | 0.2277 ± 0.0901 | 31.04 | 109.33 | 38.2 |
| UNeXt | 0.5472 ± 0.0175 | 4.7531 ± 0.2985 | 1.43 | 37.14 | 40.4 |

**UNeXt gagal konvergen** (MAE 4,75 mm ≫ rentang CIMT fisiologis). Laporkan sebagai
kegagalan di teks; jangan masukkan ke figure F1 (skrip mengeluarkannya otomatis).

### 11-bis. CACAT PROTOKOL UNet++ — wajib diungkap, jangan dihapus

Run UNet++ menugaskan **fold 4 = ulangan fold 1** dan **fold 5 = ulangan fold 2**: hanya
**3 partisi validasi berbeda** (union val 1.365 citra, bukan 2.275). Citra di fold kanonik
4 dan 5 tidak pernah masuk validasi. Diverifikasi dengan matriks irisan nama-citra.

- **Test set TIDAK terpengaruh**: 401 citra, hash split identik dengan semua run lain,
  disjoint dari seluruh pool training (2.275 + 401 = 2.676). **Semua uji berpasangan sah.**
- Kolom CV terpengaruh kecil: 5-baris 0.8219/0.1460 vs 3-fold-sah 0.8187/0.1477 (Δ 0.0017 mm).
  **Keputusan user 2026-07-14: pakai angka 5-baris** (di tabel atas). Tidak di-re-run.
- **ATURAN:** boleh melaporkan angkanya, TETAPI naskah **DILARANG menyatakan** UNet++ memakai
  5-fold cross-validation yang identik. Wajib ada catatan kaki bahwa penugasan fold-nya
  mengulang dua partisi. Reviewer bisa memintanya; jangan sampai naskah menyatakan protokol
  yang tidak dijalankan.

### 11a. PERANCU: early stopping pada Dice memangkas konvergensi CIMT

Lintas enam arsitektur yang konvergen, **lama latihan menjelaskan MAE hampir sempurna,
sedangkan Dice hampir tidak menjelaskan apa-apa**:

- corr(mean best_epoch, mean MAE) = **−0.932**
- corr(mean best_epoch, mean Dice) = +0.181
- corr(mean Dice, mean MAE) = Pearson −0.375 (p=0,46); Spearman −0.200 (p=0,70), n=6

Early stopping memantau **val Dice** (patience 15). Begitu Dice mendatar, latihan berhenti —
padahal MAE CIMT masih turun. Karena itu model yang Dice-nya cepat jenuh (U-Net berhenti di
epoch 14 dan 16 pada dua fold) ter-checkpoint dengan batas yang belum konvergen: MAE fold-nya
0,3773 dan 0,2112 mm, versus 0,1517 mm pada fold yang berlatih 86 epoch. Pola yang sama
tampak di dalam setiap baseline (korelasi per-model best_epoch↔MAE: −0,43 s.d. −1,00).

**Konsekuensi.** (a) Ini ANCAMAN VALIDITAS: sebagian keunggulan MAE model usulan bisa
dibaca sebagai "ia berlatih lebih lama", bukan "arsitekturnya lebih baik". Reviewer akan
menanyakannya. Wajib diungkap. (b) Sekaligus ini bukti TERKUAT untuk tesis paper: kriteria
seleksi berbasis Dice menghasilkan checkpoint yang buruk untuk endpoint klinis.

**Kontrol internal (tanpa compute tambahan).** Di antara tiga model yang berlatih sama lama
(best_epoch 84,2–84,8), urutan MAE tetap: usulan 0,1199 < Swin-UNet 0,1332 < SegFormer 0,1513.
Gunakan subset ini sebagai perbandingan utama yang terkontrol-panjang-latihan.

**Counterexample yang berdiri sendiri:** Swin-UNet punya Dice TERENDAH di antara model sehat
(0,7982) tetapi MAE TERBAIK KEDUA (0,1332) — mengalahkan U-Mamba yang Dice-nya jauh lebih
tinggi (0,8424, MAE 0,1692). Peringkat Dice dan peringkat CIMT tidak sejalan. Ini argumen
paper yang ditunjukkan lintas-arsitektur, bukan hanya lintas-citra.

### 11b. Sensitivitas kriteria seleksi checkpoint (E11, nol compute) — CADANGAN, TIDAK di naskah

> **Keputusan 2026-07-10:** analisis ini DICABUT dari manuskrip. Perbandingan tabel memakai
> protokol identik (early stopping sama untuk semua model) sehingga sudah adil apa adanya;
> menyorot riwayat training justru memancing pertanyaan "kenapa tidak seleksi checkpoint via
> MAE?". Skrip `E11_selection_sensitivity.py` dan tabel di bawah DISIMPAN sebagai amunisi bila
> reviewer menantang seleksi checkpoint / menuduh baseline kurang terlatih — deploy saat revisi,
> bukan di submission pertama. B2 (retrain U-Mamba tanpa early stopping) juga dibatalkan.

Sumber: `Experiment baru 2/results/E11_selection_sensitivity.md`
(`notebooks/E11_selection_sensitivity.py`). `MAE@bestDice` = checkpoint yang benar-benar
dipakai (early stopping berbasis val Dice). `MAE@bestMAE` = **oracle validasi** — batas atas
optimistis, BUKAN angka test, tak boleh jadi headline.

| Model | MAE@bestDice | MAE@bestMAE (oracle) | penalti | penalti % | slope val MAE (10 ep) |
|---|---|---|---|---|---|
| BC-VMamba region-only (λ=0) | 0.1240 | 0.1197 | +0.0043 | 3.6% | +0.00002 (konvergen) |
| BC-VMamba +L_cimt (s42) | 0.1199 | 0.1155 | +0.0044 | 3.8% | +0.00002 (konvergen) |
| SegFormer | 0.1513 | 0.1407 | +0.0107 | 7.6% | +0.00018 (konvergen) |
| BC-VMamba +L_cimt (s1) | 0.1265 | 0.1173 | +0.0091 | 7.8% | −0.00052 |
| Swin-UNet | 0.1332 | 0.1219 | +0.0113 | 9.3% | +0.00018 (konvergen) |
| Attention U-Net | 0.1641 | 0.1444 | +0.0197 | 13.7% | −0.00217 |
| U-Net | 0.2277 | 0.1898 | +0.0379 | 20.0% | −0.00351 |
| U-Mamba | 0.1692 | 0.1397 | +0.0294 | 21.1% | −0.00330 (argmin di ujung 2/5 fold) |

**Klaim yang ditopang:**
1. Di bawah seleksi-oracle pun model usulan tetap terbaik (0,1155 vs Swin 0,1219).
2. Model usulan dengan seleksi **jujur** (0,1199) masih mengalahkan baseline terbaik dengan
   seleksi **oracle** (0,1219, Swin). Jadi "baseline salah pilih checkpoint / kurang terlatih"
   tidak dapat membalikkan kesimpulan.
3. Batas atas U-Net (0,1898) dan Attention U-Net (0,1444) di bawah seleksi-oracle tetap kalah
   dari 0,1199 → melatih ulang keduanya tak dapat mengubah urutan; itu sebabnya hanya U-Mamba
   yang perlu dilatih ulang tanpa early stopping (B2).

**ATURAN KERAS — penalti rendah = sifat ARSITEKTUR, bukan efek L_cimt.** region-only 3,6% ≈
+L_cimt 3,8%. JANGAN menulis bahwa L_cimt "membuat Dice dan CIMT selaras" atau "menurunkan
penalti seleksi" — datanya menolak klaim itu. L_cimt menurunkan MAE (0,1240→0,1199); ia tidak
mengubah keselarasan Dice↔MAE.

**ATURAN KERAS — jangan klaim keunggulan Dice atas U-Mamba.** Kedua model punya variasi
run-ke-run yang terukur:

| | Dice | CIMT MAE (mm) |
|---|---|---|
| Usulan, seed 42 | 0.8491 | 0.1199 |
| Usulan, seed 1 | 0.8476 | 0.1265 |
| *spread antar-seed* | *0.0015* | *0.0066* |
| U-Mamba, run paper | 0.8424 | 0.1692 |
| U-Mamba, run alt (seed nominal sama) | 0.8477 | 0.1503 |
| *spread run-ke-run* | *0.0053* | *0.0188* |

**Perbandingan kasus-terburuk (seed usulan terlemah vs run U-Mamba terbaik):**
Dice 0,8476 vs 0,8477 → **−0,0001 (seri; U-Mamba sepersepuluh-ribu lebih tinggi)**.
MAE 0,1265 vs 0,1503 → **+0,0239 mm lebih baik (16%)**.

Jadi pada Dice kedua model **tidak dapat dibedakan**, sementara pada endpoint klinis
selisihnya konsisten dan besar. Ini tesis paper dalam satu baris — dan sekaligus
membebaskan paper dari kriteria desk-reject CBM "minor architecture modification +
slight increase in Dice", sebab kita **tidak mengklaim kenaikan Dice sama sekali**.

**Kebijakan run ganda (keputusan 2026-07-10):** paper memakai run U-Mamba ber-MAE
**terburuk** (0,1692). Karena itu melebarkan margin model usulan, keberadaan run kedua
(`umamba__alt1`, MAE 0,1503) **wajib diungkap** di footnote/Limitations.

**Status roster.** Selesai 5 fold: `unet`, `attention_unet`, `swin_unet`, `segformer`,
`umamba` (×2 run), `unext` (divergen). Tidak selesai — sesi Kaggle mati di batas 12 jam:
`resunet` (2/5 fold), `unetpp` (1/5 fold). Belum pernah dilatih: `transunet`,
`vm_unet_baseline`. Teks `04_results.tex` masih menjanjikan **sepuluh** arsitektur —
turunkan ke enam yang benar-benar ada (lima sehat + UNeXt sebagai catatan kegagalan).

### 11c. Uji berpasangan per-citra E10 — **DIGANTIKAN oleh §13c (2026-07-14)**

> ⚠️ Tabel di bawah memakai keluarga BH atas **10 uji (5 baseline)**. Setelah ResU-Net,
> TransUNet, dan UNet++ selesai, keluarga BH menjadi **20 uji (8 baseline + ablasi)** dan
> nilai q berubah. **Naskah WAJIB memakai §13c**, bukan tabel ini. Disimpan sebagai riwayat;
> jangan kutip angkanya ke `.tex`.

Sumber: `Experiment baru 2/results/E10_significance_lcimt.md`. Model usulan
`bc_vmamba_cubs_lcimt_s42` vs **lima baseline sehat**, **test hold-out, n = 401 citra**,
rata-rata 5 fold per citra. Baseline dikurasi di
`Experiment baru 2/results/sig_baselines/` (lihat `PROVENANCE.md`): U-Net **ditambahkan
2026-07-11** (`result(7)`, per-image lengkap 5 fold); U-Mamba memakai **run paper**
(`results(4)`, sesi swin+umamba). `q` = BH atas 10 uji (5 baseline × 2 metrik). Signifikansi
dinilai dari **q**, bukan p.

| Baseline | ΔDice (usulan−base) [95% CI] | q(Dice) | ΔCIMTerr (base−usulan) mm [95% CI] | p(err) | q(err) |
|---|---|---|---|---|---|
| Attention U-Net | +0.0196 [+0.0174, +0.0218] | 2.5e-48 | +0.0559 [+0.0279, +0.0892] | 2e-12 | 2.2e-12 |
| SegFormer | +0.0230 [+0.0199, +0.0263] | 4.3e-41 | +0.0831 [+0.0223, +0.1971] | 4.1e-17 | 6.9e-17 |
| Swin-UNet | +0.0498 [+0.0464, +0.0532] | 7.2e-65 | +0.0472 [+0.0005, +0.1306] | **0.67** | **0.67** |
| U-Mamba | +0.0085 [+0.0066, +0.0104] | 5e-21 | +0.0565 [+0.0265, +0.0969] | 1e-14 | 1.5e-14 |
| U-Net | +0.0433 [+0.0406, +0.0460] | 7.2e-65 | +0.1327 [+0.0854, +0.1876] | 9.8e-14 | 1.2e-13 |

**Bacaan yang WAJIB jujur:**

1. **CIMT — inti paper.** Model usulan CIMT-error-nya **signifikan lebih rendah** dari tiga
   baseline (Attention U-Net, SegFormer, U-Mamba; semua q ≪ 0,001). **Terhadap Swin-UNet
   TIDAK signifikan** (q = 0,67): mean tetap 0,047 mm lebih baik, tetapi distribusi error
   CIMT berat-ekor sehingga uji berperingkat tak melihat urutan per-citra yang konsisten.
   (U-Net terjauh: ΔCIMT +0,1327 mm, q = 1,2e-13.)
   → **JANGAN tulis "clinical accuracy no baseline matches" tanpa kualifikasi.** Klaim yang
   benar: usulan mengungguli **empat dari lima** baseline pada CIMT; Swin-UNet setara pada CIMT
   **tetapi ditukar dengan Dice** (lihat butir 2). Swin adalah baseline CIMT terkuat.

2. **Dice — tetap TIDAK boleh jadi klaim keunggulan.** Uji per-citra memang menunjukkan
   ΔDice > 0 signifikan atas SEMUA baseline termasuk run-paper U-Mamba (+0,0085, q = 5e-21).
   Tetapi CI per-citra ini **tidak menangkap variansi antar-run**: spread Dice run-ke-run
   U-Mamba (0,0053) sebanding dengan selisih +0,0085. Aturan §11 "jangan klaim keunggulan
   Dice atas U-Mamba" **tetap berlaku** — uji ini melawan satu run U-Mamba saja, bukan
   distribusinya. Kolom Dice boleh muncul di tabel (data berpasangan; menyembunyikannya =
   cherry-picking), tetapi prosa tetap memakai bingkai seri-dalam-derau.

3. **Sinergi Dice×CIMT lawan Swin.** Kontras terkuat paper: Swin-UNet kalah telak pada Dice
   (+0,0498, q = 1e-64) namun **setara** pada CIMT — persis tesis "Dice ≠ endpoint" dibaca
   dari arah sebaliknya. Ini kalimat kunci menggantikan "no baseline matches".

---

## 12. E3' — Generalisasi lintas-pusat (LOCO), 2026-07-12

Model usulan (BC-VMamba + L_cimt, protokol identik bagian 11) **dilatih ulang dari nol**
tanpa pusat yang di-hold-out, lalu diuji pada SELURUH citra pusat itu.
Sumber: `Experiment baru 2/results/E3_loco.md` (log Kaggle `e3-loco-ready`).

Ukuran latih (log): leave-Cyprus train=1128/val=160/test=1388 · leave-Pisa_clin
train=1659/val=229/test=788 · leave-AllTech train=1914/val=262/test=500.

**Referensi in-distribution = pusat yang SAMA saat ia ikut dilatih** (prediksi val pooled
5-fold, bagian 8/E8), bukan rata-rata global. Ini pembanding yang jujur: selisihnya murni
efek "pusat tak terlihat", bukan efek komposisi pusat.

| Held-out | test n | Dice LOCO | Dice in-dist | ΔDice | MAE LOCO (mm) | MAE in-dist (mm) | ΔMAE (mm) | ΔMAE rel |
|---|---|---|---|---|---|---|---|---|
| leave-Cyprus        | 1388 | 0.8160 | 0.8434 | −0.0274 | 0.1827 | 0.1177 | +0.0650 | +55% |
| leave-Pisa-clinical |  788 | 0.8374 | 0.8460 | −0.0086 | 0.1619 | 0.1257 | +0.0362 | +29% |
| leave-all-technical |  500 | 0.7993 | 0.8487 | −0.0494 | 0.1933 | 0.1170 | +0.0763 | +65% |

### 12a. R4 — analisis per-citra (2026-07-12). **INI YANG DIPAKAI DI NASKAH.**

Sumber: `Experiment baru 2/results/R4_loco_perimage.md`, dari CSV per-citra LOCO yang
akhirnya diunduh (`hasil e3 loco/results/bcvmamba_a1/loco_bcvmamba/`). Tabel di atas
dibangun dari **log**, tanpa satu pun interval. Yang di bawah punya CI dan kohort setara.

**Kohort setara.** LOCO diuji pada seluruh citra pusat; referensi in-dist hanya punya
prediksi out-of-fold untuk porsi val pool. Di bawah, keduanya diskor pada citra **identik**.
Model LOCO = **satu** model, jadi acuannya prediksi out-of-fold (juga satu model/citra) —
bukan rata-rata 5 fold.

| Held-out | n cocok | Dice LOCO | ΔDice [95% CI] | CIMT LOCO | ΔCIMT (mm) [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|---|
| Cyprus | 1180 | **0.8142** | −0.0292 [−0.0322, −0.0266] | **0.1791** | **+0.0614** [+0.0513, +0.0729] | 4.8e-99 |
| Pisa_clin | 670 | **0.8339** | −0.0122 [−0.0159, −0.0090] | **0.1595** | **+0.0339** [+0.0141, +0.0579] | 1.2e-05 |
| AllTech | 425 | **0.7963** | −0.0525 [−0.0648, −0.0411] | **0.1962** | **+0.0792** [+0.0452, +0.1218] | 1.2e-07 |

**Pita observer pada KOHORT YANG SAMA** (prosedur mask, citra identik — wajib dipakai
untuk perbandingan LOCO vs manusia; angka global 0.1686/0.1319 hanya untuk test set 401):

| Sumber | n | error LOCO | inter (A2−A1) | intra (A1′−A1) | putusan |
|---|---|---|---|---|---|
| Cyprus | 1180 | 0.1791 | **0.1701** | 0.1364 | model **LEBIH BURUK** dari dua ahli |
| Pisa_clin | 670 | 0.1595 | **0.1947** | 0.1537 | model di bawah pita ahli |
| AllTech | 425 | 0.1962 | **0.1716** | 0.1562 | model **LEBIH BURUK** dari dua ahli |

### 12b. Breakdown per-situs arm AllTech — **KLAIM MEKANISME LAMA GUGUR**

| Situs | n | ΔDice | ΔCIMT (mm) [95% CI] |
|---|---|---|---|
| Munich | 85 | −0.0169 | +0.0213 [−0.0047, +0.0475] |
| Pisa_tech | 85 | −0.0053 | +0.0575 [+0.0037, +0.1568] |
| Porto | 85 | −0.0054 | +0.0100 [+0.0028, +0.0177] |
| Torino | 85 | −0.0114 | +0.0043 [−0.0142, +0.0217] |
| **Toronto** | 85 | **−0.2234** | **+0.3030** [+0.1534, +0.4752] |

**Uji leave-one-site-out pada arm AllTech (ΔCIMT):**

| Arm | ΔCIMT | urutan vs Cyprus (+0.0614) & Pisa_clin (+0.0339) |
|---|---|---|
| lengkap (5 situs) | **+0.0792** | terburuk dari ketiganya |
| tanpa Torino | +0.0980 | terburuk |
| tanpa Porto | +0.0965 | terburuk |
| tanpa Munich | +0.0937 | terburuk |
| tanpa Pisa_tech | +0.0847 | terburuk |
| **tanpa Toronto** | **+0.0233** | **TERINGAN dari ketiganya → urutan TERBALIK** |

Buang situs mana pun kecuali Toronto → AllTech tetap terburuk. Buang Toronto → ia jadi
**paling ringan**. Klaim mekanisme bersandar pada **satu scanner**.

### 12c. Sifat kegagalan Toronto: tingkat kegagalan, bukan pergeseran massal

| Besaran | Toronto (LOCO) | 4 situs lain (LOCO) |
|---|---|---|
| Dice rata-rata | 0.6678 | 0.8284 |
| error CIMT **rata-rata** | 0.3984 | 0.1457 |
| error CIMT **median** | **0.1629** | 0.1107 |
| citra dengan Dice < 0.5 | **17/85** | 1/340 |

Sanity check (bukan artefak): GT identik (cimt_gt 1.0005 LOCO vs 1.0053 in-dist), CF
valid (0.039–0.053). Kegagalan ada di **prediksi**. Toronto punya Dice in-distribution
**tertinggi dari ketujuh sumber** (0.8912) — yang termudah saat dilatih, yang paling runtuh
saat tak dilihat.

### 12c-bis. Kebocoran statistik normalisasi (R6) — terkuantifikasi, tidak mengubah kesimpulan

Sumber: `results/R6_norm_leak.md`. `DATASET_MEAN`/`DATASET_STD` dihitung dari SELURUH
2.275 citra CV — termasuk pusat yang di-hold-out. Itu kebocoran.

**Mengapa pergeseran μ tidak relevan:** transformasi afin global (x−μ)/σ diterapkan
IDENTIK ke latih dan uji, jadi jarak antar-domain = (μ_A − μ_B)/σ — **μ hilang saat
pengurangan** dan diserap bias konvolusi pertama. Hanya **σ** yang berpengaruh, karena
ia menskala jarak domain.

| Setting | σ seharusnya | σ dipakai | selisih | arah bias |
|---|---|---|---|---|
| leave-Cyprus | 0.203147 | 0.191656 | **−5.7%** | **konservatif** (gap dilaporkan = batas atas) |
| leave-Pisa_clin | 0.177426 | 0.191656 | **+8.0%** | optimistis (gap mungkin sedikit lebih besar) |
| leave-AllTech | 0.191203 | 0.191656 | **+0.2%** | **netral — dapat diabaikan** |

**ATURAN PENULISAN:** WAJIB dilaporkan di §5.6, terkuantifikasi. Arm AllTech (tempat
temuan Toronto) σ meleset hanya 0,2% → **keruntuhan Toronto TIDAK dapat dijelaskan oleh
kebocoran ini**. DILARANG mendiamkannya: reviewer yang menemukannya sendiri akan menuduh
kita menyembunyikan.

### 12d. Asosiasi Dice↔CIMT di bawah pergeseran — **STABIL** (hipotesis "melemah" GUGUR)

| Kondisi | n | Pearson r [95% CI] | Spearman ρ |
|---|---|---|---|
| LOCO gabungan | 2676 | −0.507 [−0.583, −0.437] | −0.604 |
| In-distribution | 2275 | −0.528 [−0.594, −0.469] | −0.526 |

**DILARANG menulis "asosiasinya melemah di luar distribusi".** CI bertumpang tindih;
Spearman malah bergerak ke arah sebaliknya. Yang boleh: dekopling Dice↔CIMT **direplikasi
out-of-distribution** — ia bukan artefak satu kohort, dan itu memperkuat r = −0.53.

---

### BACAAN YANG WAJIB (aturan penulisan E3) — DIREVISI 2026-07-12

1. **Dice menyembunyikan ongkos transfer.** Dice turun 1,2–5,3%, sementara error klinis naik
   **29–65%**. Semua signifikan dengan CI (12a). Ini tetap argumen LOCO utama paper.
2. **KLAIM MEKANISME LAMA DICABUT.** Kalimat *"acquisition shift, not data volume, is the
   mechanism"* / *"withholding the five technical sites is the costliest setting even though it
   removes the least training data"* **DILARANG**. Bukti: 12b. Klaim itu bersandar sepenuhnya
   pada Toronto; tanpa Toronto urutannya terbalik.

   **Yang menggantikannya (lebih kuat, bukan lebih lemah):**
   > Kegagalan transfer lintas-pusat **bukan pajak yang merata, melainkan undian.** Empat dari
   > lima scanner tak-terlihat nyaris tak berbiaya (ΔCIMT +0.004 .. +0.058 mm); yang kelima
   > runtuh (+0.303 mm, Dice 0.6678). Dan scanner yang runtuh adalah **yang paling mudah ketika
   > ikut dilatih** (Dice in-dist 0.8912, tertinggi dari tujuh sumber). **Performa in-distribution
   > sebuah situs tidak memprediksi apakah model akan bertahan di situs itu bila tak pernah
   > dilihat.**
3. **SAMBUNGKAN KE BAB UQ (12c).** Di dalam distribusi, sisa error = *shared offset* +0.064 mm
   yang tak terlihat oleh varians → gerbang uncertainty tak berguna untuk milimeter. **Di luar
   distribusi mode kegagalannya BERUBAH**: bukan bias halus, melainkan keruntuhan mask
   (17/85 citra Dice<0.5) — dan itu persis yang dideteksi `fg_entropy` (AUC 0.740). Gerbang yang
   tak berguna di dalam distribusi menjadi **justru yang paling dibutuhkan** di luarnya. Ini
   memulihkan rekomendasi klinis paper yang sebelumnya dibantah bab UQ sendiri.
4. **LOCO vs pita inter-observer — DIKOREKSI 2026-07-12.** Versi lama membandingkan error
   LOCO (prosedur mask) dengan inter-observer 0,1906 mm (prosedur *native*) — perbandingan
   tak setara yang sama dengan temuan fatal bagian 4. Dengan prosedur mask **dan** dihitung
   pada **sumber yang sama** (bukan satu angka global):

   | Sumber tak-terlihat | n | error LOCO | inter-observer (A2−A1, mask, sumber itu) | putusan |
   |---|---|---|---|---|
   | Cyprus | 1388 | **0.1827** | 0.1695 | model **LEBIH BURUK** dari ketidaksepakatan ahli |
   | Pisa-clinical | 788 | **0.1619** | 0.1924 | model masih di bawah pita ahli |
   | 5 situs teknis | 500 | **0.1933** | 0.1695 | model **LEBIH BURUK** dari ketidaksepakatan ahli |

   Intra-observer (A1′−A1, mask) di sumber yang sama: Cyprus 0.1355 · Pisa_clin 0.1503 ·
   teknis 0.1536 — **model LOCO di atas repeatability pembaca di ketiganya.**

   **ATURAN PENULISAN (wajib).** Klaim lama *"pada pusat tak terlihat error model masuk ke
   rentang ketidaksepakatan antar-ahli"* **DILARANG** — terlalu murah hati. Yang benar:
   *pada dua dari tiga sumber tak-terlihat, error model MELAMPAUI ketidaksepakatan dua ahli
   pada sumber itu.* Ini memperkuat, bukan melemahkan, kesimpulan bahwa kalibrasi
   spesifik-situs adalah prasyarat, bukan penyempurnaan.
5. Sel plot terakhir notebook (`all_hist_dfs`) gagal SETELAH semua CSV tersimpan — kegagalan
   kosmetik, nol dampak pada angka di atas.

---

## 13. R2 — Uji tesis inti: apakah Dice meranking arsitektur seperti CIMT?

Sumber: `Experiment baru 2/results/R2_baseline_ranking.md`. Angka Dice/MAE diambil dari
`*_fold_metrics.csv` — **sumber yang sama dengan Tabel 2**, jadi ini statistik baru di atas
angka lama, bukan angka baru.

### 13a. Spearman ρ antara peringkat-Dice dan peringkat-CIMT

Konvensi: ρ dihitung antara (−Dice) dan CIMT MAE, keduanya "besar = buruk".
ρ = +1 → kedua metrik meranking arsitektur identik (Dice = proxy sempurna).
ρ = 0 → peringkatnya tak berhubungan (Dice = proxy tak berguna).

**DIPERBARUI 2026-07-14: k naik 6 → 9** arsitektur (+ResU-Net, TransUNet, UNet++).
Angka k=6 lama disimpan sebagai baris riwayat; **yang dipakai naskah adalah k=9.**

| Himpunan arsitektur | k | ρ | p |
|---|---|---|---|
| **9 model, U-Mamba run buruk (dipakai Tabel 2)** | **9** | **+0.017** | **0.966** |
| **9 model, U-Mamba run baik** | **9** | **+0.133** | **0.732** |
| 10 model (+ UNeXt), U-Mamba run buruk | 10 | +0.285 | 0.425 |
| 10 model (+ UNeXt), U-Mamba run baik | 10 | +0.370 | 0.293 |
| *(riwayat)* 6 model, U-Mamba run buruk | 6 | +0.200 | 0.704 |
| *(riwayat)* 6 model, U-Mamba run baik | 6 | +0.371 | 0.468 |

Bootstrap 95% CI atas citra test (n=401, 5.000 resample): **[−0.08, +0.73]**.

Peringkat eksplisit, k=9 (region-only, U-Mamba run baik):

| Model | Dice | rank Dice | CIMT MAE | rank CIMT |
|---|---|---|---|---|
| BC-VMamba (region) | 0.8485 | 1 | 0.1240 | 1 |
| Swin-UNet | 0.7982 | **9** | 0.1332 | **2** |
| UNet++ | 0.8219 | 7 | 0.1460 | 3 |
| TransUNet | 0.8292 | 4 | 0.1481 | 4 |
| U-Mamba | 0.8477 | 2 | 0.1503 | 5 |
| SegFormer | 0.8267 | 6 | 0.1513 | 6 |
| Attention U-Net | 0.8281 | 5 | 0.1641 | 7 |
| ResU-Net | 0.8312 | **3** | 0.1890 | **8** |
| U-Net | 0.8055 | 8 | 0.2277 | 9 |

### ATURAN PENULISAN ρ (wajib) — DIREVISI 2026-07-14

1. **Laporkan ρ pada k=9**, bukan k=6. Dengan tiga arsitektur tambahan, ρ **runtuh ke nol**
   (+0.02 s.d. +0.13): peringkat Dice praktis tidak membawa informasi tentang peringkat CIMT.
2. **ρ tetap TIDAK signifikan** (p = 0.73–0.97). Sebab statistiknya penting dan wajib ditulis
   benar: nilai p besar di sini berarti kita **tidak dapat menolak H0: ρ = 0** — dan ρ = 0
   justru KONSISTEN dengan tesis. Tetapi **tidak menolak H0 bukan berarti membuktikan H0**.
   DILARANG menulis "ρ membuktikan Dice tidak berkorelasi".
   Yang SAH: CI bootstrap **mengesampingkan ρ > 0.73**, jadi kita punya **batas atas** —
   Dice terbukti **bukan proxy peringkat yang kuat**. Itulah klaim yang boleh ditarik.
3. **Laporkan ρ dengan DAN tanpa UNeXt.** Membuang UNeXt menurunkan ρ (+0.37 → +0.13) —
   yakni menguntungkan tesis. Menyembunyikan itu = cherry-picking.
4. Bukti kualitatif terkuat, berdiri sendiri tanpa perlu ρ signifikan:
   **Swin-UNet peringkat 9/9 pada Dice tetapi 2/9 pada CIMT**; **ResU-Net peringkat 3 pada
   Dice tetapi 8 pada CIMT**. Dua inversi berlawanan arah pada roster yang sama.

### 13b. Sebaran run-to-run (U-Mamba, seed nominal sama)

| | run 1 | run 2 (dipakai Tabel 2) | sebaran |
|---|---|---|---|
| Dice | 0.8477 | 0.8424 | 0.0053 |
| CIMT MAE (mm) | 0.1503 | 0.1692 | **0.0188** |

Jarak juara-1 → juara-2 di Tabel 2 (BC-VMamba 0.1240 vs Swin-UNet 0.1332) = **0.0092 mm**
= **0.49×** sebaran run-to-run itu.

**ATURAN PENULISAN:** setiap sel Tabel 2 adalah **satu run**. Peringkat yang lebih halus dari
0.0188 mm tidak dapat dipertahankan. DILARANG menulis *"the lowest CIMT error of the tested
models"*. Nyatakan keterbatasan ini eksplisit di §4.1.

### 13c. Uji berpasangan test set (n=401, Wilcoxon + Benjamini-Hochberg)

Δ positif = model usulan lebih baik.

Δ positif = model usulan lebih baik. **DIPERLUAS 2026-07-14** ke 8 baseline; BH atas 20 uji.

| Pembanding | ΔDice | q(Dice) | ΔCIMT (mm) | q(CIMT) | putusan endpoint |
|---|---|---|---|---|---|
| U-Net | +0.0433 | 1.4e-64 | +0.1327 | 1.8e-13 | lebih baik |
| Attention U-Net | +0.0196 | 3.8e-48 | +0.0559 | 3.3e-12 | lebih baik |
| SegFormer | +0.0230 | 5e-41 | +0.0831 | 9.1e-17 | lebih baik |
| **Swin-UNet** | +0.0498 | 1.4e-64 | +0.0472 | **0.67** | **SERI** |
| **UNet++** | +0.0231 | 8e-55 | +0.0272 | **0.16** | **SERI** |
| **ResU-Net** | +0.0166 | 1.3e-47 | +0.0786 | **0.11** | **SERI** |
| TransUNet | +0.0197 | 7.8e-43 | +0.0453 | 0.00012 | lebih baik |
| U-Mamba (run baik) | +0.0024 | 0.015 | +0.0495 | 2.3e-07 | lebih baik |
| U-Mamba (run buruk) | +0.0085 | 6.3e-21 | +0.0565 | 2e-14 | lebih baik |
| BC-VMamba region-only (ablasi λ) | +0.0011 | 0.0025 | +0.0040 | 1.9e-08 | lebih baik |

### ATURAN PENULISAN (wajib) — standar bukti tunggal, DIREVISI 2026-07-14

**TIGA baseline kini SERI pada endpoint** (Swin-UNet, UNet++, ResU-Net), bukan satu.
Ini WAJIB ditulis apa adanya. Jika 0.0472 mm terhadap Swin ditolak sebagai tidak nyata,
maka 0.0040 mm dari L_cimt **tidak boleh** dirayakan dengan standar yang lebih longgar.

**TETAPI Wilcoxon menguji MEDIAN, sedangkan MAE adalah MEAN.** "Seri" di sini hanya berarti
seri **pada citra tipikal**. Lihat §15 (R7): pada ekor kegagalan ketiganya TIDAK seri.
Melaporkan hanya salah satu dari keduanya = menyesatkan. Wajib dua-duanya.

---

## 14. R3 — Uji stratifikasi (val pooled, n = 2.275)

Sumber: `Experiment baru 2/results/R3_assoc_strat_uq.md` §2. Bootstrap 5.000 resample citra.

| Kontras | n | Δ | 95% CI | p (MWU) | putusan |
|---|---|---|---|---|---|
| SNR rendah − tinggi, **CIMT MAE** | 40/52 | +0.0471 | [+0.0013, +0.1009] | **0.096** | **LEMAH** |
| SNR rendah − tinggi, Dice | 40/52 | −0.0603 | [−0.0892, −0.0317] | <0.001 | kokoh |
| Morfologi 2 − 3, **CIMT MAE** | 90/170 | +0.0236 | [−0.0059, +0.0561] | **0.359** | **NIHIL** |
| Morfologi 2 − 3, Dice | 90/170 | −0.0009 | [−0.0164, +0.0142] | 0.805 | nihil |
| Munich − Toronto, CIMT MAE | 85/85 | +0.0421 | [+0.0127, +0.0752] | 0.058 | sedang |
| Munich − Toronto, **Dice** | 85/85 | −0.0646 | [−0.0800, −0.0494] | <0.001 | **kokoh** |

Uji tren monoton (Spearman pada 340 citra ber-SNR, bukan pada 3 rata-rata):

| Tren | ρ | 95% CI | p | putusan |
|---|---|---|---|---|
| SNR → **CIMT MAE** | −0.100 | [−0.208, +0.007] | **0.066** | **TIDAK SIGNIFIKAN** |
| SNR → Dice | +0.231 | — | 1.6e-05 | **kokoh** |

### ATURAN PENULISAN STRATIFIKASI (wajib)

1. **Klaim "CIMT error rises 50% from best to worst SNR" DICABUT** dari Highlight, abstrak,
   Results, dan Kesimpulan. Tren SNR pada **endpoint tidak terbukti** (p=0.066).
2. Yang boleh: **kualitas akuisisi (SNR) memengaruhi Dice secara kokoh** (ρ=+0.23, p=1.6e-5);
   efeknya pada endpoint **searah tetapi tidak mencapai signifikansi** pada n=340.
   Ini sendiri menarik: SNR menggerakkan overlap tanpa terbukti menggerakkan milimeter —
   contoh lain dari disosiasi Dice/endpoint yang jadi tesis paper.
3. **Kontras morfologi 2-vs-3 DICABUT** sepenuhnya (p=0.359). Jangan dipakai sebagai bukti apa pun.
4. **Munich–Toronto BOLEH dipertahankan** (sebaran Dice 6,5 poin; t besar). Beri CI.
5. Setiap klaim strata **wajib** memuat CI. Tanpa CI, jangan ditulis.

---

## 15. R7 — Di mana letak keunggulan endpoint: citra tipikal atau ekor? (2026-07-14)

Sumber: `Experiment baru 2/results/R7_failure_tail.md`. Test hold-out, **n = 401 citra**,
identik untuk semua model, rata-rata 5 fold per citra.
Selisih d = err(baseline) − err(usulan); **d > 0 = usulan lebih baik**.

Menjawab paradoks §13c: ResU-Net MAE-nya jauh lebih buruk (0.1986 vs 0.1199) tetapi Wilcoxon
menyebutnya SERI. Sebabnya **bukan** kesalahan — **Wilcoxon menguji median, MAE adalah mean**.

### 15a. Citra tipikal — SERI

| Baseline | median d (mm) | win-rate | Wilcoxon p |
|---|---|---|---|
| Swin-UNet | +0.0001 | 50% | 0.67 |
| ResU-Net | +0.0003 | 50% | 0.10 |
| UNet++ | +0.0009 | 52% | 0.15 |
| U-Mamba (run 1) | +0.0052 | 64% | 1.6e-07 |
| TransUNet | +0.0056 | 58% | 9.3e-05 |
| Attention U-Net | +0.0078 | 65% | 2e-12 |
| U-Net | +0.0096 | 66% | 9.8e-14 |
| SegFormer | +0.0275 | 70% | 4.1e-17 |
| BC-VMamba region-only | +0.0054 | 63% | 1.2e-08 |

Pada citra biasa, model usulan **tidak lebih baik** dari tiga baseline terkuat.

### 15b. Ekor kegagalan — TIDAK seri

Jumlah citra (dari 401) dengan error CIMT di atas ambang. τ disapu agar tak terlihat dipilih.

| Model | τ > 0.3 mm | τ > 0.5 mm | τ > 1.0 mm |
|---|---|---|---|
| **BC-VMamba (usulan)** | **12** | **2** | **1** |
| Swin-UNet | 19 | 6 | 2 |
| U-Mamba | 22 | 10 | 8 |
| TransUNet | 22 | 9 | 5 |
| UNet++ | 23 | 10 | 4 |
| Attention U-Net | 32 | 17 | 8 |
| SegFormer | 33 | 4 | 3 |
| ResU-Net | 39 | 26 | 12 |
| U-Net | 53 | 29 | 18 |

McNemar eksak pada pasangan diskordan (τ = 0.5 mm), format `gagal-hanya-di-baseline / gagal-hanya-di-usulan`:

| Baseline | diskordan | p |
|---|---|---|
| ResU-Net | 24 / 0 | **1.2e-07** |
| U-Net | 28 / 1 | 1.1e-07 |
| Attention U-Net | 16 / 1 | 0.00027 |
| U-Mamba | 9 / 1 | 0.021 |
| UNet++ | 9 / 1 | 0.021 |
| TransUNet | 7 / 0 | 0.016 |
| Swin-UNet | 6 / 2 | **0.29 (SERI)** |
| SegFormer | 4 / 2 | 0.69 (SERI) |

### 15c. Selisih MEAN (statistik yang benar untuk MAE) + CI bootstrap

CI 95%, 5.000 resample atas citra. **Semua tidak memuat 0**, termasuk Swin (nyaris menyentuh).

| Baseline | mean d (mm) | 95% CI |
|---|---|---|
| ResU-Net | +0.0786 | [+0.0470, +0.1160] |
| SegFormer | +0.0831 | [+0.0222, +0.1954] |
| U-Net | +0.1327 | [+0.0827, +0.1886] |
| Attention U-Net | +0.0559 | [+0.0298, +0.0892] |
| U-Mamba | +0.0495 | [+0.0196, +0.0866] |
| TransUNet | +0.0453 | [+0.0152, +0.0883] |
| UNet++ | +0.0272 | [+0.0101, +0.0494] |
| **Swin-UNet** | +0.0472 | **[+0.0006, +0.1322]** |
| BC-VMamba region-only | +0.0040 | [+0.0015, +0.0064] |

### ATURAN PENULISAN R7 (wajib)

1. **Wajib laporkan median DAN mean.** Melaporkan hanya Wilcoxon menyembunyikan efek nyata;
   melaporkan hanya mean menyembunyikan bahwa citra tipikal seri. Keduanya, selalu.
2. **DILARANG menulis "lowest CIMT error"** tanpa kualifikasi. Klaim yang SAH:
   > Pada citra tipikal model usulan **setara** dengan baseline terkuat; keunggulannya pada
   > endpoint berasal dari **berkurangnya kegagalan ketebalan katastrofik** — yaitu justru
   > kesalahan yang berbahaya secara klinis.
3. **Swin-UNet tetap tak terbedakan pada SETIAP uji**: median seri (p=0.67), ekor seri
   (p=0.29), CI mean nyaris menyentuh nol ([+0.0006, ...]). Pembeda terhadap Swin adalah
   **efisiensi** (9.63 M vs 34.17 M), **bukan akurasi**. Tulis persis begitu.
4. Ambang τ **wajib** dilaporkan sebagai sapuan (0.3/0.5/1.0), bukan satu nilai terpilih.
5. Angka usulan 12 / 2 / 1 adalah rujukan ekor; jangan tukar dengan angka val pool.
