# IEEE Access revision: post hoc analyses and figures

This folder holds the scripts for two parts of the IEEE Access version of the manuscript:

- the post hoc exploratory analyses;
- the vector figures.

All analyses run on a CPU and read only archived outputs. No model is trained or re-run.

Per-image derived records are not committed, in line with the repository policy on participant-level records. Running the scripts regenerates them locally in `ieee_access/data/`. The committed file `analysis/exploratory_numbers.json` contains only the aggregate estimates quoted in the manuscript.

## Inputs

1. **CUBS data:** set `CUBS_DATA_ROOT` as described in the top-level README.
2. **Archived per-image outputs.** Put the following folders in one directory, here called `KAGGLE_OUTPUTS`. Each holds the `*_per_image.csv` files written by the training notebooks:

   | Folder | Run | Also contains |
   | --- | --- | --- |
   | `bcv_v1/results/bcvmamba_a1/bc_vmamba_cubs_a1/` | BC-VMamba, region-only objective, seed 42 | |
   | `bcv_v2/results/bcvmamba_a1/bc_vmamba_cubs_lcimt/` | BC-VMamba, measurement-aligned, seed 42 | `test_fold*_probs.npz` |
   | `bcv_v3/results/bcvmamba_a1/bc_vmamba_cubs_lcimt/` | BC-VMamba, measurement-aligned, seed 1 | |
   | `bc-vmamba-cubs-a1/results/bcvmamba_a1/bc_vmamba_cubs_lcimt/bc_vmamba_cubs_a1_training_history.csv` | aligned seed-1 training log | |
   | `bl_v2/results/baselines/unet/` | U-Net baseline run | |
   | `bl_v4/results/baselines/transunet/` | TransUNet baseline run | |

3. **Reader comparison:** `ieee_access/data/reader_bland_altman.csv`. Build it from `results/R1_observer_matched.csv`, which `analysis/R1_observer_matched.py` writes, by keeping the columns `image_id, release, center, model_mm, msk_A1, msk_A1p, msk_A2, msk_A3` for the fixed test set.

## Steps

```bash
# 1. Dice-CIMT scatter data (probability-averaged ensemble of the aligned seed-42 run)
python ieee_access/data/derive_overlap.py \
    "$KAGGLE_OUTPUTS/bcv_v2/results/bcvmamba_a1/bc_vmamba_cubs_lcimt" \
    ieee_access/data/overlap_scatter.csv

# 2. Post hoc exploratory analyses
python -I ieee_access/analysis/exploratory.py "$KAGGLE_OUTPUTS" data/master_index.csv

# 3. Figures (needs pdfLaTeX with TikZ/PGFPlots and newtx)
CUBS_SOURCE_PDF=/path/to/source_manuscript.pdf python ieee_access/scripts/build_figures.py
```

`exploratory.py` asserts the previously reported values before it computes any new estimate:

- Table 2 Dice, MAE and tail counts;
- Table 3 paired differences;
- r = −0.456;
- Bland–Altman bias +0.0639 mm;
- the reader MAEs;
- the matched −0.0040 mm change and its 62.6% share.

It stops if any of these fails to match. Confidence intervals use the 238-cluster bootstrap: 5,000 replicates, seed 20260830.

`build_figures.py` crops two empirical panels from the source manuscript PDF. `CUBS_SOURCE_PDF` points to that file.
