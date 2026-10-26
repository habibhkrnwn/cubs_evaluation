"""Post hoc exploratory analyses of archived per-image fixed-test outputs.

Usage: python -I exploratory.py <kaggle_outputs_dir> <master_index.csv>

Reads only archived per-image CSVs, probability maps and training logs; no model is
trained or re-run. Writes tidy CSVs to ../data/ for the figures and every number
quoted in the manuscript to exploratory_numbers.json. Published values are asserted
before anything new is computed.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

K = Path(sys.argv[1])
MASTER = Path(sys.argv[2])
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'
B = 5000
SEED = 20260830
FAIL_MM = 0.1686  # procedure-matched A2-A1 MAE, the reported CIMT-failure threshold

RUNS = {
    'region_s42': K / 'bcv_v1/results/bcvmamba_a1/bc_vmamba_cubs_a1',
    'aligned_s42': K / 'bcv_v2/results/bcvmamba_a1/bc_vmamba_cubs_lcimt',
    'aligned_s1': K / 'bcv_v3/results/bcvmamba_a1/bc_vmamba_cubs_lcimt',
    'TransUNet': K / 'bl_v4/results/baselines/transunet',
    'U-Net': K / 'bl_v2/results/baselines/unet',
    'U-Mamba': K / 'md_bl_v4/results/baselines/umamba',
    'ResU-Net': K / 'md_bl_v5/results/baselines/resunet',
    'Attention U-Net': K / 'md_bl_v2/results/baselines/attention_unet',
    'SegFormer': K / 'md_bl_v3/results/baselines/segformer',
    'Swin-UNet': K / 'md_bl_v4/results/baselines/swin_unet',
}
# Table 2 order (decreasing Dice); BC-VMamba is the region-only seed-42 run.
ARCH = [('region_s42', 'BC-VMamba'), ('U-Mamba', 'U-Mamba'), ('ResU-Net', 'ResU-Net'), ('Attention U-Net', 'Attention U-Net'),
        ('TransUNet', 'TransUNet'), ('SegFormer', 'SegFormer'), ('U-Net', 'U-Net'), ('Swin-UNet', 'Swin-UNet')]
HISTORY = K / 'bc-vmamba-cubs-a1/results/bcvmamba_a1/bc_vmamba_cubs_lcimt/bc_vmamba_cubs_a1_training_history.csv'

out = {}


def load_folds(path, split='test'):
    fs = sorted(glob.glob(str(path / f'{split}_fold*_per_image.csv')))
    assert len(fs) == 5, path
    df = pd.concat([pd.read_csv(f).assign(fold=i + 1) for i, f in enumerate(fs)])
    df['signed'] = df.cimt_pred_mm - df.cimt_gt_mm
    return df


def per_image(df):
    """Average each metric over the five checkpoints (the Table 2 rule)."""
    g = df.groupby('name').agg(
        release=('release', 'first'), center=('center', 'first'),
        dice=('dice', 'mean'), precision=('precision', 'mean'), recall=('recall', 'mean'),
        pred=('cimt_pred_mm', 'mean'), ref=('cimt_gt_mm', 'first'),
        abs_err=('imt_abs_err_mm', 'mean'), signed=('signed', 'mean'),
        side=('roi_square_side', 'first'), cf=('cf', 'first'))
    g['px'] = g.cf * g.side / 256  # mm per pixel of the 256 x 256 grid
    g['err_px'] = g.abs_err / g.px
    g['ref_px'] = g.ref / g.px
    g['log2rp'] = np.log2(g.recall / g.precision)
    return g


# ---------- clusters and bootstrap ----------
master = pd.read_csv(MASTER).set_index('image_id')
runs = {k: load_folds(v) for k, v in RUNS.items()}
img = {k: per_image(v) for k, v in runs.items()}
names = img['region_s42'].index
for k in img:
    assert img[k].index.equals(names), k
cluster = master.loc[names, 'patient_id'].values
_, cid = np.unique(cluster, return_inverse=True)
assert cid.max() + 1 == 238
members = [np.flatnonzero(cid == c) for c in range(238)]
rng = np.random.default_rng(SEED)
DRAWS = [np.concatenate([members[c] for c in rng.integers(0, 238, 238)]) for _ in range(B)]


def boot(stat, *cols):
    """Point estimate and 95% percentile CI from the 238-cluster bootstrap."""
    cols = [np.asarray(c) for c in cols]
    est = stat(*cols)
    bs = np.array([stat(*[c[ix] for c in cols]) for ix in DRAWS])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return [float(est), float(lo), float(hi)]


rho = lambda a, b: spearmanr(a, b).statistic
r_p = lambda a, b: pearsonr(a, b).statistic

# ---------- assert published values ----------
for k, mae in [('region_s42', 0.1241), ('aligned_s42', 0.1200), ('aligned_s1', 0.1307),
               ('TransUNet', 0.1654), ('U-Net', 0.2528)]:
    assert round(img[k].abs_err.mean(), 4) == mae, (k, img[k].abs_err.mean())
TABLE2 = {'region_s42': (0.8442, 0.1241, 3), 'U-Mamba': (0.8369, 0.1765, 17), 'ResU-Net': (0.8287, 0.1986, 26),
          'Attention U-Net': (0.8258, 0.1760, 17), 'TransUNet': (0.8257, 0.1654, 9), 'SegFormer': (0.8224, 0.2031, 4),
          'U-Net': (0.8021, 0.2528, 29), 'Swin-UNet': (0.7956, 0.1673, 6)}
for k, (dice, mae, tail) in TABLE2.items():
    assert (round(img[k].dice.mean(), 4), round(img[k].abs_err.mean(), 4), int((img[k].abs_err > 0.5).sum())) == (dice, mae, tail), k
for k, diff in [('U-Net', 0.1287), ('Attention U-Net', 0.0519), ('ResU-Net', 0.0745), ('Swin-UNet', 0.0432),
                ('TransUNet', 0.0413), ('SegFormer', 0.0791), ('U-Mamba', 0.0525)]:
    assert round((img[k].abs_err - img['region_s42'].abs_err).mean(), 4) == diff, k
ov = pd.read_csv(DATA / 'overlap_scatter.csv').set_index('image_id').loc[names]
assert round(r_p(ov.mean_dice_5ckpt, ov.abs_cimt_err_ensemble_mm), 3) == -0.456
ba = pd.read_csv(DATA / 'reader_bland_altman.csv').set_index('image_id').loc[names]
assert np.allclose(ba.msk_A1, img['aligned_s42'].ref) and np.allclose(ba.model_mm, img['aligned_s42'].pred)
assert round((ba.model_mm - ba.msk_A1).mean(), 4) == 0.0639

# ---------- A. geometric determinants (region-only s42; aligned s42 replication) ----------
geo = {}
for k in ['region_s42', 'aligned_s42']:
    g = img[k]
    geo[k] = {
        'px_median_mm': float(g.px.median()), 'px_q1_q3': [float(g.px.quantile(.25)), float(g.px.quantile(.75))],
        'px_min_max': [float(g.px.min()), float(g.px.max())],
        'err_px_median': float(g.err_px.median()), 'err_px_q1_q3': [float(g.err_px.quantile(.25)), float(g.err_px.quantile(.75))],
        'share_err_below_1px': boot(lambda e: np.mean(e < 1), g.err_px),
        'rho_abs_err_px_size': boot(rho, g.abs_err, g.px),
        'rho_dice_px_size': boot(rho, g.dice, g.px),
        'rho_dice_ref_px': boot(rho, g.dice, g.ref_px),
        'pearson_dice_err_mm': boot(r_p, g.dice, g.abs_err),
        'pearson_dice_err_px': boot(r_p, g.dice, g.err_px),
        'share_recall_gt_precision': boot(lambda r, p: np.mean(r > p), g.recall, g.precision),
        'mean_log2_rp': boot(np.mean, g.log2rp),
        'mean_signed_mm': boot(np.mean, g.signed),
        'mean_signed_px': boot(np.mean, g.signed / g.px),
        'rho_logratio_cimt_vs_log2rp': boot(rho, np.log2(g.pred / g.ref), g.log2rp),
        'rho_dice_vs_abs_log2rp': boot(rho, g.dice, np.abs(g.log2rp)),
        'prop_bias_slope': boot(lambda d, m: np.polyfit(m, d, 1)[0], g.pred - g.ref, (g.pred + g.ref) / 2),
    }

    def r2gain(e, d, l):
        X1 = np.c_[np.ones_like(d), d]
        X2 = np.c_[X1, l]
        r2 = lambda X: 1 - np.sum((e - X @ np.linalg.lstsq(X, e, rcond=None)[0]) ** 2) / np.sum((e - e.mean()) ** 2)
        return r2(X2) - r2(X1)
    geo[k]['r2_dice_only'] = float(1 - np.sum((g.abs_err - np.poly1d(np.polyfit(g.dice, g.abs_err, 1))(g.dice)) ** 2) / np.sum((g.abs_err - g.abs_err.mean()) ** 2))
    geo[k]['r2_gain_abs_log2rp'] = boot(r2gain, g.abs_err, g.dice, np.abs(g.log2rp))
out['geometry'] = geo
g = img['region_s42']
g.assign(image_id=g.index, signed_px=g.signed / g.px, abs_log2rp=np.abs(g.log2rp)).round(6)[
    ['image_id', 'release', 'center', 'dice', 'precision', 'recall', 'ref', 'pred', 'abs_err', 'signed',
     'px', 'err_px', 'ref_px', 'signed_px', 'log2rp', 'abs_log2rp']].to_csv(DATA / 'geometry_points.csv', index=False)

# ---------- B. architectures with archived per-image outputs ----------
arch = []
for k, lab in ARCH:
    a = img[k]
    arch.append({
        'model': lab, 'dice': float(a.dice.mean()), 'mae': float(a.abs_err.mean()),
        'median_abs_err': boot(np.median, a.abs_err), 'p90_abs_err': boot(lambda e: np.percentile(e, 90), a.abs_err),
        'mean_signed_mm': boot(np.mean, a.signed), 'share_recall_gt_precision': boot(lambda r, p: np.mean(r > p), a.recall, a.precision),
        'mean_log2_rp': boot(np.mean, a.log2rp), 'mean_precision': float(a.precision.mean()), 'mean_recall': float(a.recall.mean()),
        'rho_dice_err': boot(rho, a.dice, a.abs_err), 'err_px_median': float(a.err_px.median()),
        'n_gt_0p5': int((a.abs_err > 0.5).sum()), 'n_gt_1p0': int((a.abs_err > 1.0).sum()),
    })
    e = np.sort(a.abs_err.values)
    pd.DataFrame({'abs_err': e, 'cdf': np.arange(1, len(e) + 1) / len(e)}).round(6).to_csv(DATA / f'ecdf_{lab.replace("-", "").replace(" ", "").lower()}.csv', index=False)
out['architectures'] = arch
pd.DataFrame([{k: (v[0] if isinstance(v, list) else v) for k, v in r.items()} |
              {f'{k}_lo': v[1] for k, v in r.items() if isinstance(v, list)} |
              {f'{k}_hi': v[2] for k, v in r.items() if isinstance(v, list)} for r in arch]).round(6).to_csv(DATA / 'architecture_summary.csv', index=False)
# Per-image errors across architectures, and architecture-level association of size balance with error.
E8 = np.stack([img[k].abs_err.values for k, _ in ARCH])
pair = [spearmanr(E8[i], E8[j]).statistic for i in range(8) for j in range(i + 1, 8)]
L8 = np.stack([img[k].log2rp.values for k, _ in ARCH])
S8 = np.stack([img[k].signed.values for k, _ in ARCH])
out['architecture_level'] = {
    'pairwise_rho_abs_err_min_max': [float(min(pair)), float(max(pair))],
    'rho_mean_log2rp_vs_mae': boot(lambda ix: rho(L8[:, ix].mean(1), E8[:, ix].mean(1)), np.arange(401)),
    'rho_mean_log2rp_vs_mean_signed': boot(lambda ix: rho(L8[:, ix].mean(1), S8[:, ix].mean(1)), np.arange(401)),
    'rho_median_abs_err_vs_mae': boot(lambda ix: rho(np.median(E8[:, ix], 1), E8[:, ix].mean(1)), np.arange(401)),
    'rho_p90_abs_err_vs_mae': boot(lambda ix: rho(np.percentile(E8[:, ix], 90, axis=1), E8[:, ix].mean(1)), np.arange(401)),
}

# ---------- C. discordance and failure anatomy (aligned s42 ensemble, as in Fig. 4) ----------
q1 = float(ov.mean_dice_5ckpt.quantile(0.25))
assert round(q1, 3) == 0.821
d, e = ov.mean_dice_5ckpt.values, ov.abs_cimt_err_ensemble_mm.values
low, fail = d < q1, e > FAIL_MM
quad = {'highDice_ok': int((~low & ~fail).sum()), 'highDice_fail': int((~low & fail).sum()),
        'lowDice_ok': int((low & ~fail).sum()), 'lowDice_fail': int((low & fail).sum())}
assert sum(quad.values()) == 401
a42 = img['aligned_s42']
out['discordance'] = {
    'dice_q1': q1, 'counts': quad, 'n_fail': int(fail.sum()),
    'share_fail_with_high_dice': boot(lambda f, l: np.sum(f & ~l) / np.sum(f), fail, low),
    'share_lowdice_without_fail': boot(lambda f, l: np.sum(l & ~f) / np.sum(l), fail, low),
    'auc_1minus_dice_cimt_fail': boot(lambda f, s: roc_auc_score(f, s), fail, 1 - d),
    'auc_abs_log2rp_cimt_fail': boot(lambda f, s: roc_auc_score(f, s), fail, np.abs(a42.log2rp.values)),
}
# Shape of the ensemble masks, from the archived probability maps.
probs = [np.load(f) for f in sorted(glob.glob(str(RUNS['aligned_s42'] / 'test_fold*_probs.npz')))]
assert len(probs) == 5
feat = []
for n in names:
    m = np.mean([p[n].astype(np.float32) for p in probs], axis=0) > 0.5
    ext, cnt = [], []
    for c in range(m.shape[1]):
        r = np.flatnonzero(m[:, c])
        if r.size >= 2:
            ext.append(r[-1] - r[0] + 1)
            cnt.append(r.size)
    ext, cnt = np.array(ext), np.array(cnt)
    feat.append({'valid_cols': len(ext), 'gap_col_share': float(np.mean(cnt < ext)) if len(ext) else np.nan,
                 'thick_cv': float(np.std(ext) / np.mean(ext)) if len(ext) else np.nan})
feat = pd.DataFrame(feat, index=names)
disc = pd.DataFrame({'image_id': names, 'center': ov.center.values, 'release': ov.release.values, 'dice': d, 'abs_err': e,
                     'quadrant': np.select([~low & ~fail, ~low & fail, low & ~fail], ['highDice_ok', 'highDice_fail', 'lowDice_ok'], 'lowDice_fail'),
                     'log2rp': a42.log2rp.values, 'px': a42.px.values}).join(feat, on='image_id')
disc.round(6).to_csv(DATA / 'discordance_points.csv', index=False)
anat = {}
for q in ['highDice_ok', 'highDice_fail', 'lowDice_ok', 'lowDice_fail']:
    s = disc[disc.quadrant == q]
    anat[q] = {c: float(s[c].median()) for c in ['valid_cols', 'gap_col_share', 'thick_cv', 'log2rp', 'px', 'abs_err', 'dice']}
out['discordance']['anatomy_median'] = anat
out['discordance']['share_images_with_gap_columns'] = float(np.mean(disc.gap_col_share > 0))

# ---------- D. model-selection practice ----------
h = pd.read_csv(HISTORY)
assert sorted(h.fold.unique()) == [1, 2, 3, 4, 5]
sel = []
h['eligible'] = False
for f, s in h.groupby('fold'):
    el = (s.epoch > 5) & (s.val_dice >= 0.95 * s.val_dice.max())
    h.loc[s.index, 'eligible'] = el
    se = s[el]
    bd = se.loc[se.val_dice.idxmax()]
    bm = se.loc[se.val_mae_cimt_mm.idxmin()]
    near = se[se.val_dice >= se.val_dice.max() - 0.002]
    sel.append({'fold': int(f), 'n_epochs': int(len(s)), 'n_eligible': int(el.sum()),
                'epoch_best_dice': int(bd.epoch), 'epoch_best_mae': int(bm.epoch),
                'mae_at_best_dice': float(bd.val_mae_cimt_mm), 'min_mae': float(bm.val_mae_cimt_mm),
                'dice_at_best_mae': float(bm.val_dice), 'max_dice': float(bd.val_dice),
                'regret_mm': float(bd.val_mae_cimt_mm - bm.val_mae_cimt_mm),
                'near_max_dice_mae_range': [float(near.val_mae_cimt_mm.min()), float(near.val_mae_cimt_mm.max())],
                'n_near_max_dice': int(len(near)),
                'rho_dice_mae_eligible': float(spearmanr(se.val_dice, se.val_mae_cimt_mm).statistic)})
h[['fold', 'epoch', 'val_dice', 'val_mae_cimt_mm', 'eligible']].round(6).to_csv(DATA / 'training_dynamics.csv', index=False)
pd.DataFrame(sel).drop(columns='near_max_dice_mae_range').round(6).to_csv(DATA / 'selection_summary.csv', index=False)
# Fifteen fold checkpoints of BC-VMamba on the fixed test set.
ck = []
for k in ['region_s42', 'aligned_s42', 'aligned_s1']:
    for f, s in runs[k].groupby('fold'):
        s = s.set_index('name').loc[names]
        ck.append({'run': k, 'fold': int(f), 'dice': float(s.dice.mean()), 'mae': float(s.imt_abs_err_mm.mean())})
ck = pd.DataFrame(ck)
ck.round(6).to_csv(DATA / 'checkpoints15.csv', index=False)
ckd = {(k, f): runs[k][runs[k].fold == f].set_index('name').loc[names] for k, f in zip(ck.run, ck.fold)}
D15 = np.stack([v.dice.values for v in ckd.values()])
E15 = np.stack([v.imt_abs_err_mm.values for v in ckd.values()])
out['selection'] = {
    'folds': sel,
    'median_regret_mm': float(np.median([s['regret_mm'] for s in sel])),
    'checkpoints_dice_range': [float(ck.dice.min()), float(ck.dice.max())],
    'checkpoints_mae_range': [float(ck.mae.min()), float(ck.mae.max())],
    'checkpoints_rho_dice_negmae': boot(lambda ix: rho(D15[:, ix].mean(1), -E15[:, ix].mean(1)), np.arange(401)),
}

# ---------- E. sources with reader variability (aligned s42, as in the reader analysis) ----------
src = []
order = ['Cyprus', 'Pisa_clin', 'Munich', 'Pisa_tech', 'Porto', 'Torino', 'Toronto']
for c in order:
    ix = np.flatnonzero(a42.center.values == c)
    sub_c = cid[ix]
    _, sc = np.unique(sub_c, return_inverse=True)
    mem = [ix[sc == u] for u in range(sc.max() + 1)]
    r2 = np.random.default_rng(SEED)
    draws = [np.concatenate([mem[u] for u in r2.integers(0, len(mem), len(mem))]) for _ in range(B)]

    def bsub(fn, col):
        col = np.asarray(col)
        bs = [fn(col[dd]) for dd in draws]
        return [float(fn(col[ix])), *map(float, np.percentile(bs, [2.5, 97.5]))]
    src.append({'center': c, 'n': int(len(ix)), 'clusters': len(mem),
                'dice': bsub(np.mean, a42.dice), 'mae_mm': bsub(np.mean, a42.abs_err), 'mae_px': bsub(np.mean, a42.err_px),
                'px_median': float(a42.px.values[ix].mean()),
                'a1p_mae': bsub(np.mean, (ba.msk_A1p - ba.msk_A1).abs()), 'a2_mae': bsub(np.mean, (ba.msk_A2 - ba.msk_A1).abs()),
                'bias_mm': bsub(np.mean, a42.signed)})
out['sources'] = src
rows = []
for s in src:
    row = {'center': s['center'], 'n': s['n'], 'px': s['px_median']}
    for k in ['dice', 'mae_mm', 'mae_px', 'a1p_mae', 'a2_mae', 'bias_mm']:
        row[k], row[k + '_lo'], row[k + '_hi'] = s[k]
    rows.append(row)
pd.DataFrame(rows).round(6).to_csv(DATA / 'source_strata.csv', index=False)
assert round((ba.msk_A2 - ba.msk_A1).abs().mean(), 4) == 0.1686 and round((ba.msk_A1p - ba.msk_A1).abs().mean(), 4) == 0.1319

# ---------- F. run stability and how the aligned loss acts ----------
R, A, S1 = img['region_s42'], img['aligned_s42'], img['aligned_s1']
top = lambda s: set(s.sort_values(ascending=False).index[:40])
jac = lambda x, y: len(top(x) & top(y)) / len(top(x) | top(y))
out['runs'] = {
    'rho_err_region_aligned42': boot(rho, R.abs_err, A.abs_err),
    'rho_err_aligned42_aligned1': boot(rho, A.abs_err, S1.abs_err),
    'top10pct_jaccard_aligned42_aligned1': jac(A.abs_err, S1.abs_err),
    'top10pct_jaccard_region_aligned42': jac(R.abs_err, A.abs_err),
    'delta_precision_aligned_minus_region': boot(np.mean, A.precision - R.precision),
    'delta_recall_aligned_minus_region': boot(np.mean, A.recall - R.recall),
    'delta_signed_aligned_minus_region': boot(np.mean, A.signed - R.signed),
    'delta_abs_aligned_minus_region': boot(np.mean, A.abs_err - R.abs_err),
    'share_improved': float(np.mean(A.abs_err < R.abs_err)),
    'dice_by_run': {k: float(img[k].dice.mean()) for k in ['region_s42', 'aligned_s42', 'aligned_s1']},
}
assert round(out['runs']['delta_abs_aligned_minus_region'][0], 4) == -0.0040
assert round(out['runs']['share_improved'] * 100, 1) == 62.6
pd.DataFrame({'image_id': names, 'region_s42': R.abs_err.values, 'aligned_s42': A.abs_err.values, 'aligned_s1': S1.abs_err.values}).round(6).to_csv(DATA / 'run_pairs.csv', index=False)

# ---------- misc ----------
out['ref_cimt_mm'] = {'median': float(R.ref.median()), 'q1_q3': [float(R.ref.quantile(.25)), float(R.ref.quantile(.75))]}
out['ensemble_vs_checkpoint_mae'] = {'ensemble': float(ov.abs_cimt_err_ensemble_mm.mean()), 'checkpoint_mean': float(A.abs_err.mean()),
                                     'single_fold_range': [float(ck[ck.run == 'aligned_s42'].mae.min()), float(ck[ck.run == 'aligned_s42'].mae.max())]}
(HERE / 'exploratory_numbers.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
