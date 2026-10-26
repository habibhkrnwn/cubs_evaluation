"""Dice-CIMT scatter data: mean checkpoint Dice and probability-averaged ensemble error.

Usage: python derive_overlap.py <aligned_s42_run_dir> <output.csv>
"""
import sys,glob,numpy as np,pandas as pd
from scipy.stats import pearsonr,spearmanr
d=sys.argv[1]
csvs=[pd.read_csv(f) for f in sorted(glob.glob(d+'/test_fold*_per_image.csv'))]
probs=[np.load(f) for f in sorted(glob.glob(d+'/test_fold*_probs.npz'))]
def cimt(m,side,cf):
    cols=[];
    for c in range(m.shape[1]):
        r=np.flatnonzero(m[:,c])
        if r.size>=2: cols.append(r[-1]-r[0])
    return 0.0 if not cols else float(np.mean(cols))*side/256*cf
c0=csvs[0].set_index('name')
# calibrate formula on fold1
diff=[];
for n in c0.index[:40]:
    m=probs[0][n].astype(np.float32)>0.5
    diff.append(cimt(m,c0.loc[n,'roi_square_side'],c0.loc[n,'cf'])-c0.loc[n,'cimt_pred_mm'])
print('fold1 formula max|diff|',np.max(np.abs(diff)))
rows=[]
for n in c0.index:
    p=np.mean([pr[n].astype(np.float32) for pr in probs],axis=0)
    e=abs(cimt(p>0.5,c0.loc[n,'roi_square_side'],c0.loc[n,'cf'])-c0.loc[n,'cimt_gt_mm'])
    md=np.mean([c.set_index('name').loc[n,'dice'] for c in csvs])
    me=np.mean([c.set_index('name').loc[n,'imt_abs_err_mm'] for c in csvs])
    rows.append((n,md,me,e))
df=pd.DataFrame(rows,columns=['name','mean_dice','mean_err','ensprob_err'])
print('MAE ensprob',df.ensprob_err.mean())
for col in ['mean_err','ensprob_err']:
    print(col,'pearson',pearsonr(df.mean_dice,df[col]).statistic,'spearman',spearmanr(df.mean_dice,df[col]).statistic)
# Output in the figure format: one row per fixed-test image, with release and source.
out=pd.DataFrame({'image_id':df.name,'release':c0.loc[df.name,'release'].values,'center':c0.loc[df.name,'center'].values,
                  'mean_dice_5ckpt':df.mean_dice,'abs_cimt_err_ensemble_mm':df.ensprob_err})
out.to_csv(sys.argv[2],index=False)
