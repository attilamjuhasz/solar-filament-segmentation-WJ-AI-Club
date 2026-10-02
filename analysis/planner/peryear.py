import sys, os, json, numpy as np, pandas as pd
from collections import defaultdict
sys.path.insert(0,'src')
from assemble import load_cands, assemble, FastGT
from common import load_inst
P=json.load(open('configs/assemble_v2.json'))
cd='runs/s2_r34/cands_val_tta_last'
st=sorted(f[:-4] for f in os.listdir(cd) if f.endswith('.pkl'))
C=load_cands(cd,st); fin={s:assemble(C[s],P) for s in st}
m=pd.read_csv('data/cache/meta.csv'); m['stem']=m.file_name.str[:-5]; m=m[m.stem.isin(st)]
gt=FastGT(st)
# per-reading stats: FastGT.by_stem[stem] is list in meta order -> map to image_id
rows=[]
for s,g in m.groupby('stem'):
    for (lab,areas),rid in zip(gt.by_stem[s], g.image_id):
        class One: pass
        sub=FastGT.__new__(FastGT); sub.by_stem={s:[(lab,areas)]}
        S,TP,FP,FN=sub.stats(s,fin[s])
        rows.append(dict(stem=s,rid=rid,ann=rid.split('-')[0],year=int(s[:4]),site=s[-2:],nread=len(g),S=S,TP=TP,FP=FP,FN=FN))
R=pd.DataFrame(rows)
def pq(df): S,TP,FP,FN=df[['S','TP','FP','FN']].sum(); return S/(TP+.5*FP+.5*FN)
print('all', round(pq(R),4), len(R))
R['yb']=pd.cut(R.year,[2010,2014,2017,2020,2022],labels=['11-14','15-17','18-20','21-22'])
for k,g in R.groupby('yb',observed=True): print('years',k,'readings',len(g),'PQ %.3f'%pq(g))
for k,g in R.groupby('nread'): print('n_readings',k,'readings',len(g),'PQ %.3f'%pq(g))
for k,g in R.groupby('site'): print('site',k,len(g),'PQ %.3f'%pq(g))
a=R.groupby('ann').apply(lambda g: pd.Series(dict(n=len(g),pq=pq(g))))
a=a[a.n>=6].sort_values('pq'); print('per-annotator (>=6 readings): min %.3f median %.3f max %.3f' % (a.pq.min(), a.pq.median(), a.pq.max())); print(a.round(3).to_string())
