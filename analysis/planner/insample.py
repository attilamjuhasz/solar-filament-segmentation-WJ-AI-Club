import sys, os, numpy as np, pandas as pd, pickle
sys.path.insert(0,'src')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
from s1 import postprocess_s1
from assemble import FastGT
m=pd.read_csv('data/cache/meta.csv'); m['stem']=m.file_name.str[:-5]
rng=np.random.default_rng(0)
tr=sorted(set(m[m.fold!=0].stem)); va=sorted(set(m[m.fold==0].stem))
tr=list(rng.choice(tr,80,replace=False)); va=list(rng.choice(va,80,replace=False))
P=dict(t_hi=0.6,t_lo=0.45,gap=8,min_area=400,head=0)
for name,st in (('train(in-sample)',tr),('val(oof)',va)):
    gt=FastGT(st)
    fin={s:postprocess_s1(np.load(f'runs/s1_r34_f0/probs_plain/{s}.npy').astype(np.float32)/255.,s,**P) for s in st}
    print(name, gt.pq(fin), flush=True)
    # proposal-level stats: mean_p of level A proposals and fraction with best IoU>.5
    mp=[]
    for s in st[:40]:
        props=pickle.load(open(f'runs/s1_r34_f0/props_plain/{s}.pkl','rb'))
        mp+= [d['mean_p'] for d in props if d['level']=='A']
    print('   level-A props/stem %.1f  mean_p median %.3f' % (len(mp)/40, np.median(mp)))
