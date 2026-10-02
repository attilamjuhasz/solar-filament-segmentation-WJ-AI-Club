import json, numpy as np, pandas as pd
from pycocotools import mask as mu
from collections import defaultdict
d=json.load(open('data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
m=pd.read_csv('data/cache/meta.csv'); fold=dict(zip(m.image_id,m.fold))
A=defaultdict(list)
for a in d['annotations']:
    A[a['image_id']].append(mu.merge(mu.frPyObjects(a['segmentation'],2048,2048)))
byfile=defaultdict(list)
for im in d['images']: byfile[im['file_name']].append(im['id'])
st=defaultdict(lambda: np.zeros(4)); gts=defaultdict(list)
for fn,ids in byfile.items():
    f=fold[ids[0]]
    for i in ids: gts[f]+= list(mu.area(A[i])) if A[i] else []
    if len(ids)<2: continue
    for i in ids:
        for j in ids:
            if i==j: continue
            P,G=A[i],A[j]
            if not P or not G:
                st[f]+=[0,0,len(P),len(G)]; continue
            iou=np.asarray(mu.iou(P,G,[0]*len(G))); mm=iou>.5
            tp=mm.any(1).sum(); st[f]+=[iou[mm].sum(),tp,len(P)-tp,len(G)-mm.any(0).sum()]
for f in sorted(st):
    S,TP,FP,FN=st[f]; a=np.array(gts[f])
    print(f"fold {f}: pairwise annotator PQ {S/(TP+.5*FP+.5*FN):.3f} (RQ {TP/(TP+.5*FP+.5*FN):.3f}) | GT inst {len(a)}, <400px {np.mean(a<400):.3f}, <1000px {np.mean(a<1000):.3f}, median area {np.median(a):.0f}")
