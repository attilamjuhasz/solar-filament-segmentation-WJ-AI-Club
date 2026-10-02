import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0,'src')
from assemble import load_cands, assemble, cand_score
P=json.load(open('configs/assemble_v2.json'))
def run(cdir):
    st=sorted(f[:-4] for f in os.listdir(cdir) if f.endswith('.pkl'))
    C=load_cands(cdir, st)
    n=[];keep=[];allc=[]
    for s in st:
        fin=assemble(C[s],P); n.append(len(fin))
        for c in C[s]:
            if c['level'] in P['levels']: allc.append((c['q'],c['mean_p'],c['q']*c['mean_p'],c['area50']))
    a=np.array(allc)
    return st,np.array(n),a
for name,cd in (('val','runs/s2_r34/cands_val_tta_last'),('test','runs/s2_r34/cands_test_tta_last')):
    st,n,a=run(cd)
    print(f"{name}: stems {len(st)} kept/stem mean {n.mean():.2f} median {np.median(n):.0f} zero {int((n==0).sum())} | cands/stem {len(a)/len(st):.1f} | q med {np.median(a[:,0]):.3f} mean_p med {np.median(a[:,1]):.3f} score>=.225 frac {np.mean(a[:,2]>=.225):.3f} | area50 med {np.median(a[:,3]):.0f}")
d=json.load(open('data/cache/disk.json'))
m=pd.read_csv('data/cache/meta.csv'); tr=set(m.file_name.str[:-5])
for name,keys in (('train',[k for k in d if k in tr]),('test',[k for k in d if k not in tr])):
    r=np.array([d[k]['r'] for k in keys]); iqr=np.array([d[k]['iqr'] for k in keys]); med=np.array([d[k]['med'] for k in keys])
    print(name, len(keys), 'r pct', np.percentile(r,[0,5,50,95,100]).round(1), 'med', np.percentile(med,[5,50,95]).round(0), 'iqr', np.percentile(iqr,[5,50,95]).round(1))
