"""Compare JSON 'area' against pycocotools / cv2 rasterizations; find within-reading GT overlaps."""
import json, collections
import numpy as np, cv2
from pycocotools import mask as mu
import sys
sys.path.insert(0, 'src')
from common import load_inst
d = json.load(open('data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
anns = d['annotations']
rng = np.random.default_rng(0)
idx = rng.choice(len(anns), 400, replace=False)
rows = []
for i in idx:
    a = anns[i]
    poly = np.asarray(a['segmentation'][0], np.float64).reshape(-1, 2)
    rle = mu.merge(mu.frPyObjects(a['segmentation'], 2048, 2048))
    pc = float(mu.area(rle))
    m = np.zeros((2048, 2048), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    cvr = float(m.sum())
    # shoelace polygon area
    x, y = poly[:, 0], poly[:, 1]
    sh = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    # iou between pycocotools and cv2
    m2 = mu.decode(rle)
    iou = (m & m2).sum() / max((m | m2).sum(), 1)
    rows.append((a['area'], pc, cvr, sh, iou))
r = np.array(rows)
print('n', len(r))
for j, name in [(1, 'pycoco'), (2, 'cv2fill'), (3, 'shoelace')]:
    dif = r[:, 0] - r[:, j]
    print(f"area - {name}: exact-eq {np.mean(np.abs(dif) < 0.5):.3f} mean {dif.mean():.2f} medianrel {np.median(dif / r[:, 0]):.4f}")
print('IoU(pycoco, cv2fillPoly) median', np.median(r[:, 4]), 'p10', np.percentile(r[:, 4], 10))
print('area field integer?', np.mean(r[:, 0] == np.round(r[:, 0])))

# within-reading overlaps
by = collections.defaultdict(list)
for a in anns:
    by[a['image_id']].append(a)
n_ov_read = n_ov_pairs = 0
lost = []
for rid, al in by.items():
    rles = [mu.merge(mu.frPyObjects(a['segmentation'], 2048, 2048)) for a in al]
    iou = np.asarray(mu.iou(rles, rles, [0] * len(rles)))
    np.fill_diagonal(iou, 0)
    if (iou > 0).any():
        n_ov_read += 1
        n_ov_pairs += int((iou > 0).sum() // 2)
        lab = load_inst(rid)
        areas_lab = np.bincount(lab.ravel(), minlength=len(rles) + 1)[1:]
        tot = np.array([mu.area(r) for r in sorted(rles, key=lambda r: -mu.area(r))])
        # inst painted by -a['area'] order, approx
        lost.append((rid, int(iou.max() * 1000) / 1000, int((iou > 0).sum() // 2), int(tot.sum() - areas_lab.sum())))
print('readings with overlapping GT', n_ov_read, 'pairs', n_ov_pairs)
print('max pairwise IoU within reading among those, top:', sorted(lost, key=lambda t: -t[1])[:15])
print('any pair IoU>0.5 within a reading?', sum(t[1] > 0.5 for t in lost))
