import numpy as np
for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    k = int(t * 255)
    # soft = round(loc*255); soft > k  <=>  loc >= (k+0.5)/255
    print(f"thr={t}: t*255={t*255!r} int={k} -> effective loc >= {(k + .5) / 255:.4f} (delta {(k + .5) / 255 - t:+.4f})")
# brute-force consistency: soft>int(t*255) vs loc>t on a dense grid
loc = np.linspace(0.3, 1, 200001, dtype=np.float32)
soft = np.round(loc * 255).astype(np.uint8)
for t in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    print(t, "mismatch frac", np.mean((soft > int(t * 255)) != (loc > t)))
