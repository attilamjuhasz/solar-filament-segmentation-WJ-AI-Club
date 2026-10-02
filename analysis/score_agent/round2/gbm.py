"""Tiny histogram gradient-boosting (logistic loss on soft targets in [0,1], sample weights). numpy only."""
import numpy as np


class GBM:
    def __init__(self, n_trees=300, lr=0.03, depth=3, min_leaf=30, l2=1.0, subsample=0.8, colsample=0.8, nbins=32, seed=0):
        self.__dict__.update(locals()); del self.__dict__["self"]

    def _bin(self, X):
        return np.stack([np.searchsorted(e, X[:, j], side="right") for j, e in enumerate(self.edges)], 1).astype(np.int16)

    def fit(self, X, y, w=None):
        X = np.nan_to_num(np.asarray(X, float), nan=-1.0)
        n, d = X.shape
        w = np.ones(n) if w is None else np.asarray(w, float)
        self.edges = [np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.nbins + 1)[1:-1])) for j in range(d)]
        B = self._bin(X)
        rng = np.random.default_rng(self.seed)
        p0 = np.clip((w * y).sum() / w.sum(), 1e-3, 1 - 1e-3)
        self.f0 = np.log(p0 / (1 - p0))
        F = np.full(n, self.f0)
        self.trees = []
        for t in range(self.n_trees):
            p = 1 / (1 + np.exp(-F))
            g = w * (p - y)
            h = w * np.maximum(p * (1 - p), 1e-6)
            rows = np.nonzero(rng.random(n) < self.subsample)[0]
            cols = np.nonzero(rng.random(d) < self.colsample)[0]
            if not len(cols):
                cols = np.array([rng.integers(d)])
            tree = self._grow(B, g, h, w, rows, cols, 0)
            self.trees.append(tree)
            F += self.lr * self._apply(tree, B)
        return self

    def _grow(self, B, g, h, w, idx, cols, depth):
        G, H = g[idx].sum(), h[idx].sum()
        if depth >= self.depth or w[idx].sum() < 2 * self.min_leaf:
            return ("leaf", -G / (H + self.l2))
        best = (0.0, None)
        base = G * G / (H + self.l2)
        for j in cols:
            nb = len(self.edges[j]) + 1
            b = B[idx, j]
            gs = np.bincount(b, g[idx], nb); hs = np.bincount(b, h[idx], nb); ws = np.bincount(b, w[idx], nb)
            GL, HL, WL = np.cumsum(gs)[:-1], np.cumsum(hs)[:-1], np.cumsum(ws)[:-1]
            WR = ws.sum() - WL
            gain = GL ** 2 / (HL + self.l2) + (G - GL) ** 2 / (H - HL + self.l2) - base
            gain[(WL < self.min_leaf) | (WR < self.min_leaf)] = -1
            k = int(np.argmax(gain)) if len(gain) else 0
            if len(gain) and gain[k] > best[0]:
                best = (gain[k], (j, k))
        if best[1] is None:
            return ("leaf", -G / (H + self.l2))
        j, k = best[1]
        left = idx[B[idx, j] <= k]
        right = idx[B[idx, j] > k]
        return ("split", j, k, self._grow(B, g, h, w, left, cols, depth + 1), self._grow(B, g, h, w, right, cols, depth + 1))

    def _apply(self, tree, B):
        out = np.zeros(len(B))
        stack = [(tree, np.arange(len(B)))]
        while stack:
            node, idx = stack.pop()
            if node[0] == "leaf":
                out[idx] = node[1]
            else:
                _, j, k, L, R = node
                m = B[idx, j] <= k
                stack.append((L, idx[m])); stack.append((R, idx[~m]))
        return out

    def predict(self, X):
        X = np.nan_to_num(np.asarray(X, float), nan=-1.0)
        B = self._bin(X)
        F = np.full(len(X), self.f0)
        for t in self.trees:
            F += self.lr * self._apply(t, B)
        return 1 / (1 + np.exp(-F))

    def importance(self, d):
        imp = np.zeros(d)
        def walk(n):
            if n[0] == "split":
                imp[n[1]] += 1; walk(n[3]); walk(n[4])
        for t in self.trees:
            walk(t)
        return imp / imp.sum()
