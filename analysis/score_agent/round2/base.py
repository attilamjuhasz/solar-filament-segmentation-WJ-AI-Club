"""Shared context for the optimizer session: ep8 (last) TTA candidates on fold-0 val, v2 config."""
import json, os, pickle, sys, time
import cv2
import numpy as np
os.environ.setdefault("CANDS", "runs/s2_r34/cands_val_tta_last")
os.environ.setdefault("TAG", "ep8")
from ana import ROOT, Ctx, assemble, assemble_prov, cand_mask, pq, cnt  # noqa
from pycocotools import mask as mu

V2 = json.load(open(os.path.join(ROOT, "configs/assemble_v2.json")))
OPT = os.path.dirname(os.path.abspath(__file__))


def half_report(ctx, fin, name):
    tA, tB = ctx.pq_counts(fin, ctx.A), ctx.pq_counts(fin, ctx.B)
    print(f"{name:60s} A={pq(tA):.4f} B={pq(tB):.4f} all={pq(tA + tB):.4f} {cnt(tA + tB)}", flush=True)
    return pq(tA), pq(tB), pq(tA + tB), tA, tB


def load_ctx(load_s1=True):
    ctx = Ctx(load_s1=load_s1)
    rows = pickle.load(open(os.path.join(OPT, "q1_rows_ep8.pkl"), "rb"))
    it = iter(rows)
    for s in ctx.stems:
        for c in ctx.C[s]:
            r = next(it)
            assert r[0] == s and r[2] == c["level"] and abs(r[3] - c["q"]) < 1e-9
            c["yv"] = r[8]
    return ctx


def props_of(stem):
    return pickle.load(open(os.path.join(ROOT, "runs/s1_r34_f0/props_tta", stem + ".pkl"), "rb"))
