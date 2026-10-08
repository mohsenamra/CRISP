# Worker functions (no torch import: safe for Windows 'spawn' workers)
import numpy as np, pickle, os, time
from bench_lib import GENERATORS, solve, features
LAMS = [0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999]
PS_K, PS_D = 0.7, 5
SIZES = {"setcover": {}, "cauction": {}, "facility": dict(n_customers=100, n_facilities=50), "indset": {}}

def gen_work(args):
    prob, k = args; f = os.path.join("data", prob, f"{k:04d}.pkl")
    if os.path.exists(f): return prob, k, None
    rng = np.random.default_rng(100_000 * (1 + list(GENERATORS).index(prob)) + k)
    P = GENERATORS[prob](rng, **SIZES[prob]); F, tlp, zlp = features(P)
    z, x, t, _, st = solve(P, tlim=900)
    pickle.dump(dict(P=P, F=F, t_lp=tlp, zlp=zlp, z=z, x=None if x is None else np.round(x), t_full=t, status=st), open(f, "wb"))
    return prob, k, t

def eval_work(args):
    prob, name, p, tpred = args; out = os.path.join("eval", prob, name)
    if os.path.exists(out): return
    r = pickle.load(open(os.path.join("data", prob, name), "rb")); P, z, tf = r["P"], r["z"], r["t_full"]
    ints = np.where(P["integ"])[0]; xh = (p >= 0.5).astype(int); s = np.maximum(p, 1 - p); tl = max(1.5 * tf, 2.0)   # reduced solve slower than this -> fallback
    rows = {}
    for lam in LAMS:
        sel = ints[s >= lam]; vals = xh[s >= lam]
        if len(sel) == 0:                       # nothing fixed: the reduced problem IS the full problem
            rows[lam] = dict(nfix=0, gap=0.0, status="optimal", t_red=tf, wrong=0); continue
        zl, _, t, _, st = solve(P, fix=dict(zip(sel.tolist(), vals.tolist())), tlim=tl)
        rows[lam] = dict(nfix=len(sel), gap=None if zl is None else max((zl - z) / abs(z), 0.0), status=st, t_red=t,
                         wrong=int((vals != r["x"][sel]).sum()))
    # PS baseline (Han et al. 2023): fix-free trust region around the top PS_K fraction by confidence, radius PS_D
    order = np.argsort(-s)[: int(PS_K * len(ints))]; sel = ints[order]; xs = xh[order]
    row = np.zeros(len(P["c"])); row[sel] = np.where(xs == 1, -1.0, 1.0)
    zl, _, t, _, st = solve(P, tlim=3 * tl, extra_row=(row, -np.inf, PS_D - xs.sum()))
    ps = dict(gap=None if zl is None else max((zl - z) / abs(z), 0.0), status=st, t_red=t)
    pickle.dump(dict(t_full=tf, t_lp=r["t_lp"], t_pred=tpred, n_int=len(ints), acc=float((xh == r["x"][ints]).mean()),
                     rows=rows, ps=ps), open(out, "wb"))
