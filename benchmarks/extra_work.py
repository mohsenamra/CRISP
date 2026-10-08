# CRISP supplementary experiments - worker functions (no torch import: safe for Windows 'spawn' workers).
#   shift : distribution shift (same size, different generator parameters)
#   large : size transfer (predictor trained on standard size, applied to 1.5-2x larger instances)
#   scip  : second exact solver (SCIP) on the standard evaluation instances, same predictions
import numpy as np, pickle, os, time
from bench_lib import GENERATORS, solve, features
from bench_work import LAMS
LX = [0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9999]      # coarser grid for the expensive runs (subset of LAMS)
SHIFT = {"setcover": dict(density=0.075),                                      # denser covering matrix
         "cauction": dict(add_item_prob=0.75),                                 # larger bundles
         "facility": dict(n_customers=100, n_facilities=50, ratio=3.0),        # tighter capacities
         "indset":   dict(n_nodes=400, affinity=5)}                            # denser graphs (smaller, so solve times stay comparable)
LARGE = {"setcover": dict(nrows=750),
         "cauction": dict(n_items=200, n_bids=1000),                       # = Gasse et al. "medium"
         "facility": dict(n_customers=200, n_facilities=100),               # = Gasse et al. "medium"
         "indset":   dict(n_nodes=750)}
CFG = {"shift": SHIFT, "large": LARGE}
SEED = {"shift": 5_000_000, "large": 7_000_000}
TLIM_FULL = 1200.0

def gen_x(args):
    exp, prob, k = args; f = os.path.join("data_x", exp, prob, f"{k:04d}.pkl")
    if os.path.exists(f): return
    rng = np.random.default_rng(SEED[exp] + 100_000 * (1 + list(GENERATORS).index(prob)) + k)
    P = GENERATORS[prob](rng, **CFG[exp][prob]); F, tlp, zlp = features(P)
    z, x, t, _, st = solve(P, tlim=TLIM_FULL)
    pickle.dump(dict(P=P, F=F, t_lp=tlp, zlp=zlp, z=z, x=None if x is None else np.round(x), t_full=t, status=st), open(f, "wb"))

def eval_x(args):
    """Hard fixing on a lambda grid (same rules as bench_work.eval_work, without the PS baseline)."""
    src, out, p, tpred, lams = args
    if os.path.exists(out): return
    r = pickle.load(open(src, "rb")); P, z, tf = r["P"], r["z"], r["t_full"]
    ints = np.where(P["integ"])[0]; xh = (p >= 0.5).astype(int); s = np.maximum(p, 1 - p); tl = max(1.5 * tf, 2.0)
    rows = {}
    for lam in lams:
        sel = ints[s >= lam]; vals = xh[s >= lam]
        if len(sel) == 0: rows[lam] = dict(nfix=0, gap=0.0, status="optimal", t_red=tf, wrong=0); continue
        zl, _, t, _, st = solve(P, fix=dict(zip(sel.tolist(), vals.tolist())), tlim=tl)
        rows[lam] = dict(nfix=len(sel), gap=None if zl is None else max((zl - z) / abs(z), 0.0), status=st, t_red=t,
                         wrong=int((vals != r["x"][sel]).sum()))
    pickle.dump(dict(t_full=tf, t_lp=r["t_lp"], t_pred=tpred, n_int=len(ints), acc=float((xh == r["x"][ints]).mean()), rows=rows),
                open(out, "wb"))

# ---------------------------------------------------------------- SCIP
def solve_scip(P, fix=None, tlim=1e9):
    from pyscipopt import Model, quicksum
    m = Model(); m.hideOutput(); m.setParam("limits/time", float(tlim)); m.setParam("limits/gap", 1e-6)
    for k, v in (("lp/threads", 1), ("timing/clocktype", 2)):          # single LP thread; wall-clock time limit
        try: m.setParam(k, v)
        except Exception: pass
    lb, ub = P["lb"].copy(), P["ub"].copy()
    if fix:
        for j, v in fix.items(): lb[j] = ub[j] = v
    x = [m.addVar(lb=lb[j], ub=ub[j], vtype="I" if P["integ"][j] else "C", obj=float(P["c"][j])) for j in range(len(P["c"]))]
    A = P["A"].tocsr(); fin = lambda v: np.isfinite(v) and abs(v) < 1e19
    for i in range(A.shape[0]):
        a, b = A.indptr[i], A.indptr[i + 1]; e = quicksum(float(v) * x[j] for j, v in zip(A.indices[a:b], A.data[a:b]))
        lo, up = P["lo"][i], P["up"][i]
        if fin(lo) and fin(up) and lo == up: m.addCons(e == float(lo))
        else:
            if fin(lo): m.addCons(e >= float(lo))
            if fin(up): m.addCons(e <= float(up))
    m.setMinimize(); t = time.time(); m.optimize(); t = time.time() - t; st = m.getStatus()
    if st == "infeasible": return None, t, "infeasible"
    if st != "optimal": return None, t, "timeout"
    return m.getObjVal(), t, "optimal"

def scip_work(args):
    src, out, p, tpred, lams = args
    if os.path.exists(out): return
    r = pickle.load(open(src, "rb")); P = r["P"]
    z, tf, st = solve_scip(P, tlim=TLIM_FULL)
    res = dict(z_highs=r["z"], t_full_highs=r["t_full"], z=z, t_full=tf, status_full=st, t_lp=r["t_lp"], t_pred=tpred, rows={})
    if st == "optimal":
        ints = np.where(P["integ"])[0]; xh = (p >= 0.5).astype(int); s = np.maximum(p, 1 - p); tl = max(1.5 * tf, 2.0)
        res["n_int"] = len(ints)
        for lam in lams:
            sel = ints[s >= lam]; vals = xh[s >= lam]
            if len(sel) == 0: res["rows"][lam] = dict(nfix=0, gap=0.0, status="optimal", t_red=tf, wrong=0); continue
            zl, t, s2 = solve_scip(P, fix=dict(zip(sel.tolist(), vals.tolist())), tlim=tl)
            res["rows"][lam] = dict(nfix=len(sel), gap=None if zl is None else max((zl - z) / abs(z), 0.0), status=s2, t_red=t,
                                    wrong=int((vals != r["x"][sel]).sum()))
    pickle.dump(res, open(out, "wb"))
