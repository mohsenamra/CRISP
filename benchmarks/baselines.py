import numpy as np, pickle, glob, os, json
from bench_work import LAMS
from bench_analyze import env, crc, ltt, fastest, EPS, BF
from extra_analyze import mats, evaluate, summarize
REPS = 2000
def load(p): return [pickle.load(open(f, "rb")) for f in sorted(glob.glob(p))]
out = {}
for prob in ["setcover", "cauction", "facility", "indset"]:
    M = mats(load(os.path.join("eval", prob, "*.pkl")), LAMS); N = M["N"]; res = {}
    # fixed hand-set thresholds (no calibration at all)
    for lam in (0.9, 0.95, 0.99):
        i = LAMS.index(lam)
        res[f"handset_{lam}"] = dict(mean_gap=float(M["LM"][:, i].mean()), P_gap_gt1pct=float(M["P1"][:, i].mean()), P_not_opt=float(M["P0"][:, i].mean()),
                                     speedup_median=float(np.median(M["spd"][:, i])), speedup_total=float(M["TF"].sum() / M["T"][:, i].sum()))
    # empirical tuning on a validation set (no finite-sample correction, no envelope): most aggressive lambda whose
    # validation risk <= alpha  vs. CRISP (CRC / LTT) on the same calibration instances
    rng = np.random.default_rng(1)
    for lossname, lc, alphas in (("meangap", "LM", (0.005, 0.01, 0.02)), ("P(gap>1%)", "P1", (0.05, 0.1, 0.2))):
        for n in (25, 50, 100):
            for a in alphas:
                rows_e, rows_c = [], []
                for _ in range(REPS):
                    pm = rng.permutation(N); cal, te = pm[:n], pm[n:]
                    ok = np.where(M[lc][cal].mean(0) <= a)[0]; sp_cal = np.median(M['TF'][cal][:, None] / M['T'][cal], axis=0); i = int(ok[np.argmax(sp_cal[ok])]) if len(ok) else None
                    if i is None: rows_e.append((0.0, 1.0, 1.0, 1))
                    else: rows_e.append((float(M[lc][te, i].mean()), float(np.median(M["spd"][te, i])), float(M["TF"][te].sum() / M["T"][te, i].sum()), 0))
                    name = "CRC_meangap" if lc == "LM" else "LTT_P(gap>1%)"
                    rows_c.append(evaluate(M, M, name, a, cal, te))
                res[f"empirical_{lossname}|n{n}|a{a}"] = summarize(rows_e, a)
                res[f"CRISP_{'CRC' if lc=='LM' else 'LTT'}_{lossname}|n{n}|a{a}"] = summarize(rows_c, a)
    out[prob] = res
json.dump(out, open("summary_baselines.json", "w"), indent=1)
for p, r in out.items():
    print("==", p)
    for k, v in r.items():
        if k.startswith("handset"): print(f"  {k:28s} meangap {v['mean_gap']:.4f} P>1% {v['P_gap_gt1pct']:.3f} Pnotopt {v['P_not_opt']:.3f} sp {v['speedup_median']:.2f}")
        elif "|n25|" in k or "|n100|" in k: print(f"  {k:36s} risk {v['realized_risk']:.4f} above {v['frac_splits_above_target']:.3f} sp {v['speedup_median']:.2f}")
