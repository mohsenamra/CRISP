# Robustness check for the time budget: replaces the per-instance (oracle) budget tau_i = max(1.5 t_full_i, 2 s)
# by a budget a planner can set in advance, tau = max(c * median full solve time of the family, 2 s), c in {1.5, 3},
# and re-runs the calibration on exactly the same 2,000 splits (n = 100) as bench_analyze.py. No MILP is solved:
# outcomes under the new budget are derived from the stored solve records. Runs that hit the oracle budget and would
# get a larger budget have an unknown outcome; they are imputed pessimistically (worst loss, time tau + t_full), in
# calibration and test alike, so the reported risks are upper bounds and the speed-ups lower bounds.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pickle, os, glob, json, sys
from bench_analyze import env, crc, ltt, fastest, REPS, EPS, BF, GTOL
from bench_work import LAMS

def outcomes(R, mode, c=None):
    TF = np.array([r["t_full"] for r in R]); TOV = np.array([r["t_lp"] + r["t_pred"] for r in R])
    tau_or = np.maximum(1.5 * TF, 2.0)
    tau = tau_or if mode == "oracle" else np.full(len(R), max(c * np.median(TF), 2.0))
    N, K = len(R), len(LAMS)
    G = np.zeros((N, K)); T = np.zeros((N, K)); fb = np.zeros((N, K), bool); unk = np.zeros((N, K), bool); S = np.zeros((N, K))
    for i, r in enumerate(R):
        for k, l in enumerate(LAMS):
            row = r["rows"][l]; st, t = row["status"], row["t_red"]; S[i, k] = float(row["wrong"] > 0)
            if row["nfix"] == 0:                              # nothing fixed: this is the full solve, no budget
                G[i, k] = 0.0; T[i, k] = TOV[i] + TF[i]; continue
            if st == "optimal" and t <= tau[i]:
                G[i, k] = max(row["gap"] or 0.0, 0.0); T[i, k] = TOV[i] + t
            elif st == "optimal" or st == "infeasible" or tau[i] <= tau_or[i] + 1e-9:
                fb[i, k] = True; T[i, k] = TOV[i] + min(t, tau[i]) + TF[i]
            else:                                             # hit the oracle budget, new budget larger: unknown outcome
                unk[i, k] = True; G[i, k] = np.inf; T[i, k] = TOV[i] + tau[i] + TF[i]
    P1 = np.where(fb, 0.0, (G > EPS).astype(float)); LM = np.where(fb, 0.0, np.minimum(G, BF))
    return TF, T, P1, LM, fb, unk, tau

def run(prob, mode, c=None, ncal=100):
    R = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(os.path.join("eval", prob, "*.pkl")))]
    TF, T, P1, LM, fb, unk, tau = outcomes(R, mode, c); spd = TF[:, None] / T; N = len(R)
    rng = np.random.default_rng(0)
    for nc in (25, 50, 100):                                  # reproduce the RNG stream of bench_analyze.py
        if nc > N - 20: continue
        splits = [rng.permutation(N) for _ in range(REPS)]
        if nc == ncal: break
    out = dict(tau_seconds=float(tau[0]) if mode != "oracle" else None, median_full_time=float(np.median(TF)))
    for name, a, rule, L in [("CRC_meangap", 0.01, lambda L, a: crc(env(L), a, BF), LM), ("LTT_P(gap>1%)", 0.10, lambda L, a: ltt(env(L), a), P1)]:
        risk, above, sp, fbr, unkr = [], [], [], [], []
        for pm in splits:
            cal, te = pm[:ncal], pm[ncal:]; i0 = rule(L[cal], a); i = fastest(i0, T[cal], TF[cal])
            if i is None: risk.append(0.0); above.append(0.0); sp.append(1.0); fbr.append(0.0); unkr.append(0.0); continue
            rk = float(L[te, i].mean()); risk.append(rk); above.append(float(rk > a)); sp.append(float(np.median(spd[te, i])))
            fbr.append(float(fb[te, i].mean())); unkr.append(float(unk[te, i].mean()))
        out[name] = dict(realized_risk=float(np.mean(risk)), frac_splits_above_target=float(np.mean(above)),
                         speedup_median=float(np.median(sp)), fallback_rate=float(np.mean(fbr)), unknown_rate=float(np.mean(unkr)))
    return out

if __name__ == "__main__":
    res = {}
    for prob in ["setcover", "cauction", "facility", "indset", "ssoa"]:
        if not glob.glob(os.path.join("eval", prob, "*.pkl")): continue   # ssoa: also unzip application/results/ssoa_records.zip here
        res[prob] = {"oracle": run(prob, "oracle"), "c1.5": run(prob, "fixed", 1.5), "c3": run(prob, "fixed", 3.0)}
        print(prob, json.dumps({k: {m: {kk: round(vv, 4) for kk, vv in v[m].items()} for m in ("CRC_meangap", "LTT_P(gap>1%)")} for k, v in res[prob].items()}))
    json.dump(res, open(os.path.join("results", "summary_budget.json"), "w"), indent=1)
