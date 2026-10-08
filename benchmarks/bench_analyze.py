# Analysis for the CRISP benchmarks: validity + speed of CRC / LTT / solve-free calibration vs PS and naive thresholds.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pickle, os, glob, json
from scipy.stats import binom
from bench_work import LAMS
REPS, DELTA, EPS, BF = 2000, 0.10, 0.01, 0.10
GTOL = 1e-6                                   # = solver relative MIP gap tolerance; smaller gaps are numerical noise
import sys
SELECT = "fastest" if "--first" not in sys.argv else "first"
env = lambda L: np.maximum.accumulate(L[:, ::-1], axis=1)[:, ::-1]
def crc(L, a, B=1.0):
    n = len(L); ok = (n / (n + 1)) * L.mean(0) + B / (n + 1) <= a
    return next((i for i in range(L.shape[1]) if ok[i:].all()), None)
def fastest(i0, Tcal, TFcal):
    """Any lambda >= the certified lambda_hat keeps the guarantee (loss is non-increasing in lambda);
    pick the one with the best median calibration speed-up."""
    if i0 is None: return None
    sp = np.median(TFcal[:, None] / Tcal[:, i0:], axis=0); return i0 + int(np.argmax(sp))
def ltt(L, a):
    best = None
    for i in range(L.shape[1] - 1, -1, -1):
        if binom.cdf(L[:, i].sum(), len(L), a) <= DELTA: best = i
        else: break
    return best
def analyse(prob):
    R = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(os.path.join("eval", prob, "*.pkl")))]
    if not R: return None
    N = len(R); TF = np.array([r["t_full"] for r in R]); TOV = np.array([r["t_lp"] + r["t_pred"] for r in R])
    st = np.array([[r["rows"][l]["status"] for l in LAMS] for r in R])
    G = np.array([[np.nan if r["rows"][l]["gap"] is None else r["rows"][l]["gap"] for l in LAMS] for r in R])
    W = np.array([[r["rows"][l]["wrong"] for l in LAMS] for r in R]); S = (W > 0).astype(float)      # solve-free surrogate
    NF = np.array([[r["rows"][l]["nfix"] / r["n_int"] for l in LAMS] for r in R])
    fb = st != "optimal"                                                                                # infeasible or time limit -> fallback
    T = TOV[:, None] + np.array([[r["rows"][l]["t_red"] for l in LAMS] for r in R]) + np.where(fb, TF[:, None], 0)
    spd = TF[:, None] / T
    P0 = np.where(fb, 0.0, np.nan_to_num(G) > GTOL); P1 = np.where(fb, 0.0, np.nan_to_num(G) > EPS); LM = np.where(fb, 0.0, np.minimum(np.nan_to_num(G), BF))
    Gm = np.where(st == "infeasible", np.inf, np.where(st == "optimal", G, np.nan))
    with np.errstate(invalid="ignore"): mono = int(np.nansum(np.diff(Gm, axis=1) > 1e-6))
    out = dict(problem=prob, n_eval=N, full_time_mean=float(TF.mean()), full_time_median=float(np.median(TF)),
               accuracy=float(np.mean([r["acc"] for r in R])), monotonicity_violations=mono,
               solvefree_lemma_violations=int(((S == 0) & (st == "optimal") & (np.nan_to_num(G) > GTOL)).sum()),
               curve=[dict(lam=l, frac_fixed=float(NF[:, i].mean()), fallback=float(fb[:, i].mean()), surrogate=float(S[:, i].mean()),
                           P_gap_gt0=float(P0[:, i].mean()), P_gap_gt1pct=float(P1[:, i].mean()), mean_gap=float(LM[:, i].mean()),
                           speedup_median=float(np.median(spd[:, i])), speedup_total=float(TF.sum() / T[:, i].sum())) for i, l in enumerate(LAMS)])
    ps = [r["ps"] for r in R]; psT = np.array([TOV[i] + p["t_red"] + (TF[i] if p["status"] != "optimal" else 0) for i, p in enumerate(ps)])
    out["PS_baseline"] = dict(fallback=float(np.mean([p["status"] != "optimal" for p in ps])),
                              P_gap_gt1pct=float(np.mean([p["status"] == "optimal" and p["gap"] > EPS for p in ps])),
                              speedup_median=float(np.median(TF / psT)), speedup_total=float(TF.sum() / psT.sum()))
    rng = np.random.default_rng(0); res = {}
    methods = [("CRC_meangap", (0.005, 0.01, 0.02), lambda L, a: crc(env(L), a, BF), LM, LM),
               ("CRC_P(gap>1%)", (0.05, 0.10, 0.20), lambda L, a: crc(env(L), a), P1, P1),
               ("LTT_P(gap>1%)", (0.05, 0.10, 0.20), lambda L, a: ltt(env(L), a), P1, P1),
               ("solvefree_CRC_P(not optimal)", (0.05, 0.10, 0.20), lambda L, a: crc(L, a), S, P0),
               ("solvefree_LTT_P(not optimal)", (0.05, 0.10, 0.20), lambda L, a: ltt(L, a), S, P0)]
    for ncal in (25, 50, 100):
        if ncal > N - 20: continue
        splits = [rng.permutation(N) for _ in range(REPS)]
        for name, alphas, rule, Lcal, Leval in methods:
            for a in alphas:
                risk, above, sp, tot, nfb = [], [], [], [], 0
                for pm in splits:
                    cal, te = pm[:ncal], pm[ncal:]; i0 = rule(Lcal[cal], a)
                    i = fastest(i0, T[cal], TF[cal]) if (SELECT == "fastest" and not name.startswith("solvefree")) else i0   # solve-free: no calibration solve times available
                    if i is None:                       # nothing certified: solve the full problem (risk 0, speedup ~1)
                        nfb += 1; risk.append(0.0); above.append(0.0); sp.append(float(np.median(TF[te] / (TF[te] + TOV[te])))); tot.append(float(TF[te].sum() / (TF[te] + TOV[te]).sum())); continue
                    rk = float(Leval[te, i].mean()); risk.append(rk); above.append(float(rk > a))
                    sp.append(float(np.median(spd[te, i]))); tot.append(float(TF[te].sum() / T[te, i].sum()))
                res[f"{name}|n{ncal}|a{a}"] = dict(target=a, realized_risk=float(np.mean(risk)), frac_splits_above_target=float(np.mean(above)),
                    speedup_median=float(np.median(sp)), speedup_total=float(np.median(tot)), full_solve_fallback_rate=nfb / REPS,
                    calibration_reduced_solves=0 if name.startswith("solvefree") else ncal * len(LAMS))
    out["results"] = res; return out
if __name__ == "__main__":
    allr = {}
    for prob in ["setcover", "cauction", "facility", "indset"]:
        r = analyse(prob)
        if r: allr[prob] = r; print(prob, "| monotonicity violations:", r["monotonicity_violations"], "| solve-free lemma violations:", r["solvefree_lemma_violations"])
    pr = {p: pickle.load(open(os.path.join("preds", p, "preds.pkl"), "rb")) for p in allr}
    for p in allr: allr[p]["predictor"] = dict(used=pr[p]["use"], validation_scores=pr[p]["scores"])
    json.dump(allr, open(os.path.join("results", "summary_bench_%s.json" % SELECT), "w"), indent=1)
    import zipfile                                   # pack raw evaluation records so that results can be re-analysed without new solves
    with zipfile.ZipFile(os.path.join("results", "eval_records.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(os.path.join("eval", "*", "*.pkl")) + glob.glob(os.path.join("preds", "*", "preds.pkl")): z.write(f); print("analysis done -> results/summary_bench.json")
