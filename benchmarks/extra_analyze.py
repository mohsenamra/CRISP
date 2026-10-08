# Analysis of the supplementary experiments (distribution shift, size transfer, second solver).
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pickle, os, glob, json, zipfile
from bench_work import LAMS
from bench_analyze import env, crc, ltt, fastest, REPS, EPS, BF, GTOL
from extra_work import LX
PROBS = ["setcover", "cauction", "facility", "indset"]
ALPHA = {"CRC_meangap": (0.005, 0.01, 0.02), "CRC_P(gap>1%)": (0.05, 0.10, 0.20), "LTT_P(gap>1%)": (0.05, 0.10, 0.20),
         "solvefree_CRC_P(not optimal)": (0.05, 0.10, 0.20), "solvefree_LTT_P(not optimal)": (0.05, 0.10, 0.20)}
RULE = {"CRC_meangap": lambda L, a: crc(env(L), a, BF), "CRC_P(gap>1%)": lambda L, a: crc(env(L), a), "LTT_P(gap>1%)": lambda L, a: ltt(env(L), a),
        "solvefree_CRC_P(not optimal)": lambda L, a: crc(L, a), "solvefree_LTT_P(not optimal)": lambda L, a: ltt(L, a)}

def load(pattern): return [pickle.load(open(f, "rb")) for f in sorted(glob.glob(pattern))]

def mats(R, grid):
    TF = np.array([r["t_full"] for r in R]); TOV = np.array([r["t_lp"] + r["t_pred"] for r in R])
    st = np.array([[r["rows"][l]["status"] for l in grid] for r in R])
    G = np.array([[np.nan if r["rows"][l]["gap"] is None else r["rows"][l]["gap"] for l in grid] for r in R])
    S = np.array([[float(r["rows"][l]["wrong"] > 0) for l in grid] for r in R]); fb = st != "optimal"
    T = TOV[:, None] + np.array([[r["rows"][l]["t_red"] for l in grid] for r in R]) + np.where(fb, TF[:, None], 0)
    P0 = np.where(fb, 0.0, np.nan_to_num(G) > GTOL); P1 = np.where(fb, 0.0, np.nan_to_num(G) > EPS); LM = np.where(fb, 0.0, np.minimum(np.nan_to_num(G), BF))
    Gm = np.where(st == "infeasible", np.inf, np.where(st == "optimal", G, np.nan))
    with np.errstate(invalid="ignore"): mono = int(np.nansum(np.diff(Gm, axis=1) > 1e-6))
    lemma = int(((S == 0) & (st == "optimal") & (np.nan_to_num(G) > GTOL)).sum())
    NF = np.array([[r["rows"][l]["nfix"] / r["n_int"] for l in grid] for r in R])
    return dict(N=len(R), TF=TF, TOV=TOV, T=T, spd=TF[:, None] / T, fb=fb, S=S, P0=P0, P1=P1, LM=LM, mono=mono, lemma=lemma, NF=NF,
                acc=float(np.mean([r["acc"] for r in R])) if "acc" in R[0] else None)
LOSS = {"CRC_meangap": ("LM", "LM"), "CRC_P(gap>1%)": ("P1", "P1"), "LTT_P(gap>1%)": ("P1", "P1"),
        "solvefree_CRC_P(not optimal)": ("S", "P0"), "solvefree_LTT_P(not optimal)": ("S", "P0")}

def evaluate(C, Tt, name, a, cal_idx, te_idx):
    """Calibrate on rows cal_idx of C (dict of matrices), test on rows te_idx of Tt. Returns (risk, speedup_median, speedup_total, fell_back)."""
    lc, le = LOSS[name]; i0 = RULE[name](C[lc][cal_idx], a); i = i0 if name.startswith("solvefree") else fastest(i0, C["T"][cal_idx], C["TF"][cal_idx])   # solve-free uses no calibration solves
    if i is None:
        tf, tov = Tt["TF"][te_idx], Tt["TOV"][te_idx]; return 0.0, float(np.median(tf / (tf + tov))), float(tf.sum() / (tf + tov).sum()), 1
    return float(Tt[le][te_idx, i].mean()), float(np.median(Tt["spd"][te_idx, i])), float(Tt["TF"][te_idx].sum() / Tt["T"][te_idx, i].sum()), 0

def summarize(rows, a):
    r = np.array(rows); return dict(target=a, realized_risk=float(r[:, 0].mean()), frac_splits_above_target=float((r[:, 0] > a).mean()),
                                    speedup_median=float(np.median(r[:, 1])), speedup_total=float(np.median(r[:, 2])), full_solve_fallback_rate=float(r[:, 3].mean()))

def transfer(src, tgt, ncal_src=100, ncal_tgt=(25, 50)):
    """naive: calibrate on the source distribution, deploy on the target; recalibrated: calibrate on a few target instances."""
    rng = np.random.default_rng(0); out = {}
    for name, alphas in ALPHA.items():
        for a in alphas:
            rows = []
            for _ in range(REPS):
                cal = rng.permutation(src["N"])[:ncal_src]; rows.append(evaluate(src, tgt, name, a, cal, np.arange(tgt["N"])))
            out[f"{name}|naive_n{ncal_src}src|a{a}"] = summarize(rows, a)
            for n in ncal_tgt:
                if n > tgt["N"] - 20: continue
                rows = []
                for _ in range(REPS):
                    pm = rng.permutation(tgt["N"]); rows.append(evaluate(tgt, tgt, name, a, pm[:n], pm[n:]))
                out[f"{name}|recal_n{n}tgt|a{a}"] = summarize(rows, a)
    return out

def describe(M, grid):
    return dict(n=M["N"], full_time_mean=float(M["TF"].mean()), full_time_median=float(np.median(M["TF"])), accuracy=M["acc"],
                monotonicity_violations=M["mono"], solvefree_lemma_violations=M["lemma"],
                curve=[dict(lam=l, frac_fixed=float(M["NF"][:, i].mean()), fallback=float(M["fb"][:, i].mean()), surrogate=float(M["S"][:, i].mean()),
                            P_gap_gt0=float(M["P0"][:, i].mean()), P_gap_gt1pct=float(M["P1"][:, i].mean()), mean_gap=float(M["LM"][:, i].mean()),
                            speedup_median=float(np.median(M["spd"][:, i]))) for i, l in enumerate(grid)])

if __name__ == "__main__":
    allr = {}
    for prob in PROBS:
        res = {}
        ind_dir = os.path.join("eval_x", "indist", prob, "*.pkl") if glob.glob(os.path.join("eval_x", "indist", prob, "*.pkl")) else os.path.join("eval", prob, "*.pkl")
        Rin = load(ind_dir)
        for exp, grid in (("shift", LAMS), ("large", LX)):
            Rt = load(os.path.join("eval_x", exp, prob, "*.pkl"))
            if len(Rt) < 40: continue
            src, tgt = mats(Rin, grid), mats(Rt, grid)
            res[exp] = dict(source=describe(src, grid), target=describe(tgt, grid), results=transfer(src, tgt))
            print(prob, exp, "n =", tgt["N"], "| monotonicity violations:", tgt["mono"], "| solve-free lemma violations:", tgt["lemma"])
        Rs = [r for r in load(os.path.join("eval_x", "scip", prob, "*.pkl")) if r["status_full"] == "optimal" and r["rows"]]
        if len(Rs) >= 40:
            for r in Rs: r["acc"] = 0.0
            Ms = mats(Rs, LX)
            # solver independence of the loss: compare SCIP and HiGHS gaps at every lambda where both proved optimality
            names = sorted(os.path.basename(f) for f in glob.glob(os.path.join("eval_x", "scip", prob, "*.pkl")))
            Hd = {n: pickle.load(open(os.path.join("eval", prob, n), "rb")) for n in names}
            Sd = {n: pickle.load(open(os.path.join("eval_x", "scip", prob, n), "rb")) for n in names}
            diffs, zdiff = [], []
            for n in names:
                s, h = Sd[n], Hd[n]
                if s["status_full"] != "optimal": continue
                zdiff.append(abs(s["z"] - s["z_highs"]) / max(abs(s["z_highs"]), 1e-9))
                for l in LX:
                    a, b = s["rows"][l], h["rows"][l]
                    if a["status"] == "optimal" and b["status"] == "optimal": diffs.append(abs(a["gap"] - b["gap"]))
            rng = np.random.default_rng(0); sres = {}
            for name, alphas in ALPHA.items():
                for a in alphas:
                    for n in (25, 50):
                        if n > Ms["N"] - 20: continue
                        rows = []
                        for _ in range(REPS):
                            pm = rng.permutation(Ms["N"]); rows.append(evaluate(Ms, Ms, name, a, pm[:n], pm[n:]))
                        sres[f"{name}|n{n}|a{a}"] = summarize(rows, a)
            res["scip"] = dict(n=Ms["N"], full_time_mean_scip=float(Ms["TF"].mean()), full_time_mean_highs_same_instances=float(np.mean([Sd[n]["t_full_highs"] for n in names if Sd[n]["status_full"] == "optimal"])),
                               max_rel_objective_difference_scip_vs_highs=float(max(zdiff)) if zdiff else None,
                               max_gap_difference_scip_vs_highs=float(max(diffs)) if diffs else None, n_gap_pairs=len(diffs),
                               monotonicity_violations=Ms["mono"], solvefree_lemma_violations=Ms["lemma"], results=sres)
            print(prob, "scip n =", Ms["N"], "| max gap difference SCIP vs HiGHS:", res["scip"]["max_gap_difference_scip_vs_highs"])
        if res: allr[prob] = res
    json.dump(allr, open(os.path.join("results", "summary_extra.json"), "w"), indent=1)
    with zipfile.ZipFile(os.path.join("results", "extra_records.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(os.path.join("eval_x", "*", "*", "*.pkl")): z.write(f)
        for f in []:               # small metadata only (no matrices)
            r = pickle.load(open(f, "rb")); z.writestr(f.replace("data_x", "meta_x") , pickle.dumps(dict(z=r["z"], t_full=r["t_full"], status=r["status"], t_lp=r["t_lp"])))
    print("analysis done -> results/summary_extra.json, results/extra_records.zip")
