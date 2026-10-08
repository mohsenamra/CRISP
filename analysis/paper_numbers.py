# Derived numbers reported in the paper that are not in the summary files. Uses exactly the splits of bench_analyze.py.
import json, glob, pickle, numpy as np, sys
sys.argv = ["x", "--first"]
import bench_analyze as ba
from bench_work import LAMS
REPS = ba.REPS
def load(prob):
    R = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"eval/{prob}/*.pkl"))]
    TF = np.array([r["t_full"] for r in R]); TOV = np.array([r["t_lp"] + r["t_pred"] for r in R])
    st = np.array([[r["rows"][l]["status"] for l in LAMS] for r in R]); fb = st != "optimal"
    G = np.array([[np.nan if r["rows"][l]["gap"] is None else r["rows"][l]["gap"] for l in LAMS] for r in R]); G = np.where(fb, 0.0, np.nan_to_num(G))
    TR = np.array([[r["rows"][l]["t_red"] for l in LAMS] for r in R]); T = TOV[:, None] + TR + np.where(fb, TF[:, None], 0)
    NF = np.array([[r["rows"][l]["nfix"] for l in LAMS] for r in R])
    return dict(R=R, N=len(R), TF=TF, T=T, TR=TR, fb=fb, G=G, LM=np.minimum(G, ba.BF), P1=(G > ba.EPS).astype(float), spd=TF[:, None] / T, NF=NF)
def splits(N):
    rng = np.random.default_rng(0); out = {}
    for n in (25, 50, 100):
        if n > N - 20: continue
        out[n] = [rng.permutation(N) for _ in range(REPS)]
    return out
res = {}
for prob in ["setcover", "cauction", "facility", "indset", "ssoa"]:
    D = load(prob); N = D["N"]; S = splits(N); r = {}
    # 1) tuned (no correction) vs CRISP on identical splits, n = 100
    for lc, name, a in [("LM", "CRC_meangap", 0.01), ("P1", "LTT_P(gap>1%)", 0.1)]:
        tun, cri = [], []
        for pm in S[100]:
            c, t = pm[:100], pm[100:]
            ok = np.where(D[lc][c].mean(0) <= a)[0]; spc = np.median(D["TF"][c][:, None] / D["T"][c], axis=0); i = int(ok[np.argmax(spc[ok])])
            tun.append((D[lc][t, i].mean(), np.median(D["spd"][t, i])))
            i0 = ba.crc(ba.env(D[lc][c]), a, ba.BF) if lc == "LM" else ba.ltt(ba.env(D[lc][c]), a); j = ba.fastest(i0, D["T"][c], D["TF"][c])
            cri.append((D[lc][t, j].mean(), np.median(D["spd"][t, j])) if j is not None else (0.0, 1.0))
        tun, cri = np.array(tun), np.array(cri)
        r[f"tuned_vs_crisp|{name}|a{a}"] = dict(tuned=dict(risk=tun[:, 0].mean(), above=(tun[:, 0] > a).mean(), speedup=np.median(tun[:, 1])),
                                                 crisp=dict(risk=cri[:, 0].mean(), above=(cri[:, 0] > a).mean(), speedup=np.median(cri[:, 1])))
    # 2) fallback rate and truncation at the selected thresholds (CRC expected gap, n = 100, all three alphas)
    fbr, trunc_share, trunc_effect = [], [], []
    for pm in S[100]:
        c, t = pm[:100], pm[100:]
        for a in (0.005, 0.01, 0.02):
            i = ba.fastest(ba.crc(ba.env(D["LM"][c]), a, ba.BF), D["T"][c], D["TF"][c])
            if i is None: continue
            fbr.append(D["fb"][t, i].mean()); trunc_share.append((D["G"][t, i] > ba.BF).mean()); trunc_effect.append(D["G"][t, i].mean() - D["LM"][t, i].mean())
    r["fallback_rate_at_selected"] = float(np.mean(fbr)); r["share_gap_above_B_at_selected"] = float(np.mean(trunc_share)); r["mean_gap_effect_of_truncation"] = float(np.mean(trunc_effect))
    # 3) calibration cost: one pass over the grid, in full-solve equivalents per instance (grid points that fix nothing skipped)
    fixes = D["NF"] > 0
    r["calibration_cost_full_solve_equiv"] = float(np.mean((np.where(fixes, D["TR"], 0).sum(1)) / D["TF"]))
    r["calibration_cost_incl_fallback"] = float(np.mean((np.where(fixes, D["T"] - (D["T"] - D["TR"] - np.where(D["fb"], D["TF"][:, None], 0)), 0).sum(1)) / D["TF"]))
    res[prob] = r
# 4) first vs fastest rule (independent set, CRC expected gap 2%, n = 100)
D = load("indset"); S = splits(D["N"]); f1, f2 = [], []
for pm in S[100]:
    c, t = pm[:100], pm[100:]; i0 = ba.crc(ba.env(D["LM"][c]), 0.02, ba.BF); j = ba.fastest(i0, D["T"][c], D["TF"][c])
    f1.append(np.median(D["spd"][t, i0]) if i0 is not None else 1.0); f2.append(np.median(D["spd"][t, j]) if j is not None else 1.0)
res["indset"]["first_vs_fastest_CRC2pct"] = dict(first=float(np.median(f1)), fastest=float(np.median(f2)))
# 5) chronological test for the application (calibrate on 1401-1402, test on 1403)
yrs = json.load(open("results/eval_years.json")); D = load("ssoa")
import os
names = [os.path.basename(f) for f in sorted(glob.glob("eval/ssoa/*.pkl"))]; Y = np.array([yrs[n] for n in names]); c = np.where(Y < 1403)[0]; t = np.where(Y == 1403)[0]
chron = {}
for lc, name, a in [("LM", "CRC_meangap", 0.01), ("P1", "LTT_P(gap>1%)", 0.1)]:
    i0 = ba.crc(ba.env(D[lc][c]), a, ba.BF) if lc == "LM" else ba.ltt(ba.env(D[lc][c]), a); j = ba.fastest(i0, D["T"][c], D["TF"][c])
    chron[name] = dict(risk=float(D[lc][t, j].mean()), speedup_median=float(np.median(D["spd"][t, j])), speedup_total=float(D["TF"][t].sum() / D["T"][t, j].sum()), lam=LAMS[j])
res["ssoa"]["chronological"] = chron
def conv(o):
    if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
    return float(o) if isinstance(o, (np.floating, np.integer)) else o
json.dump(conv(res), open("results/paper_numbers.json", "w"), indent=1)
for p, r in res.items():
    print(p, {k: (v if not isinstance(v, dict) else {kk: (round(vv, 4) if isinstance(vv, float) else {k3: round(v3, 4) for k3, v3 in vv.items()} if isinstance(vv, dict) else vv) for kk, vv in v.items()}) for k, v in r.items()})
