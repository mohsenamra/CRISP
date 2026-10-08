# CRISP application: SSOA instances built from real Codal data. Resumable. Run from this folder.
# Design: 52 instances per decision year 1397-1403 (364 in total). Predictor trained on 1397-1400, CRISP calibrated and
# tested on 1401-1403 (later years = deployment), so the predictor never sees the evaluation years.
import os, sys, time, glob, pickle, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "benchmarks")); sys.path.insert(0, HERE)
os.chdir(HERE)
import numpy as np
from multiprocessing import Pool, cpu_count
YEARS = list(range(1397, 1404)); PER_YEAR = int(os.environ.get("CRISP_PER_YEAR", 52)); TRAIN_YEARS = YEARS[:4]
NSUP = int(os.environ.get("CRISP_NSUP", 40))
def log(msg):
    print(msg, flush=True)
    with open(os.path.join("results", "progress_ssoa.txt"), "a", encoding="utf-8") as f: f.write(time.strftime("%d %H:%M:%S ") + msg + "\n")
def gen_app(k):
    from ssoa_gen import gen_ssoa
    from bench_lib import solve, features
    f = os.path.join("data_inst", "ssoa", f"{k:04d}.pkl")
    if os.path.exists(f): return
    yr = YEARS[k // PER_YEAR]
    for attempt in range(50):                       # redraw the pool if the instance is infeasible (rare)
        rng = np.random.default_rng(9_000_000 + 100 * k + attempt); P = gen_ssoa(rng, n_sup=NSUP, year=yr, return_meta=True)
        if solve(P, integer=False)[4] == "optimal": break
    F, tlp, zlp = features(P)
    z, x, t, _, st = solve(P, tlim=900)
    pickle.dump(dict(P=P, F=F, t_lp=tlp, zlp=zlp, z=z, x=None if x is None else np.round(x), t_full=t, status=st, year=yr), open(f, "wb"))
if __name__ == "__main__":
    import multiprocessing as _mp; _mp.set_start_method("spawn", force=True)
    for d in ("results", "data_inst/ssoa", "eval/ssoa", "preds/ssoa"): os.makedirs(d, exist_ok=True)
    nw = max(1, min(int(os.environ.get("CRISP_WORKERS", cpu_count() - 1)), 6)); N = PER_YEAR * len(YEARS); t0 = time.time()
    log(f"[ssoa] stage 1: {N} instances ({NSUP} suppliers, years {YEARS[0]}-{YEARS[-1]}), {nw} workers")
    with Pool(nw) as pool:
        for i, _ in enumerate(pool.imap_unordered(gen_app, range(N))):
            if i % 20 == 0: log(f"[ssoa]   {i}/{N}  {(time.time()-t0)/60:.1f} min")
    files = sorted(glob.glob(os.path.join("data_inst", "ssoa", "*.pkl"))); R = {f: pickle.load(open(f, "rb")) for f in files}
    ok = [f for f in files if R[f]["status"] == "optimal"]
    train = [f for f in ok if R[f]["year"] in TRAIN_YEARS]; ev = [f for f in ok if R[f]["year"] not in TRAIN_YEARS]
    log(f"[ssoa] solved to optimality: {len(ok)}/{len(files)}; train {len(train)}, evaluation {len(ev)}")
    del R
    # bench_work / bench_gnn read instances from data/<prob>/ : expose them there
    os.makedirs(os.path.join("data", "ssoa"), exist_ok=True)
    for f in ok:
        g = os.path.join("data", "ssoa", os.path.basename(f))
        if not os.path.exists(g):
            import shutil; shutil.copyfile(f, g)
    train = [os.path.join("data", "ssoa", os.path.basename(f)) for f in train]; ev = [os.path.join("data", "ssoa", os.path.basename(f)) for f in ev]
    pf = os.path.join("preds", "ssoa", "preds.pkl")
    if not os.path.exists(pf):
        from bench_gnn import train_predict
        use, scores, preds = train_predict("ssoa", train, ev, log); pickle.dump(dict(use=use, scores=scores, preds=preds), open(pf, "wb"))
    pr = pickle.load(open(pf, "rb")); from bench_work import eval_work
    jobs = [("ssoa", name, p, t) for name, p, t in pr["preds"]]; t0 = time.time()
    log(f"[ssoa] stage 3: reduced solves on {len(jobs)} evaluation instances (predictor {pr['use']})")
    with Pool(nw) as pool:
        for i, _ in enumerate(pool.imap_unordered(eval_work, jobs)):
            if i % 20 == 0: log(f"[ssoa]   {i}/{len(jobs)}  {(time.time()-t0)/60:.1f} min")
    log("[ssoa] analysis")
    import bench_analyze
    out = bench_analyze.analyse("ssoa"); out["predictor"] = dict(used=pr["use"], validation_scores=pr["scores"])
    yrs = {os.path.basename(f): pickle.load(open(f, "rb"))["year"] for f in ev}; out["eval_years"] = yrs
    json.dump(out, open(os.path.join("results", "summary_ssoa.json"), "w"), indent=1)
    import zipfile
    with zipfile.ZipFile(os.path.join("results", "ssoa_records.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(os.path.join("eval", "ssoa", "*.pkl")) + [pf]: z.write(f)
        z.writestr("eval_years.json", json.dumps(yrs))
    log("[ssoa] ALL DONE")
