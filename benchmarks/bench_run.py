# CRISP benchmarks driver. Resumable. Usage: python bench_run.py [problems...]
import numpy as np, pickle, os, sys, time, glob, json
from multiprocessing import Pool, cpu_count
from bench_work import gen_work, eval_work
N_TRAIN, N_EVAL = int(os.environ.get("CRISP_NTRAIN", 200)), int(os.environ.get("CRISP_NEVAL", 160))
def log(msg):
    print(msg, flush=True)
    with open(os.path.join("results", "progress.txt"), "a", encoding="utf-8") as f: f.write(time.strftime("%H:%M:%S ") + msg + "\n")
if __name__ == "__main__":
    import multiprocessing as _mp; _mp.set_start_method("spawn", force=True)
    probs = sys.argv[1:] or ["cauction", "setcover", "indset", "facility"]
    os.makedirs("results", exist_ok=True); nw = max(1, min(int(os.environ.get("CRISP_WORKERS", cpu_count() - 1)), 6))
    for prob in probs:
        for d in ("data", "eval", "preds"): os.makedirs(os.path.join(d, prob), exist_ok=True)
        # 1) instances + exact solutions
        jobs = [(prob, k) for k in range(N_TRAIN + N_EVAL)]; t0 = time.time()
        log(f"[{prob}] stage 1: {len(jobs)} instances, {nw} workers")
        with Pool(nw) as pool:
            for i, _ in enumerate(pool.imap_unordered(gen_work, jobs)):
                if i % 40 == 0: log(f"[{prob}]   {i}/{len(jobs)}  {(time.time()-t0)/60:.1f} min")
        files = sorted(glob.glob(os.path.join("data", prob, "*.pkl")))
        ok = [f for f in files if pickle.load(open(f, "rb"))["status"] == "optimal"]
        log(f"[{prob}] solved to optimality: {len(ok)}/{len(files)}")
        train, ev = ok[:N_TRAIN], ok[N_TRAIN:N_TRAIN + N_EVAL]
        # 2) predictor (GBM vs GNN, chosen with the solve-free surrogate on validation data)
        pf = os.path.join("preds", prob, "preds.pkl")
        if not os.path.exists(pf):
            from bench_gnn import train_predict
            use, scores, preds = train_predict(prob, train, ev, log)
            pickle.dump(dict(use=use, scores=scores, preds=preds), open(pf, "wb"))
        pr = pickle.load(open(pf, "rb"))
        # 3) evaluation solves (hard fixing on the lambda grid + PS baseline)
        jobs = [(prob, name, p, t) for name, p, t in pr["preds"]]; t0 = time.time()
        log(f"[{prob}] stage 3: evaluation solves on {len(jobs)} instances (predictor {pr['use']})")
        with Pool(nw) as pool:
            for i, _ in enumerate(pool.imap_unordered(eval_work, jobs)):
                if i % 20 == 0: log(f"[{prob}]   {i}/{len(jobs)}  {(time.time()-t0)/60:.1f} min")
        log(f"[{prob}] done")
    log("ALL PROBLEMS DONE")
