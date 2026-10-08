# CRISP supplementary experiments driver. Resumable: re-running skips everything already finished.
# Usage: python extra_run.py [shift] [large] [scip]      (default: all three, in this order)
import numpy as np, pickle, os, sys, time, glob
from multiprocessing import Pool, cpu_count
from bench_work import LAMS
from extra_work import LX, gen_x, eval_x, scip_work
N_X = {"shift": int(os.environ.get("CRISP_NX_SHIFT", 120)), "large": int(os.environ.get("CRISP_NX_LARGE", 100))}; N_SCIP = int(os.environ.get("CRISP_NSCIP", 80))
PROBS = ["cauction", "facility", "setcover", "indset"]

def log(msg):
    print(msg, flush=True)
    with open(os.path.join("results", "progress_extra.txt"), "a", encoding="utf-8") as f: f.write(time.strftime("%d %H:%M:%S ") + msg + "\n")

def run_pool(fn, jobs, nw, tag, every=20):
    t0 = time.time()
    with Pool(nw) as pool:
        for i, _ in enumerate(pool.imap_unordered(fn, jobs)):
            if i % every == 0: log(f"{tag}   {i}/{len(jobs)}  {(time.time()-t0)/60:.1f} min")
    log(f"{tag}   {len(jobs)}/{len(jobs)}  {(time.time()-t0)/60:.1f} min")

def main_files(prob):
    files = sorted(glob.glob(os.path.join("data", prob, "*.pkl")))
    ok = [f for f in files if pickle.load(open(f, "rb"))["status"] == "optimal"]
    return ok[:int(os.environ.get("CRISP_NTRAIN", 200))]

if __name__ == "__main__":
    import multiprocessing as _mp; _mp.set_start_method("spawn", force=True)
    from extra_model import get_model, predict, check_reproduction
    exps = [a for a in sys.argv[1:] if a in ("shift", "large", "scip")] or ["shift", "large", "scip"]
    os.makedirs("results", exist_ok=True); nw = max(1, min(int(os.environ.get("CRISP_WORKERS", cpu_count() - 1)), 6))
    log(f"=== supplementary experiments {exps}, {nw} workers ===")
    for exp in exps:
        if exp == "scip":
            try: import pyscipopt  # noqa
            except ImportError: log("[scip] pyscipopt is not installed -> SCIP experiment skipped"); continue
        for prob in PROBS:
            tag = f"[{exp}|{prob}]"
            if exp == "scip":
                pr = pickle.load(open(os.path.join("preds", prob, "preds.pkl"), "rb"))["preds"][:N_SCIP]
                os.makedirs(os.path.join("eval_x", "scip", prob), exist_ok=True)
                jobs = [(os.path.join("data", prob, n), os.path.join("eval_x", "scip", prob, n), p, t, LX) for n, p, t in pr]
                log(f"{tag} SCIP full + reduced solves on {len(jobs)} standard instances"); run_pool(scip_work, jobs, nw, tag); continue
            # predictor of the main run (re-trained once with the same seeds, then cached in models\)
            M = get_model(prob, main_files(prob), log)
            if "repro" not in M:
                M["repro"] = check_reproduction(prob, M, log); M.pop("_net", None); pickle.dump(M, open(os.path.join("models", prob + ".pkl"), "wb"))
            if M["repro"] > 1e-4:                    # stored main-run records were made with slightly different scores: redo them
                d = os.path.join("eval_x", "indist", prob); os.makedirs(d, exist_ok=True)
                pr = pickle.load(open(os.path.join("preds", prob, "preds.pkl"), "rb"))["preds"]; jobs = []
                for n, _, _ in pr:
                    p, t = predict(M, pickle.load(open(os.path.join("data", prob, n), "rb"))); jobs.append((os.path.join("data", prob, n), os.path.join(d, n), p, t, LAMS))
                log(f"{tag} re-evaluating standard instances with the re-trained predictor"); run_pool(eval_x, jobs, nw, "[indist|" + prob + "]")
            # 1) new instances + exact solutions
            os.makedirs(os.path.join("data_x", exp, prob), exist_ok=True)
            log(f"{tag} stage 1: {N_X[exp]} instances"); run_pool(gen_x, [(exp, prob, k) for k in range(N_X[exp])], nw, tag)
            files = sorted(glob.glob(os.path.join("data_x", exp, prob, "*.pkl")))
            ok = [f for f in files if pickle.load(open(f, "rb"))["status"] == "optimal"]
            log(f"{tag} solved to optimality: {len(ok)}/{len(files)}")
            # 2) predictions with the frozen predictor (no re-training on the new distribution / size)
            d = os.path.join("eval_x", exp, prob); os.makedirs(d, exist_ok=True); jobs = []
            for f in ok:
                p, t = predict(M, pickle.load(open(f, "rb"))); jobs.append((f, os.path.join(d, os.path.basename(f)), p, t, LAMS if exp == "shift" else LX))
            # 3) reduced solves
            log(f"{tag} stage 3: reduced solves on {len(jobs)} instances"); run_pool(eval_x, jobs, nw, tag)
            log(f"{tag} done")
    log("ALL SUPPLEMENTARY EXPERIMENTS DONE")
