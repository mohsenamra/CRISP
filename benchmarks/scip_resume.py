# Finishes the SCIP experiment with a per-instance watchdog: an instance that takes longer than CRISP_TASK_LIMIT seconds
# (wall clock) is stopped and recorded as "hung" instead of blocking the whole run. Usage: python scip_resume.py [problem ...]
import os, sys, time, pickle, multiprocessing as mp
from extra_work import scip_work, LX
N_SCIP = int(os.environ.get("CRISP_NSCIP", 80)); LIMIT = float(os.environ.get("CRISP_TASK_LIMIT", 900))
def log(msg):
    print(msg, flush=True)
    with open(os.path.join("results", "progress_extra.txt"), "a", encoding="utf-8") as f: f.write(time.strftime("%d %H:%M:%S ") + msg + "\n")
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    probs = sys.argv[1:] or ["indset"]; nw = max(1, min(int(os.environ.get("CRISP_WORKERS", mp.cpu_count() - 1)), 6))
    for prob in probs:
        d = os.path.join("eval_x", "scip", prob); os.makedirs(d, exist_ok=True)
        pr = pickle.load(open(os.path.join("preds", prob, "preds.pkl"), "rb"))["preds"][:N_SCIP]
        todo = [(os.path.join("data", prob, n), os.path.join(d, n), p, t, LX) for n, p, t in pr if not os.path.exists(os.path.join(d, n))]
        log(f"[scip-resume|{prob}] {len(todo)} instances left, watchdog {LIMIT:.0f} s per instance, {nw} workers")
        running, done, hung = [], 0, 0
        while todo or running:
            while todo and len(running) < nw:
                a = todo.pop(0); p = mp.Process(target=scip_work, args=(a,)); p.start(); running.append((p, a, time.time()))
            time.sleep(2)
            for item in running[:]:
                p, a, t0 = item
                if not p.is_alive():
                    p.join(); running.remove(item); done += 1
                elif time.time() - t0 > LIMIT:
                    p.terminate(); p.join(); running.remove(item); hung += 1
                    pickle.dump(dict(status_full="hung", rows={}, t_full=None, z=None), open(a[1], "wb"))
                    log(f"[scip-resume|{prob}]   {os.path.basename(a[1])} stopped after {LIMIT:.0f} s (recorded as hung)")
            if (done + hung) and (done + hung) % 10 == 0 and running == []: pass
        log(f"[scip-resume|{prob}] finished: {done} completed, {hung} hung")
    log("SCIP RESUME DONE")
