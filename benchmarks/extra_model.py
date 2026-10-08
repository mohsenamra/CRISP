# Re-creates the predictor chosen in the main benchmark run (same data, same seeds, same code path as bench_gnn.train_predict),
# saves it under models\, and checks that it reproduces the stored predictions on the standard evaluation instances.
import numpy as np, pickle, os, time
from multiprocessing import cpu_count

def _train(prob, use, train_files, log):
    import torch; torch.manual_seed(0); torch.set_num_threads(max(1, cpu_count() - 1))
    from bench_gnn import graph, build
    recs = [pickle.load(open(f, "rb")) for f in train_files]; nval = 25; tr, va = recs[:-nval], recs[-nval:]
    ints = lambda r: r["P"]["integ"]
    if use == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(
            np.vstack([r["F"][ints(r)] for r in tr]), np.concatenate([r["x"][ints(r)] for r in tr]))
        return dict(kind="gbm", gbm=gbm)
    gtr, gva = [graph(r) for r in tr], [graph(r) for r in va]
    pos = float(np.mean(np.concatenate([g["y"].numpy() for g in gtr])))
    dv, dc = gtr[0]["V"].shape[1], gtr[0]["C"].shape[1]
    model = build(dv, dc); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(10.0, (1 - pos) / max(pos, 1e-3))))
    best, state, wait, t0 = 1e9, None, 0, time.time()
    for ep in range(int(os.environ.get("CRISP_EPOCHS", 150))):
        model.train()
        for k in np.random.default_rng(ep).permutation(len(gtr)):
            opt.zero_grad(); lossf(model(gtr[k]), gtr[k]["y"]).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = float(np.mean([torch.nn.functional.binary_cross_entropy_with_logits(model(g), g["y"]).item() for g in gva]))
        if vl < best - 1e-4: best, state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else: wait += 1
        if ep % 10 == 0: log(f"  [{prob}] GNN epoch {ep} val loss {vl:.4f} ({(time.time()-t0)/60:.1f} min)")
        if wait >= 20: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        lv = torch.cat([model(g) for g in gva]); yv = torch.cat([g["y"] for g in gva]); Ts = np.linspace(0.3, 3, 28)
        T = float(Ts[int(np.argmin([torch.nn.functional.binary_cross_entropy_with_logits(lv / t, yv).item() for t in Ts]))])
    return dict(kind="gnn", state=state, T=T, dv=dv, dc=dc)

def get_model(prob, train_files, log):
    os.makedirs("models", exist_ok=True); f = os.path.join("models", prob + ".pkl")
    if os.path.exists(f): return pickle.load(open(f, "rb"))
    use = pickle.load(open(os.path.join("preds", prob, "preds.pkl"), "rb"))["use"]
    log(f"[{prob}] re-training the {use.upper()} predictor of the main run (same seeds)"); M = _train(prob, use, train_files, log)
    pickle.dump(M, open(f, "wb")); return M

def predict(M, rec):
    t0 = time.time()
    if M["kind"] == "gbm": p = M["gbm"].predict_proba(rec["F"][rec["P"]["integ"]])[:, 1]
    else:
        import torch
        from bench_gnn import graph, build
        if "_net" not in M:
            net = build(M["dv"], M["dc"]); net.load_state_dict(M["state"]); net.eval(); M["_net"] = net
        rec = dict(rec)
        if rec.get("x") is None: rec["x"] = np.zeros(len(rec["P"]["c"]))
        with torch.no_grad(): p = torch.sigmoid(M["_net"](graph(rec)) / M["T"]).numpy()
    return p.astype(float), time.time() - t0

def check_reproduction(prob, M, log):
    """Max |p_new - p_stored| on the standard evaluation instances (0 means the stored main-run records stay valid)."""
    pr = pickle.load(open(os.path.join("preds", prob, "preds.pkl"), "rb"))["preds"]; d = 0.0
    for name, p, _ in pr[:40]:
        q, _ = predict(M, pickle.load(open(os.path.join("data", prob, name), "rb"))); d = max(d, float(np.abs(q - p).max()))
    log(f"[{prob}] reproduction check: max |p_new - p_main| = {d:.2e}"); return d
