# Bipartite GNN (Gasse et al. 2019 style) + GBM predictors; predictor chosen per problem by the SOLVE-FREE surrogate on validation data.
import numpy as np, pickle, os, time
from multiprocessing import cpu_count

def graph(r):
    import torch
    P = r["P"]; A = P["A"].tocoo(); m, n = A.shape
    rdeg = np.asarray((P["A"] != 0).sum(1)).ravel(); cdeg = np.asarray((P["A"] != 0).sum(0)).ravel()
    lo, up = P["lo"], P["up"]; fin_lo, fin_up = np.isfinite(lo), np.isfinite(up)
    C = np.column_stack([fin_lo, fin_up, np.where(fin_lo, lo, 0) / (np.abs(lo[fin_lo]).mean() + 1e-9) if fin_lo.any() else np.zeros(m),
                         np.where(fin_up, up, 0) / (np.abs(up[fin_up]).mean() + 1e-9) if fin_up.any() else np.zeros(m), rdeg / rdeg.mean()]).astype(np.float32)
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long); val = A.data.astype(np.float32)
    Acv = torch.sparse_coo_tensor(idx, torch.tensor((val / np.maximum(rdeg[A.row], 1)).astype(np.float32)), (m, n)).coalesce()
    Avc = torch.sparse_coo_tensor(idx.flip(0), torch.tensor((val / np.maximum(cdeg[A.col], 1)).astype(np.float32)), (n, m)).coalesce()
    ints = np.where(P["integ"])[0]
    return dict(C=torch.tensor(C), V=torch.tensor(r["F"], dtype=torch.float32), Acv=Acv, Avc=Avc,
                ints=torch.tensor(ints), y=torch.tensor(r["x"][ints], dtype=torch.float32))

def build(dv, dc, h=64, layers=3):
    import torch, torch.nn as nn
    class BGNN(nn.Module):
        def __init__(s):
            super().__init__()
            s.ev = nn.Sequential(nn.Linear(dv, h), nn.ReLU(), nn.Linear(h, h)); s.ec = nn.Sequential(nn.Linear(dc, h), nn.ReLU(), nn.Linear(h, h))
            s.vc = nn.ModuleList([nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.LayerNorm(h)) for _ in range(layers)])
            s.cv = nn.ModuleList([nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.LayerNorm(h)) for _ in range(layers)])
            s.out = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        def forward(s, g):
            v, c = s.ev(g["V"]), s.ec(g["C"])
            for L1, L2 in zip(s.vc, s.cv):
                c = L1(torch.cat([c, torch.sparse.mm(g["Acv"], v)], 1)); v = L2(torch.cat([v, torch.sparse.mm(g["Avc"], c)], 1))
            return s.out(v).squeeze(1)[g["ints"]]
    return BGNN()

def surrogate_score(preds, recs, max_rate=0.10):
    """Largest mean fraction of fixed integer variables at a threshold whose solve-free surrogate rate <= max_rate (higher = better)."""
    best = 0.0
    for lam in np.linspace(0.5, 0.99999, 400):
        rate = np.mean([np.any(((p >= .5).astype(int) != r["x"][r["P"]["integ"]])[np.maximum(p, 1 - p) >= lam]) for p, r in zip(preds, recs)])
        frac = np.mean([np.mean(np.maximum(p, 1 - p) >= lam) for p in preds])
        if rate <= max_rate: best = max(best, frac)
    return best

def train_predict(prob, train_files, eval_files, log=print):
    import torch; torch.manual_seed(0); torch.set_num_threads(max(1, cpu_count() - 1))
    from sklearn.ensemble import HistGradientBoostingClassifier
    recs = [pickle.load(open(f, "rb")) for f in train_files]; nval = 25; tr, va = recs[:-nval], recs[-nval:]
    ints = lambda r: r["P"]["integ"]
    # GBM
    t0 = time.time()
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(
        np.vstack([r["F"][ints(r)] for r in tr]), np.concatenate([r["x"][ints(r)] for r in tr]))
    s_gbm = surrogate_score([gbm.predict_proba(r["F"][ints(r)])[:, 1] for r in va], va)
    log(f"  [{prob}] GBM trained ({(time.time()-t0)/60:.1f} min); validation fixable fraction at surrogate<=10%: {s_gbm:.3f}")
    # GNN
    gtr, gva = [graph(r) for r in tr], [graph(r) for r in va]
    pos = float(np.mean(np.concatenate([g["y"].numpy() for g in gtr])))
    model = build(gtr[0]["V"].shape[1], gtr[0]["C"].shape[1]); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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
        T = Ts[int(np.argmin([torch.nn.functional.binary_cross_entropy_with_logits(lv / t, yv).item() for t in Ts]))]
        s_gnn = surrogate_score([torch.sigmoid(model(g) / T).numpy() for g in gva], va)
    log(f"  [{prob}] GNN trained; validation fixable fraction at surrogate<=10%: {s_gnn:.3f}")
    use = "gnn" if s_gnn > s_gbm else "gbm"; log(f"  [{prob}] predictor chosen (solve-free, no extra solves): {use}")
    out = []
    for f in eval_files:
        r = pickle.load(open(f, "rb")); t0 = time.time()
        if use == "gbm": p = gbm.predict_proba(r["F"][ints(r)])[:, 1]
        else:
            g = graph(r)
            with torch.no_grad(): p = torch.sigmoid(model(g) / T).numpy()
        out.append((os.path.basename(f), p, time.time() - t0))
    return use, dict(gbm=s_gbm, gnn=s_gnn), out
