# CRISP benchmarks - shared library: four standard MILP families (after Gasse et al. 2019), sparse solver wrapper, features.
import numpy as np, time, highspy
import scipy.sparse as sp
INF = highspy.kHighsInf

def _P(A, c, lo, up, lb, ub, integ):
    return dict(A=sp.csr_matrix(A), c=np.asarray(c, float), lo=np.asarray(lo, float), up=np.asarray(up, float),
                lb=np.asarray(lb, float), ub=np.asarray(ub, float), integ=np.asarray(integ, bool))

def gen_setcover(rng, nrows=500, ncols=1000, density=0.05, max_coef=100):
    """Balas & Ho (1980) set covering, as in Gasse et al. (2019): min c'x, Ax >= 1, x binary."""
    nnz = int(nrows * ncols * density)
    idx = np.empty(nnz, int); idx[:ncols] = rng.permutation(ncols)          # every column used at least once
    idx[ncols:] = rng.choice(ncols, nnz - ncols)                              # remaining entries uniformly
    rows = np.empty(nnz, int); per = nnz // nrows; rows[:per * nrows] = np.repeat(np.arange(nrows), per)
    rows[per * nrows:] = rng.choice(nrows, nnz - per * nrows)
    A = sp.coo_matrix((np.ones(nnz), (rows, idx)), shape=(nrows, ncols)).tocsr(); A.data[:] = 1.0; A.sum_duplicates(); A.data[:] = 1.0
    c = rng.integers(1, max_coef + 1, ncols)
    return _P(A, c, np.ones(nrows), np.full(nrows, INF), np.zeros(ncols), np.ones(ncols), np.ones(ncols, bool))

def gen_cauction(rng, n_items=100, n_bids=500, min_value=1, max_value=100, value_deviation=0.5, add_item_prob=0.65,
                 max_n_sub_bids=5, additivity=0.2, budget_factor=1.5, resale_factor=0.5):
    """Combinatorial auction, 'arbitrary relationships' (Leyton-Brown et al. 2000), following Gasse et al. (2019)."""
    def choose(weights, n):
        w = np.asarray(weights, float); w = w / w.sum(); return rng.choice(len(w), n, replace=False, p=w)
    common = min_value + (max_value - min_value) * rng.random(n_items)
    private = common + max_value * value_deviation * (2 * rng.random(n_items) - 1)
    compat = rng.random((n_items, n_items)); compat = np.triu(compat, 1); compat = compat + compat.T; compat /= compat.sum(1, keepdims=True)
    bids, groups = [], []
    while len(bids) < n_bids:
        priv = common + max_value * value_deviation * (2 * rng.random(n_items) - 1)
        w = priv - priv.min() + 1e-6
        bundle = [int(choose(w, 1)[0])]
        while rng.random() < add_item_prob and len(bundle) < n_items:
            pr = compat[bundle].mean(0) * w; pr[bundle] = 0
            if pr.sum() <= 0: break
            bundle.append(int(choose(pr, 1)[0]))
        b = np.array(sorted(bundle)); price = priv[b].sum() + len(b) ** (1 + additivity)
        if price < 0: continue
        sub = [(b, price)]; budget = budget_factor * price; min_resale = resale_factor * common[b].sum()
        for _ in range(max_n_sub_bids):
            s = [int(choose(np.ones(n_items) * 1.0 if False else compat[b].mean(0) + 1e-9, 1)[0])]
            while len(s) < len(b) and rng.random() < add_item_prob:
                pr = compat[s].mean(0) + 1e-9; pr[s] = 0; s.append(int(choose(pr, 1)[0]))
            s = np.array(sorted(set(s))); sp_ = priv[s].sum() + len(s) ** (1 + additivity)
            if sp_ < budget and common[s].sum() >= min_resale and not any(len(s) == len(x) and (s == x).all() for x, _ in sub):
                sub.append((s, sp_))
        take = sub[: max(1, n_bids - len(bids))]
        group = list(range(len(bids), len(bids) + len(take)))
        bids += take; groups.append(group)
    rows, cols = [], []
    for j, (b, _) in enumerate(bids): rows += list(b); cols += [j] * len(b)
    r = n_items
    for g in groups:                       # dummy item: a bidder's bids are mutually exclusive (XOR)
        if len(g) > 1: rows += [r] * len(g); cols += g; r += 1
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(r, len(bids)))
    n_items_tot = r
    p = np.array([pr for _, pr in bids])
    return _P(A, -p, np.full(n_items_tot, -INF), np.ones(n_items_tot), np.zeros(len(bids)), np.ones(len(bids)), np.ones(len(bids), bool))

def gen_facility(rng, n_customers=100, n_facilities=100, ratio=5.0):
    """Capacitated facility location (Cornuejols, Sridharan & Thizy 1991), as in Gasse et al. (2019).
    Variables: x_ij (continuous, share of customer i served by j) then y_j (binary)."""
    cx, cy, fx, fy = rng.random(n_customers), rng.random(n_customers), rng.random(n_facilities), rng.random(n_facilities)
    d = rng.integers(5, 36, n_customers).astype(float); cap = rng.integers(10, 161, n_facilities).astype(float)
    fixed = (rng.integers(100, 111, n_facilities) * np.sqrt(cap) + rng.integers(0, 91, n_facilities)).astype(float)
    cap = cap * ratio * d.sum() / cap.sum()
    trans = np.sqrt((cx[:, None] - fx[None]) ** 2 + (cy[:, None] - fy[None]) ** 2) * 10 * d[:, None]
    nI, nJ = n_customers, n_facilities; nx = nI * nJ; n = nx + nJ
    X = lambda i, j: i * nJ + j; Y = lambda j: nx + j
    R, C, V, lo, up = [], [], [], [], []; r = 0
    for i in range(nI):                                   # demand: sum_j x_ij = 1
        for j in range(nJ): R.append(r); C.append(X(i, j)); V.append(1.0)
        lo.append(1.0); up.append(1.0); r += 1
    for j in range(nJ):                                   # capacity: sum_i d_i x_ij - s_j y_j <= 0
        for i in range(nI): R.append(r); C.append(X(i, j)); V.append(d[i])
        R.append(r); C.append(Y(j)); V.append(-cap[j]); lo.append(-INF); up.append(0.0); r += 1
    for j in range(nJ): R.append(r); C.append(Y(j)); V.append(cap[j])   # total capacity >= total demand
    lo.append(d.sum()); up.append(INF); r += 1
    for i in range(nI):                                   # tightening: x_ij - y_j <= 0
        for j in range(nJ):
            R += [r, r]; C += [X(i, j), Y(j)]; V += [1.0, -1.0]; lo.append(-INF); up.append(0.0); r += 1
    A = sp.csr_matrix((V, (R, C)), shape=(r, n))
    c = np.concatenate([trans.ravel(), fixed])
    integ = np.zeros(n, bool); integ[nx:] = True
    return _P(A, c, lo, up, np.zeros(n), np.ones(n), integ)

def gen_indset(rng, n_nodes=500, affinity=4):
    """Maximum independent set on Barabasi-Albert graphs with greedy clique-partition constraints (Gasse et al. 2019)."""
    adj = [set() for _ in range(n_nodes)]
    for u in range(affinity + 1):
        for v in range(u): adj[u].add(v); adj[v].add(u)
    deg = np.array([len(a) for a in adj], float)
    for u in range(affinity + 1, n_nodes):
        p = deg[:u] / deg[:u].sum(); nb = rng.choice(u, affinity, replace=False, p=p)
        for v in nb: adj[u].add(int(v)); adj[int(v)].add(u); deg[v] += 1
        deg[u] = affinity
    remaining = [set(a) for a in adj]; cliques = []
    order = sorted(range(n_nodes), key=lambda v: -len(adj[v]))
    for u in order:
        while remaining[u]:
            cl = [u]
            for v in sorted(remaining[u], key=lambda v: -len(remaining[v])):
                if all(v in adj[w] for w in cl): cl.append(v)
            for a in cl:
                for b in cl:
                    if a != b: remaining[a].discard(b)
            cliques.append(cl)
    R, C = [], []
    for r, cl in enumerate(cliques): R += [r] * len(cl); C += cl
    A = sp.csr_matrix((np.ones(len(R)), (R, C)), shape=(len(cliques), n_nodes))
    return _P(A, -np.ones(n_nodes), np.full(len(cliques), -INF), np.ones(len(cliques)), np.zeros(n_nodes), np.ones(n_nodes), np.ones(n_nodes, bool))

GENERATORS = {"setcover": gen_setcover, "cauction": gen_cauction, "facility": gen_facility, "indset": gen_indset}

def solve(P, fix=None, integer=True, tlim=1e9, extra_row=None):
    """fix: dict {var: value}; extra_row: (coef_vector, lo, up) appended constraint (trust region)."""
    A = P["A"]; m, n = A.shape
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(tlim)); h.setOptionValue("mip_rel_gap", 1e-6)
    lb, ub = P["lb"].copy(), P["ub"].copy(); lo, up = P["lo"], P["up"]
    if fix:
        for j, v in fix.items(): lb[j] = ub[j] = v
    if extra_row is not None:
        A = sp.vstack([A, sp.csr_matrix(extra_row[0][None, :])]).tocsr(); lo = np.append(lo, extra_row[1]); up = np.append(up, extra_row[2]); m += 1
    lp = highspy.HighsLp(); lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_, lp.col_lower_, lp.col_upper_, lp.row_lower_, lp.row_upper_ = P["c"], lb, ub, lo, up
    S = A.tocsc(); lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = S.indptr, S.indices, S.data.astype(float)
    if integer: lp.integrality_ = [highspy.HighsVarType.kInteger if b else highspy.HighsVarType.kContinuous for b in P["integ"]]
    h.passModel(lp); t = time.time(); h.run(); t = time.time() - t
    st = h.getModelStatus()
    if st == highspy.HighsModelStatus.kInfeasible: return None, None, t, None, "infeasible"
    if st != highspy.HighsModelStatus.kOptimal: return None, None, t, None, "timeout"
    s = h.getSolution()
    return h.getInfo().objective_function_value, np.array(s.col_value), t, np.array(s.col_dual), "optimal"

def features(P):
    """Per-variable features (all variables; only integer ones are predicted)."""
    t0 = time.time(); zlp, xlp, _, rc, _ = solve(P, integer=False); tlp = time.time() - t0
    A, c = P["A"], P["c"]; deg = np.asarray((A != 0).sum(0)).ravel() + 1e-9; ac = np.abs(c)
    F = np.column_stack([ac / (ac.mean() + 1e-9), deg / deg.mean(), (ac / deg) / ((ac / deg).mean() + 1e-9), xlp,
                         rc / (ac.mean() + 1e-9), xlp > 0.999, xlp < 1e-6, np.abs(xlp - 0.5), np.sign(c), P["integ"].astype(float)])
    return F, tlp, zlp
