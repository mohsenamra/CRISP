# CRISP application: multi-product, multi-period supplier selection and order allocation (SSOA) with all-unit quantity
# discounts, ordering set-ups, a shared warehouse and a distress-risk budget. Instances are built from real Codal data:
# supplier pools of Iranian listed manufacturers in decision year t (financial statements of year t), out-of-sample
# distress probabilities (calibrated LightGBM, rolling origin) and capacities from real revenue.
import numpy as np, pandas as pd, os
from scipy.stats import rankdata
from bench_lib import _P, INF

HERE = os.path.dirname(os.path.abspath(__file__))
CLUSTER = {  # product family supplied by each industry (the buyer is a manufacturer buying four input families)
    "Chemical products": 0, "Rubber & plastics": 0, "Paper products": 0, "Refined petroleum": 0, "Textiles": 0,
    "Basic metals": 1, "Fabricated metal products": 1, "Metal ore mining": 1,
    "Automotive & parts": 2, "Machinery & equipment": 2, "Electrical machinery": 2,
    "Cement, lime & plaster": 3, "Other non-metallic minerals": 3, "Tiles & ceramics": 3}
_POOL = None

def pool():
    global _POOL
    if _POOL is None:
        d = pd.read_csv(os.path.join(HERE, "data", "supplier_pool.csv"))
        p = pd.read_csv(os.path.join(HERE, "data", "preds_IR.csv.gz")); p = p[p.model == "LGBM"][["firm", "year", "p_cal"]]
        m = d.merge(p, left_on=["symbol", "fy"], right_on=["firm", "year"], how="inner")
        m = m[m.industry_name.isin(CLUSTER) & m.revenue_t.gt(0)].copy()
        m["size"] = m.revenue_t / m.cpi
        _POOL = m.reset_index(drop=True)
    return _POOL

def _r01(x):
    x = np.asarray(x, float); x = np.where(np.isfinite(x), x, np.nanmedian(x)); return (rankdata(x) - 1) / max(len(x) - 1, 1)

def gen_ssoa(rng, n_sup=40, n_prod=4, n_per=6, tiers=(0.0, 0.25, 0.6), disc=(0.0, 0.04, 0.08), year=None,
             kappa_p=0.10, sigma_c=0.03, risk_budget=0.035, n_max_frac=0.45, wh_frac=0.9, return_meta=False):
    """One instance. Variables, in order: y_i (contract) | o_it (order set-up) | z_ijtk (tier) | x_ijtk (quantity) | I_jt (stock).
    Supplier i offers only the product family of its industry."""
    M = pool(); years = sorted(M.fy.unique())
    yr = year if year is not None else int(rng.choice(years))
    S = M[M.fy == yr]
    for _ in range(100):                                          # every product family must have >= 3 candidates
        sel = S.iloc[np.sort(rng.choice(len(S), n_sup, replace=False))]
        prod = sel.industry_name.map(CLUSTER).values % n_prod
        if all((prod == j).sum() >= 3 for j in range(n_prod)): break
    I, J, T, K = n_sup, n_prod, n_per, len(tiers)
    p = np.clip(sel.p_cal.values, 1e-4, 0.999)
    lev = np.clip(np.nan_to_num(sel.tl_ta_w.values, nan=0.6), 0, 1); mu = 0.2 + 0.6 * lev       # severity (as in Paper 2)
    r = _r01((_r01(sel.tl_ta_w.values) + _r01(-sel.roa_w.values) + _r01(-sel.gross_margin_w.values)) / 3)
    cbar = np.array([1.0, 1.4, 2.0, 0.7])[:J] * rng.uniform(0.9, 1.1, J)
    c = cbar[prod] * (1 - kappa_p * r + rng.normal(0, sigma_c, I))                              # price-risk rule
    dbase = rng.uniform(80, 120, J); season = 1 + 0.25 * np.sin(2 * np.pi * (np.arange(T) / T + rng.uniform()))
    d = np.maximum(dbase[:, None] * season[None, :] * rng.lognormal(0, 0.15, (J, T)), 1.0)          # demand d_jt
    sq = np.sqrt(sel["size"].values); cap = np.empty(I)
    for j in range(J):
        m = prod == j; cap[m] = np.clip(0.55 * sq[m] / np.median(sq[m]), 0.15, 1.2) * d[j].mean()  # capacity per period
    f = rng.uniform(150, 300, I) * cbar[prod]                    # contracting cost
    s = rng.uniform(20, 50, I) * cbar[prod]                      # ordering set-up per period
    hold = 0.02 * cbar                                           # holding cost per unit-period
    spot = 1.6 * cbar                                            # emergency unit cost if a supplier fails
    pen = p * mu * (spot[prod] - c)                              # expected disruption cost per unit ordered (risk-neutral)
    WH = wh_frac * (d.sum(0).max())                              # shared warehouse capacity (all products)
    # indexing
    ny, no, nz = I, I * T, I * T * K; nx = nz; ni = J * T; n = ny + no + nz + nx + ni
    Y = lambda i: i; O = lambda i, t: ny + i * T + t
    Z = lambda i, t, k: ny + no + (i * T + t) * K + k; X = lambda i, t, k: ny + no + nz + (i * T + t) * K + k
    INV = lambda j, t: ny + no + nz + nx + j * T + t
    cost = np.zeros(n); lb = np.zeros(n); ub = np.full(n, INF); integ = np.zeros(n, bool)
    for i in range(I):
        cost[Y(i)] = f[i]; ub[Y(i)] = 1; integ[Y(i)] = True
        for t in range(T):
            cost[O(i, t)] = s[i]; ub[O(i, t)] = 1; integ[O(i, t)] = True
            for k in range(K):
                cost[Z(i, t, k)] = 0; ub[Z(i, t, k)] = 1; integ[Z(i, t, k)] = True
                cost[X(i, t, k)] = c[i] * (1 - disc[k]) + pen[i]
    for j in range(J):
        for t in range(T): cost[INV(j, t)] = hold[j]
    R, C, V, lo, up = [], [], [], [], []
    def row(entries, l, u):
        r_ = len(lo)
        for col, val in entries: R.append(r_); C.append(col); V.append(val)
        lo.append(l); up.append(u)
    for j in range(J):                                           # inventory balance
        for t in range(T):
            e = [(X(i, t, k), 1.0) for i in range(I) if prod[i] == j for k in range(K)] + [(INV(j, t), -1.0)]
            if t > 0: e.append((INV(j, t - 1), 1.0))
            row(e, d[j, t], d[j, t])
    for i in range(I):
        j = prod[i]; per = d[j].mean()
        for t in range(T):
            row([(X(i, t, k), 1.0) for k in range(K)] + [(O(i, t), -cap[i])], -INF, 0.0)   # capacity if ordered
            row([(O(i, t), 1.0), (Y(i), -1.0)], -INF, 0.0)                                    # order only if contracted
            row([(Z(i, t, k), 1.0) for k in range(K)] + [(O(i, t), -1.0)], -INF, 0.0)          # one tier per order
            for k in range(K):                                                                # tier ranges (all-unit)
                lo_k = tiers[k] * per; hi_k = (tiers[k + 1] * per if k + 1 < K else cap[i])
                row([(X(i, t, k), 1.0), (Z(i, t, k), -max(hi_k, lo_k))], -INF, 0.0)
                if lo_k > 0: row([(X(i, t, k), 1.0), (Z(i, t, k), -lo_k)], 0.0, INF)
    for t in range(T): row([(INV(j, t), 1.0) for j in range(J)], -INF, WH)                     # shared warehouse
    row([(Y(i), 1.0) for i in range(I)], 2 * J, max(2 * J, int(n_max_frac * I)))               # number of suppliers
    for j in range(J): row([(Y(i), 1.0) for i in range(I) if prod[i] == j], 2, INF)             # dual sourcing per family
    tot = d.sum()                                                                               # distress-risk budget
    row([(X(i, t, k), p[i]) for i in range(I) for t in range(T) for k in range(K)], -INF, risk_budget * tot)
    for i in range(I):                                                                           # max 50% share per family
        j = prod[i]; row([(X(i, t, k), 1.0) for t in range(T) for k in range(K)], -INF, 0.5 * d[j].sum())
    import scipy.sparse as sp
    A = sp.csr_matrix((V, (R, C)), shape=(len(lo), n))
    P = _P(A, cost, lo, up, lb, ub, integ)
    if return_meta: P["meta"] = dict(year=yr, firms=list(sel.symbol), p=p, prod=prod)
    return P
