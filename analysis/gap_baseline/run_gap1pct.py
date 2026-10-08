# Baseline: HiGHS with a 1% relative MIP gap (deterministic per-instance bound), versus the exact full solve, same machine.
import numpy as np, time, json, os, sys, highspy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "..", "..", "benchmarks"), os.path.join(HERE, "..", "..", "application")]
from multiprocessing import Pool
import bench_lib
from bench_lib import GENERATORS
from bench_work import SIZES
def solve_gap(P, gap):
    import scipy.sparse as sp
    A=P["A"]; m,n=A.shape; h=highspy.Highs(); h.setOptionValue("output_flag",False); h.setOptionValue("threads",1)
    h.setOptionValue("time_limit",1200.0); h.setOptionValue("mip_rel_gap",float(gap))
    lp=highspy.HighsLp(); lp.num_col_,lp.num_row_=n,m
    lp.col_cost_,lp.col_lower_,lp.col_upper_,lp.row_lower_,lp.row_upper_=P["c"],P["lb"],P["ub"],P["lo"],P["up"]
    S=A.tocsc(); lp.a_matrix_.format_=highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_,lp.a_matrix_.index_,lp.a_matrix_.value_=S.indptr,S.indices,S.data.astype(float)
    lp.integrality_=[highspy.HighsVarType.kInteger if b else highspy.HighsVarType.kContinuous for b in P["integ"]]
    h.passModel(lp); t=time.time(); h.run(); t=time.time()-t
    return h.getInfo().objective_function_value, t, str(h.getModelStatus())
def job(arg):
    prob,k=arg; f=f"out/{prob}_{k:04d}.json"
    if os.path.exists(f): return
    if prob=="ssoa":
        from ssoa_gen import gen_ssoa
        yr=1397+k//52
        for att in range(50):
            rng=np.random.default_rng(9_000_000+100*k+att); P=gen_ssoa(rng,year=yr)
            if bench_lib.solve(P,integer=False)[4]=="optimal": break
    else:
        rng=np.random.default_rng(100_000*(1+list(GENERATORS).index(prob))+k); P=GENERATORS[prob](rng,**SIZES[prob])
    z0,t0,s0=solve_gap(P,1e-6); z1,t1,s1=solve_gap(P,0.01)
    json.dump(dict(z_exact=z0,t_exact=t0,st_exact=s0,z_gap1=z1,t_gap1=t1,st_gap1=s1,true_gap=(z1-z0)/abs(z0)),open(f,"w"))
if __name__=="__main__":
    os.makedirs("out",exist_ok=True)
    jobs=[(p,k) for p in ["cauction","facility","setcover","indset"] for k in range(200,250)]+[("ssoa",k) for k in range(208,248)]
    with Pool(2) as pool:
        for i,_ in enumerate(pool.imap_unordered(job,jobs)): print(i,flush=True)
    print("DONE",flush=True)
