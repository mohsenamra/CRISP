# Re-analyses the application records (eval/ssoa, from application/results/ssoa_records.zip) with the same
# calibration code as the benchmarks; writes results/summary_ssoa_v2.json (Tables 11 and 13 of the paper).
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks"))
import bench_analyze
os.makedirs("results", exist_ok=True)
json.dump(bench_analyze.analyse("ssoa"), open(os.path.join("results", "summary_ssoa_v2.json"), "w"), indent=1)
print("written results/summary_ssoa_v2.json")
