#!/usr/bin/env python3
"""Run on ONE node: Popen n_local worker_eval procs (1 GPU each), time the slowest.
Reuses the proven single-node subprocess-pool parallelism."""
import os, sys, time, subprocess
n_local = int(sys.argv[1]) if len(sys.argv) > 1 else 8
repo = os.path.expanduser("~/sqmg_project-cudaq")
wpath = os.path.join(repo, "bench_w.npy")
procs = []
t0 = time.time()
for g in range(n_local):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(g)
    p = subprocess.Popen(
        [sys.executable, os.path.join(repo, "worker_eval.py"),
         "--weight_path", wpath, "--result_path", f"/tmp/bench_r_{g}.npy",
         "--num_heavy_atom", "9", "--num_sample", "5000", "--backend", "cudaq_nvidia"],
        env=env, cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(p)
for p in procs:
    p.wait()
print(f"NODE {os.uname().nodename} n_local={n_local} node_round_time={time.time()-t0:.2f}", flush=True)
