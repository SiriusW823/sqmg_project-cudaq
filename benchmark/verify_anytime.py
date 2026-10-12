"""benchmark/verify_anytime.py — 同硬體、同牆鐘比較的程式正確性檢查（合成目標，不碰 GPU）。
用法：cd <repo> && python benchmark/verify_anytime.py（需 numpy、scipy）。全部通過才可送實驗。"""
import csv, glob, logging, os, subprocess, sys, tempfile, time
import numpy as np
sys.path.insert(0, os.getcwd())
from optimizers import get_optimizer, REGISTRY
from evaluator import shot_seed_for, TEST_SEED_BASE, SHOT_SEED_STRIDE

log = logging.getLogger("t"); log.addHandler(logging.NullHandler()); log.setLevel(logging.CRITICAL)
ok = True
def check(label, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'✓' if cond else '✗'}] {label}{('  ' + extra) if extra else ''}", flush=True)

D, M = 12, 8
def batch_fn(delay):
    def f(X):
        time.sleep(delay)
        X = np.atleast_2d(X)
        v = np.clip(1.0 - np.abs(X - 0.3).mean(axis=1) * 2.0, 0, 1)
        u = np.clip(1.0 - np.abs(X - 0.7).mean(axis=1) * 2.0, 0, 1)
        return [(float(a), float(b)) for a, b in zip(v, u)]
    return f

def make(name, seed, d, max_evals, delay=0.0, **kw):
    cls = get_optimizer(name)
    return cls(n_params=D, max_evals=max_evals, batch_evaluate_fn=batch_fn(delay), logger=log,
               seed=seed, batch_size=M, data_dir=d, task_name=f"any_{name}_s{seed}",
               force_overwrite=True, **kw)

print("=== 1. 逐次取樣種子 ===")
s = [shot_seed_for(523, k) for k in range(20000)]
check("同一 run 內 20,000 個種子互不相同", len(set(s)) == len(s))
check("不同 base 的種子空間不重疊", max(shot_seed_for(500, k) for k in range(20000)) < shot_seed_for(501, 0))
check("最佳化種子 < 測試種子空間", shot_seed_for(1999, SHOT_SEED_STRIDE - 1) < TEST_SEED_BASE)
check("測試種子 < 2^31", TEST_SEED_BASE + 1000 < 2**31)
try:
    shot_seed_for(2000, 0); check("base 超界會報錯", False)
except ValueError:
    check("base 超界會報錯", True)

print("\n=== 2. 時間預算會停、檢查點語意正確 ===")
with tempfile.TemporaryDirectory() as d:
    from scipy.stats import qmc  # 先載入，避免首批延遲
    o = make("rr_qpso2", 3, d, max_evals=10**6, delay=0.1, alpha_max=1.2, alpha_min=0.3)
    o.time_budget_s = 3.0
    o.checkpoint_s = [1.0, 2.0, 3.0]
    t0 = time.time(); o.run(); dt = time.time() - t0
    check("在預算內停止（≤ 預算 + 一批）", dt <= 3.0 + 0.1 + 0.5, f"{dt:.2f}s")
    rows = list(csv.DictReader(open(os.path.join(d, "any_rr_qpso2_s3.csv"))))
    el = np.array([float(r["elapsed_s"]) for r in rows]); fit = np.array([float(r["fitness"]) for r in rows])
    for h in (1.0, 2.0, 3.0):
        z = np.load(os.path.join(d, f"any_rr_qpso2_s3_ckpt_{h/3600:g}h.npz"))
        expect = fit[el <= h].max() if (el <= h).any() else -np.inf
        check(f"t={h}s 的 incumbent = 該時刻前已完成評估的最佳值",
              np.isclose(float(z["f"]), expect, atol=1e-6) or (expect == -np.inf and np.isnan(z["x"]).all()),
              f"ckpt={float(z['f']):.6f} expect={expect:.6f} n={int(z['n_evals'])}")
    ns = [int(np.load(os.path.join(d, f"any_rr_qpso2_s3_ckpt_{h/3600:g}h.npz"))["n_evals"]) for h in (1.0, 2.0, 3.0)]
    check("檢查點的 n_evals 嚴格遞增且 > 0", ns[0] > 0 and ns[0] < ns[1] < ns[2], str(ns))

print("\n=== 3. 預算先用完（評估數）→ 後續檢查點以最終最佳解補上 ===")
with tempfile.TemporaryDirectory() as d:
    o = make("qpso", 4, d, max_evals=48, delay=0.0, alpha_max=1.2, alpha_min=0.3)
    o.time_budget_s = 100.0; o.checkpoint_s = [50.0, 100.0]
    o.run()
    z = np.load(os.path.join(d, f"any_qpso_s4_ckpt_{100/3600:g}h.npz"))
    check("carried=True 且 f = 最終最佳", bool(z["carried"]) and np.isclose(float(z["f"]), o.best_f),
          f"{float(z['f']):.6f} vs {o.best_f:.6f}")

print("\n=== 4. 時間預算的 run 不可續跑 ===")
with tempfile.TemporaryDirectory() as d:
    o = make("qpso", 5, d, max_evals=32, alpha_max=1.2, alpha_min=0.3); o.run()
    cls = get_optimizer("qpso")
    o2 = cls(n_params=D, max_evals=64, batch_evaluate_fn=batch_fn(0), logger=log, seed=5,
             batch_size=M, data_dir=d, task_name="any_qpso_s5", resume=True,
             alpha_max=1.2, alpha_min=0.3)
    o2.time_budget_s = 10.0
    try:
        o2.run(); check("續跑被拒絕", False)
    except RuntimeError:
        check("續跑被拒絕", True)

print("\n=== 5. 未設時間預算時行為與舊版逐位元相同 ===")
with tempfile.TemporaryDirectory() as d:
    a = make("rr_qpso2", 7, d, max_evals=96, alpha_max=1.2, alpha_min=0.3); xa, fa = a.run()
with tempfile.TemporaryDirectory() as d:
    b = make("rr_qpso2", 7, d, max_evals=96, alpha_max=1.2, alpha_min=0.3)
    b.time_budget_s = None; b.checkpoint_s = []
    xb, fb = b.run()
check("同 seed、無預算 → 結果相同", np.array_equal(xa, xb) and fa == fb, f"{fa:.10f} vs {fb:.10f}")

print("\n=== 5b. α 依牆鐘退火 ===")
with tempfile.TemporaryDirectory() as d:
    o = make("qpso", 9, d, max_evals=10**6, delay=0.05, alpha_max=1.2, alpha_min=0.3)
    o.time_budget_s = 2.0; o.alpha_by_time = True
    seen = []
    import optimizers.qpso as _q
    _clip = _q.np.clip
    def spy(a, lo, hi, *args, **kw):
        r = _clip(a, lo, hi, *args, **kw)
        if _q.np.ndim(a) == 0 and lo == 0.3 and hi == 1.2:
            seen.append(float(r))
        return r
    _q.np.clip = spy
    try:
        o.run()
    finally:
        _q.np.clip = _clip
    check("α 從接近 α_max 單調下降到接近 α_min", len(seen) > 5 and seen[0] > 1.1 and seen[-1] < 0.5
          and all(b <= a + 1e-12 for a, b in zip(seen, seen[1:])),
          f"n={len(seen)} first={seen[0]:.3f} last={seen[-1]:.3f}" if seen else "no alpha seen")
with tempfile.TemporaryDirectory() as d:
    a = make("rr_qpso2", 7, d, max_evals=96, alpha_max=1.2, alpha_min=0.3); xa, fa = a.run()
with tempfile.TemporaryDirectory() as d:
    b = make("rr_qpso2", 7, d, max_evals=96, alpha_max=1.2, alpha_min=0.3)
    b.alpha_by_time = True            # 沒有時間預算 → 必須退回依評估次數
    xb, fb = b.run()
check("alpha_by_time 但無時間預算 → 行為不變", np.array_equal(xa, xb) and fa == fb)

print("\n=== 6. CLI 接線 ===")
h = subprocess.run([sys.executable, "run_experiment.py", "--help"], capture_output=True, text=True).stdout
for flag in ("--fresh_shot_noise", "--time_budget_h", "--checkpoint_hours"):
    check(f"run_experiment 有 {flag}", flag in h)
r = subprocess.run([sys.executable, "run_experiment.py", "--optimizer", "qpso", "--no_pool",
                    "--fresh_shot_noise", "--data_dir", tempfile.mkdtemp()], capture_output=True, text=True)
check("--fresh_shot_noise 搭配非 pool 路徑會中止", r.returncode != 0 and "fresh_shot_noise" in (r.stdout + r.stderr))
import re as _re
src = open("run_experiment.py", encoding="utf-8").read()
check("檢查點字串接受冒號分隔（sbatch --export 會切逗號）", 'split(r"[,:]"' in src)
check("冒號字串解析成 7 個時間點", [float(x) for x in _re.split(r"[,:]", "2:4:8:12:16:20:24") if x] == [2, 4, 8, 12, 16, 20, 24])
sl = open("benchmark/benchmark.slurm").read()
for v in ("GPU_IDS", "FRESH_SHOT_NOISE", "TIME_BUDGET_H", "CHECKPOINT_HOURS"):
    check(f"benchmark.slurm 轉傳 {v}", v in sl)
pa = open("benchmark/paired_anytime.slurm").read()
check("配對腳本：兩臂皆 fresh noise、同一 SHOT_SEED", "FRESH_SHOT_NOISE=1" in pa and 'SHOT_SEED="$SEED"' in pa)
check("配對腳本：以時間回收 worker（各演算法一致）", "QMG_RECYCLE_MINUTES=18" in pa and "QMG_RECYCLE_EVERY=0" in pa)
check("配對腳本：population arm 帶 α 覆寫", "ALPHA_MAX=${ALPHA_MAX:-1.2}" in pa)
check("配對腳本：拒絕含逗號的 CHECKPOINT_HOURS", "CHECKPOINT_HOURS 含逗號" in pa)
check("配對腳本：population arm 帶 ALPHA_BY_TIME=1", "ALPHA_BY_TIME=1" in pa)
check("benchmark.slurm 轉傳 ALPHA_BY_TIME", "ALPHA_BY_TIME" in sl)
check("run_experiment 有 --alpha_by_time", "--alpha_by_time" in h)

print("\n" + ("全部通過" if ok else "★ 有項目未通過，不可送實驗"))
sys.exit(0 if ok else 1)
