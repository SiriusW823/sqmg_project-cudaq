"""
cudaq_arg_diagnostic.py
診斷 cudaq 0.7.1 的 list[float] 傳參問題
放到 ~/sqmg_project-cudaq/ 後執行：
  PYTHONPATH=~/sqmg_project-cudaq python cudaq_arg_diagnostic.py
"""
import math
import cudaq
import numpy as np

print("=" * 60)
print("cudaq 0.7.1 list[float] 參數傳遞診斷")
print("=" * 60)

# ── 取得 sample.py 路徑並印出關鍵部分 ─────────────────────────────
import importlib.util, inspect
spec = importlib.util.find_spec("cudaq")
cudaq_path = spec.origin  # __init__.py 路徑
import os
sample_py = os.path.join(os.path.dirname(cudaq_path), "runtime", "sample.py")
print(f"\n[INFO] sample.py 路徑: {sample_py}")
if os.path.exists(sample_py):
    with open(sample_py) as f:
        content = f.read()
    print("[INFO] sample.py 前 80 行：")
    for i, line in enumerate(content.splitlines()[:80], 1):
        print(f"  {i:3d} | {line}")
else:
    print("[WARN] sample.py 不存在，嘗試找 _sample_impl")

# ── Test 1：最小 list[float] kernel（1 個元素）─────────────────────
print("\n[Test 1] list[float] kernel，1 個元素")
try:
    @cudaq.kernel
    def k1(params: list[float]):
        q = cudaq.qvector(1)
        ry(params[0], q[0])
        mz(q[0])

    result1 = cudaq.sample(k1, [0.5], shots_count=20)
    print(f"  ✓ 成功：{dict(result1.items())}")
except Exception as e:
    print(f"  ✗ 失敗：{e}")

# ── Test 2：list[float] kernel，3 個元素 ───────────────────────────
print("\n[Test 2] list[float] kernel，3 個元素")
try:
    @cudaq.kernel
    def k2(params: list[float]):
        q = cudaq.qvector(2)
        ry(params[0], q[0])
        ry(params[1], q[1])
        ry(params[2], q[0])
        mz(q[0])
        mz(q[1])

    result2 = cudaq.sample(k2, [0.1, 0.2, 0.3], shots_count=20)
    print(f"  ✓ 成功：{dict(result2.items())}")
except Exception as e:
    print(f"  ✗ 失敗：{e}")

# ── Test 3：numpy array 傳法 ───────────────────────────────────────
print("\n[Test 3] 同 kernel k2，改傳 numpy array")
try:
    result3 = cudaq.sample(k2, np.array([0.1, 0.2, 0.3]), shots_count=20)
    print(f"  ✓ 成功：{dict(result3.items())}")
except Exception as e:
    print(f"  ✗ 失敗：{e}")

# ── Test 4：mid-circuit measurement + list[float] ─────────────────
print("\n[Test 4] 含 mid-circuit measurement 的 list[float] kernel")
try:
    @cudaq.kernel
    def k4(params: list[float]):
        q = cudaq.qvector(4)
        ry(params[0], q[0])
        b0 = mz(q[0])
        if b0:
            ry(params[1], q[1])
        mz(q[1])
        mz(q[2])
        mz(q[3])

    result4 = cudaq.sample(k4, [0.5, 0.3], shots_count=20)
    print(f"  ✓ 成功：{dict(result4.items())}")
except Exception as e:
    print(f"  ✗ 失敗：{e}")

# ── Test 5：查看 kernel 的 arg_types attribute ─────────────────────
print("\n[Test 5] 查看 kernel 的 metadata / arg_types")
for name, k in [("k1", k1), ("k2", k2), ("k4", k4)]:
    try:
        print(f"  {name}.arg_types = {k.arg_types}")
    except AttributeError:
        pass
    try:
        print(f"  {name}.arguments = {k.arguments}")
    except AttributeError:
        pass
    print(f"  {name} type = {type(k)}")
    print(f"  {name} dir  = {[x for x in dir(k) if not x.startswith('__')]}")

# ── Test 6：嘗試不同的呼叫方式 ────────────────────────────────────
print("\n[Test 6] 嘗試不同呼叫方式（小 kernel）")

# 6a: 直接呼叫 kernel
try:
    ret = k1([0.5])
    print(f"  k1([0.5]) direct call: {ret} (type: {type(ret)})")
except Exception as e:
    print(f"  k1([0.5]) direct call failed: {e}")

# 6b: async_sample
try:
    fut = cudaq.async_sample(k2, [0.1, 0.2, 0.3], shots_count=20)
    r = fut.get()
    print(f"  async_sample ✓: {dict(r.items())}")
except Exception as e:
    print(f"  async_sample ✗: {e}")

# ── Test 7：134 個元素的 list[float]（接近實際大小）─────────────
print("\n[Test 7] 134 個元素 list[float]（實際大小）")
try:
    @cudaq.kernel
    def k7(params: list[float]):
        q = cudaq.qvector(3)
        ry(params[0], q[0])
        ry(params[133], q[1])
        mz(q[0])
        mz(q[1])
        mz(q[2])

    params_134 = [0.5] * 134
    result7 = cudaq.sample(k7, params_134, shots_count=20)
    print(f"  ✓ 成功：{dict(result7.items())}")
except Exception as e:
    print(f"  ✗ 失敗：{e}")

# ── Test 8：嘗試 *args 展開傳法（如果 cudaq 期望 *args）────────────
print("\n[Test 8] 以 *args 方式傳 3 個 float（看是否匹配 list[float]）")
try:
    result8 = cudaq.sample(k2, *[0.1, 0.2, 0.3], shots_count=20)
    print(f"  ✓ 成功（cudaq 接受 *args 展開）：{dict(result8.items())}")
    print("  [IMPORTANT] cudaq 0.7.1 期望 *args 展開，不是傳 list！")
except Exception as e:
    print(f"  ✗ 失敗：{e}")

print("\n" + "=" * 60)
print("診斷完成，請把上方所有輸出貼給工程師分析")
print("=" * 60)
