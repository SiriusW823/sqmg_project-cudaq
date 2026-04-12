"""
cudaq_n9_diagnostic.py
針對 _qmg_dynamic_n9 kernel 本身的深度診斷
放到 ~/sqmg_project-cudaq/ 後執行：
  PYTHONPATH=~/sqmg_project-cudaq python cudaq_n9_diagnostic.py 2>&1 | tee n9_diag.txt
"""
import math
import cudaq
import numpy as np

print("=" * 70)
print("_qmg_dynamic_n9 Kernel 深度診斷")
print("=" * 70)

# ── Test A：直接匯入並呼叫 _qmg_dynamic_n9 ───────────────────────────────
print("\n[Test A] 直接 import 並 sample _qmg_dynamic_n9（不透過 Generator）")
try:
    cudaq.set_target("qpp-cpu")
    from qmg.utils.build_dynamic_circuit_cudaq import _qmg_dynamic_n9
    print(f"  kernel arguments = {_qmg_dynamic_n9.arguments}")
    w = [0.5] * 134
    result = cudaq.sample(_qmg_dynamic_n9, w, shots_count=5)
    print(f"  ✓ 成功：{dict(list(result.items())[:3])}...")
except Exception as e:
    print(f"  ✗ 失敗：{e}")

# ── Test B：印出 MLIR module（看 kernel 的簽名是否正確） ─────────────────
print("\n[Test B] 印出 _qmg_dynamic_n9 的 MLIR module（前 40 行）")
try:
    from qmg.utils.build_dynamic_circuit_cudaq import _qmg_dynamic_n9
    mlir_str = str(_qmg_dynamic_n9.module)
    lines = mlir_str.splitlines()
    for i, line in enumerate(lines[:40], 1):
        print(f"  {i:3d} | {line}")
    if len(lines) > 40:
        print(f"  ... ({len(lines)} lines total)")
except Exception as e:
    print(f"  ✗ 失敗：{e}")

# ── Test C：找出第一個讓 sample 失敗的操作 ────────────────────────────────
print("\n[Test C] 找出問題節點 — 逐步增加複雜度")

# C1: 只有 ry + mz（no conditional）
@cudaq.kernel
def k_c1(weights: list[float]):
    q = cudaq.qvector(20)
    ry(math.pi * weights[0], q[0])
    x(q[1])
    ry(math.pi * weights[2], q[2])
    a1_0 = mz(q[0])
    a1_1 = mz(q[1])
    a2_0 = mz(q[2])
    a2_1 = mz(q[3])

try:
    r = cudaq.sample(k_c1, [0.5]*134, shots_count=5)
    print(f"  C1 (ry+mz, no cond): ✓ {dict(list(r.items())[:2])}")
except Exception as e:
    print(f"  C1 ✗ {e}")

# C2: + 第一個 if a2_0 or a2_1 條件
@cudaq.kernel
def k_c2(weights: list[float]):
    q = cudaq.qvector(20)
    ry(math.pi * weights[0], q[0])
    x(q[1])
    ry(math.pi * weights[2], q[2])
    ry(math.pi * weights[4], q[3])
    x.ctrl(q[0], q[1])
    ry.ctrl(math.pi * weights[3], q[1], q[2])
    x.ctrl(q[2], q[3])
    ry.ctrl(math.pi * weights[1], q[0], q[1])
    x.ctrl(q[1], q[2])
    ry.ctrl(math.pi * weights[5], q[2], q[3])
    a1_0 = mz(q[0])
    a1_1 = mz(q[1])
    a2_0 = mz(q[2])
    a2_1 = mz(q[3])
    if a2_0 or a2_1:
        ry(math.pi * weights[6], q[4])
        x(q[5])
        x.ctrl(q[4], q[5])
        ry.ctrl(math.pi * weights[7], q[4], q[5])
    b21_0 = mz(q[4])
    b21_1 = mz(q[5])

try:
    r = cudaq.sample(k_c2, [0.5]*134, shots_count=5)
    print(f"  C2 (+ if a2_0 or a2_1): ✓ {dict(list(r.items())[:2])}")
except Exception as e:
    print(f"  C2 ✗ {e}")

# C3: + Phase 2（包含 if a2_0: x(q[2]) 的 reset pattern）
@cudaq.kernel
def k_c3(weights: list[float]):
    q = cudaq.qvector(20)
    ry(math.pi * weights[0], q[0])
    x(q[1])
    ry(math.pi * weights[2], q[2])
    ry(math.pi * weights[4], q[3])
    x.ctrl(q[0], q[1])
    ry.ctrl(math.pi * weights[3], q[1], q[2])
    x.ctrl(q[2], q[3])
    ry.ctrl(math.pi * weights[1], q[0], q[1])
    x.ctrl(q[1], q[2])
    ry.ctrl(math.pi * weights[5], q[2], q[3])
    a1_0 = mz(q[0])
    a1_1 = mz(q[1])
    a2_0 = mz(q[2])
    a2_1 = mz(q[3])
    if a2_0 or a2_1:
        ry(math.pi * weights[6], q[4])
        x(q[5])
        x.ctrl(q[4], q[5])
        ry.ctrl(math.pi * weights[7], q[4], q[5])
    b21_0 = mz(q[4])
    b21_1 = mz(q[5])
    # Phase 2 reset pattern
    if a2_0:
        x(q[2])
    if a2_1:
        x(q[3])
    if b21_0:
        x(q[4])
    if b21_1:
        x(q[5])
    if a2_0 or a2_1:
        ry(math.pi * weights[8],  q[2])
        ry(math.pi * weights[9],  q[3])
        ry.ctrl(math.pi * weights[10], q[2], q[3])
    a3_0 = mz(q[2])
    a3_1 = mz(q[3])

try:
    r = cudaq.sample(k_c3, [0.5]*134, shots_count=5)
    print(f"  C3 (+ Phase 2 reset+atom3): ✓ {dict(list(r.items())[:2])}")
except Exception as e:
    print(f"  C3 ✗ {e}")

# ── Test D：測試 _qmg_dynamic_n9 的 MLIR 是否能正常 compile ─────────────
print("\n[Test D] 嘗試強制 compile _qmg_dynamic_n9")
try:
    from qmg.utils.build_dynamic_circuit_cudaq import _qmg_dynamic_n9
    _qmg_dynamic_n9.compile()
    print("  compile() 成功")
except Exception as e:
    print(f"  compile() 失敗：{e}")

# ── Test E：直接呼叫 kernel 物件（不透過 cudaq.sample） ─────────────────
print("\n[Test E] 直接呼叫 _qmg_dynamic_n9([0.5]*134)（single execution）")
try:
    from qmg.utils.build_dynamic_circuit_cudaq import _qmg_dynamic_n9
    cudaq.set_target("qpp-cpu")
    # 設定 execution context（模仿 sample.py 的做法）
    import cudaq.runtime as cudaq_runtime
    cudaq_runtime.setExecutionContext("sample", 1)
    _qmg_dynamic_n9([0.5] * 134)
    cudaq_runtime.resetExecutionContext()
    print("  ✓ 直接呼叫成功")
except AttributeError:
    print("  cudaq_runtime.setExecutionContext 不存在，略過")
except Exception as e:
    print(f"  ✗ 失敗：{e}")

# ── Test F：查 sample.py 完整內容 ────────────────────────────────────────
print("\n[Test F] sample.py 完整內容")
import importlib.util, os
spec = importlib.util.find_spec("cudaq")
sample_py = os.path.join(os.path.dirname(spec.origin), "runtime", "sample.py")
if os.path.exists(sample_py):
    with open(sample_py) as f:
        lines = f.readlines()
    print(f"  共 {len(lines)} 行，完整內容：")
    for i, line in enumerate(lines, 1):
        print(f"  {i:4d} | {line}", end='')
else:
    print(f"  {sample_py} 不存在")

print("\n" + "=" * 70)
print("診斷完成")
print("=" * 70)
