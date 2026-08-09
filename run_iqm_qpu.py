#!/usr/bin/env python3
"""
==============================================================================
run_iqm_qpu.py — 在 IQM Resonance 真實 QPU 上執行 QMG N=9 電路（v11.0 新增）
==============================================================================

目的
----
把既有的 QMG N=9 動態電路（_qmg_n9）投到 https://resonance.iqm.tech/ 的真機，
取得 validity / uniqueness，並與 GPU 模擬結果對照。

CUDA-Q 原生支援 IQM target，且會在執行時向 IQM Server 取回「當前校準的動態
量子架構」，據此即時 transpile 到該 QPU 的拓樸，因此不需要另外寫 qiskit 轉譯。

先讀這段（重要）
----------------
_qmg_n9 的靜態統計（本檔 --analyze 可重新產生）：

    20 qubits ／ 90 mid-circuit measurement ／ 85 個 classical conditional
    85 個雙量子閘（81 CRY + 4 CX）／ 其中 79 個位於 conditional 區塊內
    16/20 qubits 被重複量測（最多 8 次）→ 依賴 qubit reuse + active reset

IQM 原生指令集為 prx（相位旋轉）、cz、measure，動態電路另有 cc_prx
（classically-controlled prx，單一 feedback key）。對照之下：

    ✅ 121 個 conditional 單量子閘  → 可映射為 cc_prx
    ✅ conditional-X 形式的 active reset → cc_prx 可表達
    ❌ 79 個 conditional 雙量子閘   → 需要 classically-controlled CZ，IQM ISA 沒有
    ❌ 15 個 `if a or b:` 複合條件  → cc_prx 只接受單一 feedback key，無法直接表達

因此「完整 N=9 電路直投真機」預期會在 transpile 階段就被拒絕。
本腳本的設計是讓這件事以「明確的編譯器診斷」而不是「燒掉額度後才發現」的方式呈現：
Stage 2 的 emulate 模式會連線取回真實 QPU 架構並跑完整的 target 編譯流程，
但不送出作業、不消耗額度。確認可編譯後 Stage 3 才真正投遞。

即使 transpile 過得了，保真度上限也已由電路規模決定（見 --analyze 輸出）：
    166 個原生 CZ（未計 routing）× 99.5% ≈ 43% 電路成功率；
    Garnet 20q 剛好 20 顆 qubit，沒有多餘 qubit 做 routing，實務上 SWAP 會
    再放大 3-5 倍 CZ 數 → 端到端成功率降到 ~6%。

使用方式
--------
    # 0) 取得 token：Resonance 網站 → 個人資料頁 → API Token
    export IQM_TOKEN="<your-token>"
    export IQM_SERVER_URL="https://cocos.resonance.meetiqm.com/garnet"

    python run_iqm_qpu.py --analyze          # 只做離線可行性分析，不連線
    python run_iqm_qpu.py --stage 1          # 連線煙霧測試（2 qubits，極少 shots）
    python run_iqm_qpu.py --stage 2          # emulate 模式編譯診斷（不花額度）
    python run_iqm_qpu.py --stage 3 --shots 1000   # 真機投遞完整 N=9
    python run_iqm_qpu.py --all --shots 1000       # 依序執行 1 → 2 → 3

放置位置：run_iqm_qpu.py（專案根目錄）
==============================================================================
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import sys
import time

import numpy as np


# ===========================================================================
# Stage 0：離線可行性分析（不需要 token、不需要 cudaq）
# ===========================================================================

_TWO_QUBIT = {"x.ctrl", "ry.ctrl", "z.ctrl", "h.ctrl"}
_ONE_QUBIT = {"x", "y", "z", "h", "ry", "rx", "rz", "s", "t"}


def _iter_call_names(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                yield n.func.id
            elif isinstance(n.func, ast.Attribute):
                base = n.func.value.id if isinstance(n.func.value, ast.Name) else "?"
                yield f"{base}.{n.func.attr}"


def analyze_kernel(repo: str) -> dict:
    """靜態分析 _qmg_n9，回傳與 IQM ISA 對照所需的統計量。"""
    path = os.path.join(repo, "qmg", "utils", "build_dynamic_circuit_cudaq.py")
    src  = open(path, "r", encoding="utf-8").read()
    fn   = [n for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.FunctionDef) and n.name == "_qmg_n9"][0]

    gates    = collections.Counter(_iter_call_names(fn))
    n_qubits = 0
    for n in ast.walk(fn):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "qvector" and n.args
                and isinstance(n.args[0], ast.Constant)):
            n_qubits = n.args[0].value

    measured = collections.Counter()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "mz" and n.args:
            a = n.args[0]
            if (isinstance(a, ast.Subscript) and isinstance(a.slice, ast.Constant)):
                measured[a.slice.value] += 1

    cond_total = cond_compound = cond_with_2q = 0
    cc_1q = cc_2q = 0
    for n in ast.walk(fn):
        if not isinstance(n, ast.If):
            continue
        cond_total += 1
        if isinstance(n.test, ast.BoolOp):
            cond_compound += 1
        c = collections.Counter()
        for st in n.body:
            c.update(_iter_call_names(st))
        n2 = sum(v for k, v in c.items() if k in _TWO_QUBIT)
        n1 = sum(v for k, v in c.items() if k in _ONE_QUBIT)
        cc_1q += n1
        cc_2q += n2
        if n2:
            cond_with_2q += 1

    # IQM 原生 CZ 下界：CRY ≈ 2 CZ（+prx），CX ≈ 1 CZ（+prx）
    cz_min = gates.get("ry.ctrl", 0) * 2 + gates.get("x.ctrl", 0) * 1

    return {
        "n_qubits":       n_qubits,
        "gates":          dict(gates),
        "n_mz":           gates.get("mz", 0),
        "n_2q":           sum(v for k, v in gates.items() if k in _TWO_QUBIT),
        "cond_total":     cond_total,
        "cond_compound":  cond_compound,
        "cond_with_2q":   cond_with_2q,
        "cc_1q":          cc_1q,
        "cc_2q":          cc_2q,
        "reused_qubits":  {q: c for q, c in sorted(measured.items()) if c > 1},
        "max_reuse":      max(measured.values()) if measured else 0,
        "cz_min":         cz_min,
    }


def print_analysis(a: dict) -> None:
    # 供應商公布的中位數保真度（Garnet / Emerald 同級）
    F_CZ, F_1Q, F_RO = 0.995, 0.9992, 0.9994
    n_1q = sum(v for k, v in a["gates"].items() if k in _ONE_QUBIT) + a["cc_1q"]

    def success(cz):
        return (F_CZ ** cz) * (F_1Q ** n_1q) * (F_RO ** a["n_mz"])

    print("=" * 78)
    print("QMG N=9 電路 × IQM Resonance 可行性分析（靜態，未連線）")
    print("=" * 78)
    print(f"\n[電路規模]")
    print(f"  qubits                     : {a['n_qubits']}")
    print(f"  mid-circuit measurements   : {a['n_mz']}")
    print(f"  雙量子閘                   : {a['n_2q']}  "
          f"(CRY {a['gates'].get('ry.ctrl',0)} + CX {a['gates'].get('x.ctrl',0)})")
    print(f"  classical conditionals     : {a['cond_total']}"
          f"（其中複合條件 `A or B`：{a['cond_compound']}）")
    print(f"  qubit reuse                : {len(a['reused_qubits'])}/{a['n_qubits']} 顆被重複量測，"
          f"最多 {a['max_reuse']} 次")

    print(f"\n[對照 IQM 原生指令集 prx / cz / measure / cc_prx]")
    print(f"  ✅ conditional 單量子閘 {a['cc_1q']:3d} 個 → cc_prx 可表達")
    print(f"  ✅ conditional-X 形式的 active reset → cc_prx 可表達")
    print(f"  ❌ conditional 雙量子閘 {a['cc_2q']:3d} 個 → 需要 cc_cz，IQM ISA 未提供")
    print(f"  ❌ 複合條件 `A or B` {a['cond_compound']:3d} 個 → cc_prx 只吃單一 feedback key")
    print(f"     （這 {a['cond_with_2q']} 個 conditional 區塊同時踩到上面兩點）")

    print(f"\n[保真度上限，即使能編譯]")
    print(f"  原生 CZ 下界（未計 routing）: {a['cz_min']}")
    for label, mult in (("理想 all-to-all", 1), ("Garnet 20q + SWAP ~3×", 3)):
        cz = a["cz_min"] * mult
        s  = success(cz)
        print(f"  {label:24s} CZ={cz:5d}  端到端成功率 ≈ {s*100:5.1f}%"
              f"  → 5000 shots 中約 {s*5000:.0f} 個 shot 未受閘錯誤汙染")

    print(f"\n[結論]")
    print("  完整 N=9 電路預期無法直接在 IQM 真機執行，阻礙點是 ISA 表達力"
          "（conditional 雙量子閘 + 複合條件），而非 qubit 數不足。")
    print("  Stage 2（emulate）會用真實 QPU 架構跑完整編譯流程給出確切診斷，"
          "且不消耗額度——請先跑 Stage 2 再決定是否投遞 Stage 3。")
    print("=" * 78)


# ===========================================================================
# 連線設定
# ===========================================================================

def resolve_credentials(args) -> tuple:
    """
    取得 token 與 server URL。

    CUDA-Q 讀取順序：IQM_TOKEN 環境變數優先；否則讀 IQM_TOKENS_FILE 指向的
    JSON（格式 {"access_token": "..."}）。本函式把 .env 也納入，方便本地使用，
    並且「不」把 token 印進 log。
    """
    token = os.environ.get("IQM_TOKEN")
    url   = args.url or os.environ.get("IQM_SERVER_URL")

    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if (not token or not url) and os.path.exists(env_path):
        for line in open(env_path, "r", encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if k.strip() == "IQM_TOKEN" and not token:
                token = v
            elif k.strip() == "IQM_SERVER_URL" and not url:
                url = v

    if token:
        os.environ["IQM_TOKEN"] = token
    if url:
        os.environ["IQM_SERVER_URL"] = url

    if not token and not os.environ.get("IQM_TOKENS_FILE"):
        print(
            "[ERROR] 找不到 IQM API token。取得與設定方式：\n"
            "  1. 登入 https://resonance.iqm.tech/ → 個人資料頁 → 產生 API Token\n"
            "  2. export IQM_TOKEN=\"<token>\"\n"
            "     或在專案根目錄建立 .env：IQM_TOKEN=<token>\n"
            "     或 echo '{ \"access_token\": \"<token>\" }' > resonance-token.json\n"
            "        export IQM_TOKENS_FILE=\"$PWD/resonance-token.json\"\n"
            "  ⚠ token 等同帳號憑證，請勿 commit 進 git（.env 已在 .gitignore）。",
            file=sys.stderr,
        )
        sys.exit(2)

    if not url:
        print(
            "[ERROR] 未設定 IQM_SERVER_URL。範例：\n"
            "  export IQM_SERVER_URL=\"https://cocos.resonance.meetiqm.com/garnet\"\n"
            "  （確切 URL 以 Resonance 網站上該 QPU 的頁面為準；\n"
            "    Garnet=20 qubits，Emerald=54 qubits）",
            file=sys.stderr,
        )
        sys.exit(2)

    return token, url


# QPU 名稱 → CUDA-Q iqm target 的 qpu-architecture 值。
#
# ★ 名稱必須「完全對應」CUDA-Q 隨附的 mapping 檔：
#       site-packages/targets/mapping/iqm/Crystal_5.txt
#                                        /Crystal_20.txt
#                                        /Crystal_54.txt
#   注意是底線（Crystal_54），不是 Crystal54；打錯會得到
#       Path .../mapping/iqm/Crystal54.txt does not exist
#   然後 core dump，而不是一個清楚的錯誤訊息。
#
#   也就是說 CUDA-Q 目前只支援這三種 IQM 拓樸；Resonance 上的 Sirius(16q)
#   與 Deneb(Star 6q) 沒有對應的 mapping 檔，無法用 CUDA-Q 走。
IQM_ARCH_MAP = {
    "emerald": "Crystal_54",   # 54 qubits
    "garnet":  "Crystal_20",   # 20 qubits
}

SUPPORTED_ARCHS = ["Crystal_5", "Crystal_20", "Crystal_54"]


def infer_arch(url: str, override: str = None) -> str:
    """從 server URL 尾端的機器名推出 qpu-architecture。"""
    if override:
        return override
    tail = url.rstrip("/").rsplit("/", 1)[-1].lower()
    if tail not in IQM_ARCH_MAP:
        print(f"[WARN] 無法從 URL 尾端 '{tail}' 判斷架構，預設用 Crystal_54（Emerald）。"
              f"\n       CUDA-Q 支援的 IQM 架構只有：{SUPPORTED_ARCHS}"
              f"\n       可用 --arch 明確指定。")
    return IQM_ARCH_MAP.get(tail, "Crystal_54")


def set_iqm_target(emulate: bool, url: str, arch: str, extra: dict = None):
    """
    設定 CUDA-Q 的 iqm target。

    ★ 實測（cudaq 0.7.1 與 0.12.0 皆然）：光給 url= 會被拒
        RuntimeError: QPU architecture is not provided
      必須同時給 qpu-architecture。文件上「只給 url 就會自動抓 dynamic
      quantum architecture」的行為在這兩個版本都還沒生效，所以這裡一律
      顯式帶上架構名。
    """
    import cudaq
    kwargs = {"url": url, "qpu-architecture": arch}
    if emulate:
        kwargs["emulate"] = True
    if extra:
        kwargs.update(extra)
    cudaq.set_target("iqm", **kwargs)
    return cudaq.get_target()


# ===========================================================================
# Stage 1：連線煙霧測試
# ===========================================================================

def stage1_smoke(args, url: str, arch: str) -> bool:
    """
    最小電路（2 qubits, Bell state, 少量 shots）投真機。

    目的不是算什麼，而是把「憑證是否有效、URL 是否指向可用 QPU、排隊多久、
    一次作業的往返成本」這四件事量出來，再決定要不要投大電路。
    """
    import cudaq

    print("\n" + "=" * 78)
    print(f"Stage 1：連線煙霧測試（2 qubits, {args.smoke_shots} shots，真機）")
    print("=" * 78)

    @cudaq.kernel
    def bell():
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])
        mz(q)

    try:
        tgt = set_iqm_target(emulate=False, url=url, arch=arch)
        print(f"  target       : {tgt.name}")
        print(f"  server URL   : {url}")
        print(f"  architecture : {arch}")
    except Exception as e:                              # noqa: BLE001
        print(f"[FAIL] set_target('iqm') 失敗：{e}")
        print("  常見原因：token 過期／URL 錯誤／該 QPU 目前離線維護。")
        return False

    t0 = time.time()
    try:
        res = cudaq.sample(bell, shots_count=args.smoke_shots)
    except Exception as e:                              # noqa: BLE001
        print(f"[FAIL] 投遞失敗：{e}")
        return False
    dt = time.time() - t0

    counts = {k: res.count(k) for k in res}
    total  = sum(counts.values()) or 1
    p_corr = (counts.get("00", 0) + counts.get("11", 0)) / total

    print(f"  往返總時間   : {dt:.1f}s（含排隊）")
    print(f"  counts       : {counts}")
    print(f"  Bell 正確率  : {p_corr*100:.1f}%（理想 100%，真機受雙量子閘與讀出雜訊影響）")
    print("  ✓ 憑證與連線正常" if p_corr > 0.5 else "  ⚠ 結果異常，請檢查 QPU 狀態")
    return True


# ===========================================================================
# Stage 2：emulate 模式編譯診斷（不消耗 QPU 額度）
# ===========================================================================

def stage2_emulate(args, url: str, arch: str) -> bool:
    """
    用 emulate=True 對真正的 _qmg_n9 跑完整的 IQM target 編譯流程。

    關鍵：emulate 模式「仍會連線取回該 QPU 當前校準的 dynamic quantum
    architecture」，然後據此 transpile —— 也就是說編譯階段的檢查與真機完全相同，
    只是最後在本地做無雜訊模擬而不送出作業。因此這是找出 ISA 不相容處
    最便宜、也最準確的方式。
    """
    import cudaq
    from qmg.utils.build_dynamic_circuit_cudaq import _qmg_n9
    from qmg.utils.weight_generator import ConditionalWeightsGenerator

    print("\n" + "=" * 78)
    print(f"Stage 2：emulate 編譯診斷（完整 N=9 電路，{args.emulate_shots} shots，不花額度）")
    print("=" * 78)

    save_qa = os.path.join(args.out_dir, "iqm_qpu_architecture.json")
    if not os.path.exists(save_qa):
        # 把真機當前架構存檔，之後可離線重跑 emulate（IQM_SAVE_QPU_QA 若檔案
        # 已存在會直接報錯中止，故先檢查）。
        os.environ["IQM_SAVE_QPU_QA"] = save_qa

    try:
        tgt = set_iqm_target(emulate=True, url=url, arch=arch)
        print(f"  target: {tgt.name} (emulate=True)  architecture: {arch}")
        if os.path.exists(save_qa):
            arch = json.load(open(save_qa, "r", encoding="utf-8"))
            n_q  = len(arch.get("qubits", []) or arch.get("nodes", []) or [])
            print(f"  已取回並存檔 QPU 架構：{save_qa}"
                  + (f"（{n_q} qubits）" if n_q else ""))
    except Exception as e:                              # noqa: BLE001
        print(f"[FAIL] emulate target 設定失敗：{e}")
        return False
    finally:
        os.environ.pop("IQM_SAVE_QPU_QA", None)

    cwg = ConditionalWeightsGenerator(9, smarts=None, disable_connectivity_position=[])
    w   = cwg.apply_chemistry_constraint(
        cwg.generate_conditional_random_weights(random_seed=42))
    w_list = [float(x) for x in w]

    print("  正在對 _qmg_n9 做 target 編譯（20 qubits / 90 MCM / 85 conditional）...")
    t0 = time.time()
    try:
        res = cudaq.sample(_qmg_n9, w_list, shots_count=args.emulate_shots)
    except Exception as e:                              # noqa: BLE001
        print(f"\n[編譯／執行被拒] {type(e).__name__}: {e}\n")
        print("  這正是本階段要取得的診斷。對照 --analyze 的預測，"
              "最可能的原因是 conditional 區塊內的雙量子閘（需 cc_cz）"
              "與 `if a or b:` 複合條件（cc_prx 僅接受單一 feedback key）。")
        print("  → 不建議繼續 Stage 3：真機會在同一個編譯階段以相同理由拒絕，"
              "只是多花排隊時間。")
        return False
    dt = time.time() - t0

    print(f"  ✓ 編譯通過並完成無雜訊模擬（{dt:.1f}s）")
    _decode_and_report(res, args.emulate_shots, label="emulate（無雜訊）",
                       out_dir=args.out_dir, tag="emulate")
    print("  → 編譯層面可行，可以進 Stage 3 投真機。")
    return True


# ===========================================================================
# Stage 3：真機投遞完整 N=9 電路
# ===========================================================================

def stage3_hardware(args, url: str, arch: str) -> bool:
    import cudaq
    from qmg.utils.build_dynamic_circuit_cudaq import _qmg_n9
    from qmg.utils.weight_generator import ConditionalWeightsGenerator

    print("\n" + "=" * 78)
    print(f"Stage 3：真機投遞完整 N=9 電路（{args.shots} shots）")
    print("=" * 78)
    print("  ⚠ 本階段會消耗 QPU 額度，且可能排隊數分鐘至數小時。")

    try:
        tgt = set_iqm_target(emulate=False, url=url, arch=arch)
        print(f"  target: {tgt.name}  URL: {url}  architecture: {arch}")
    except Exception as e:                              # noqa: BLE001
        print(f"[FAIL] set_target 失敗：{e}")
        return False

    cwg = ConditionalWeightsGenerator(9, smarts=None, disable_connectivity_position=[])
    w   = cwg.apply_chemistry_constraint(
        cwg.generate_conditional_random_weights(random_seed=args.seed))
    np.save(os.path.join(args.out_dir, "iqm_weights.npy"), w)
    w_list = [float(x) for x in w]

    t0 = time.time()
    try:
        res = cudaq.sample(_qmg_n9, w_list, shots_count=args.shots)
    except Exception as e:                              # noqa: BLE001
        print(f"\n[FAIL] 真機執行失敗：{type(e).__name__}: {e}")
        return False
    dt = time.time() - t0

    print(f"  ✓ 完成，往返 {dt:.1f}s（含排隊）")
    _decode_and_report(res, args.shots, label="IQM 真機",
                       out_dir=args.out_dir, tag="hardware")
    return True


# ===========================================================================
# 結果解碼（沿用既有 pipeline，確保與模擬結果可直接對比）
# ===========================================================================

def _decode_and_report(result, shots: int, label: str, out_dir: str, tag: str) -> None:
    """
    用與 GPU 模擬完全相同的解碼路徑（_reconstruct_bitstrings_n9 →
    bond disconnection correction → post_process_quantum_state → SMILES）
    算出 validity / uniqueness，因此數字可與 results_v8 等模擬 log 直接比較。
    """
    from qmg.generator_cudaq import _reconstruct_bitstrings_n9
    from qmg.utils.build_dynamic_circuit_cudaq import DynamicCircuitBuilderCUDAQ
    from qmg.utils.chemistry_data_processing import MoleculeQuantumStateGenerator

    try:
        raw = _reconstruct_bitstrings_n9(result)
    except Exception as e:                              # noqa: BLE001
        print(f"  [WARN] bitstring 重建失敗：{e}")
        print("         真機回傳的暫存器結構可能與模擬不同，"
              "請檢查 result 的 register 名稱。")
        return

    if not raw:
        print("  [WARN] raw_counts 為空 → validity=0")
        return

    builder = DynamicCircuitBuilderCUDAQ(
        num_heavy_atom=9, temperature=0.2,
        remove_bond_disconnection=True, chemistry_constraint=False,
    )
    dgen = MoleculeQuantumStateGenerator(
        heavy_atom_size=9, ncpus=1, sanitize_method="strict")

    smiles_dict: dict = {}
    n_valid = 0
    for bs, cnt in raw.items():
        bs_fixed = builder.apply_bond_disconnection_correction(bs)
        qs       = dgen.post_process_quantum_state(bs_fixed, reverse=False)
        smi      = dgen.QuantumStateToSmiles(qs)
        smiles_dict[smi] = smiles_dict.get(smi, 0) + cnt
        if smi and smi != "None":
            n_valid += cnt

    validity   = n_valid / shots
    n_unique   = len([k for k in smiles_dict if k and k != "None"])
    uniqueness = n_unique / n_valid if n_valid else 0.0

    print(f"\n  ── {label} 結果 ──")
    print(f"    shots            : {shots}")
    print(f"    distinct bitstr  : {len(raw)}")
    print(f"    valid molecules  : {n_valid}")
    print(f"    validity   (V)   : {validity:.4f}")
    print(f"    uniqueness (U)   : {uniqueness:.4f}")
    print(f"    V × U            : {validity*uniqueness:.4f}")

    top = sorted(((c, s) for s, c in smiles_dict.items() if s and s != "None"),
                 reverse=True)[:10]
    if top:
        print(f"    top SMILES       : " + ", ".join(f"{s}({c})" for c, s in top))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"iqm_{tag}_result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "label": label, "shots": shots,
            "validity": validity, "uniqueness": uniqueness,
            "vu": validity * uniqueness,
            "n_valid": n_valid, "n_distinct_bitstrings": len(raw),
            "smiles_counts": {s: c for s, c in smiles_dict.items() if s and s != "None"},
        }, f, indent=2, ensure_ascii=False)
    print(f"    已存檔           : {path}")


# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="QMG N=9 on IQM Resonance QPU (v11.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--analyze", action="store_true",
                   help="只做離線可行性分析（不連線、不需要 token）")
    p.add_argument("--stage", type=int, choices=[1, 2, 3], default=None,
                   help="1=連線煙霧測試 2=emulate 編譯診斷 3=真機投遞")
    p.add_argument("--all", action="store_true", help="依序執行 stage 1 → 2 → 3")
    p.add_argument("--url", type=str, default=None,
                   help="IQM server URL；預設讀 IQM_SERVER_URL")
    p.add_argument("--shots", type=int, default=1000, help="Stage 3 真機 shots")
    p.add_argument("--smoke_shots", type=int, default=100, help="Stage 1 shots")
    p.add_argument("--emulate_shots", type=int, default=200, help="Stage 2 shots")
    p.add_argument("--seed", type=int, default=42, help="權重向量的隨機種子")
    p.add_argument("--arch", type=str, default=None,
                   help="qpu-architecture 值。預設由 URL 尾端機器名推得"
                        "（emerald→Crystal54, garnet→Crystal20）。")
    p.add_argument("--out_dir", type=str, default="results_iqm")
    p.add_argument("--force", action="store_true",
                   help="Stage 2 診斷失敗時仍強制執行 Stage 3")
    args = p.parse_args()

    repo = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.out_dir, exist_ok=True)

    if args.analyze or (args.stage is None and not args.all):
        print_analysis(analyze_kernel(repo))
        if args.analyze:
            return
        print("\n未指定 --stage 或 --all，僅輸出離線分析。"
              "加上 --all 可依序執行連線測試。")
        return

    token, url = resolve_credentials(args)
    arch = infer_arch(url, args.arch)
    print(f"[IQM] token: ***{token[-4:] if token else '(tokens file)'}")
    print(f"[IQM] URL  : {url}")
    print(f"[IQM] arch : {arch}")

    stages = [1, 2, 3] if args.all else [args.stage]
    for s in stages:
        if s == 1:
            if not stage1_smoke(args, url, arch) and not args.force:
                print("\n[STOP] Stage 1 失敗，後續階段中止。")
                sys.exit(1)
        elif s == 2:
            if not stage2_emulate(args, url, arch) and not args.force:
                print("\n[STOP] Stage 2 診斷未通過，不投遞真機（--force 可強制）。")
                sys.exit(1)
        elif s == 3:
            if not stage3_hardware(args, url, arch):
                sys.exit(1)


if __name__ == "__main__":
    main()
