"""
==============================================================================
run_qpso_qmg_cudaq.py  — CUDA-Q 0.7.1 + AE-SOQPSO  (v10.3 / V8)
==============================================================================

v10.2 → v10.3  (V8)：
  - --iterations 預設 120 → 150
  - --alpha_min 0.40 → 0.30、--pair_interval 5 → 4（對齊 optimizer v1.5 預設）
  - 新增 --mode_collapse_u_thresh（預設 0.20），接進 AESOQPSOOptimizer
  - 搭配 qpso_optimizer_ae.py v1.5（mode collapse 回收 + adaptive V-U 權重）

v10.1 → v10.2  四項核心改動：

  ★ [改動一] num_sample 預設值 10000 → 5000
  ─────────────────────────────────────────────────────────────────────
    根據 birthday paradox 分析，V3 gbest 所在的電路參數的有效分子種數
    K ≈ 84,972。在 n=10000 shots 下，E[U] = 0.947；
    在 n=5000 shots 下，E[U] = 0.973，V×U 理論提升 +0.024。
    同時與 Chen et al. 2025 的 5000 shots 對齊，消除方法論不對稱。
    副作用：每次子行程評估時間從 ~284s 縮短至 ~142s，
            相同時間內可多跑一倍迭代數。

  ★ [改動二] Sobol 序列初始化（消除 seed 問題）
  ─────────────────────────────────────────────────────────────────────
    使用 scrambled Sobol 序列取代 pseudo-random 初始化。
    Sobol 是低差異序列（low-discrepancy sequence），保證 134D 空間
    的均勻覆蓋，且完全確定性（seed=0 → 可重現）。
    由 --sobol_init 旗標控制（預設開啟）。
    M 建議為 2 的冪次（64 = 2^6 最佳），由 --particles 64 設定。

  ★ [改動三] AE-SOQPSO v1.2（OBL + V-U 解耦 mbest）
  ─────────────────────────────────────────────────────────────────────
    引用 qpso_optimizer_ae.py v1.2 的新功能：
    - OBL Phase 0：對立粒子評估，覆蓋率等效翻倍
    - V-U 解耦 mbest：加入 V*_pos 和 U*_pos 的牽引，
      引導粒子向 V×U 聯合最優方向收斂
    由 --obl / --no_obl 和 --vu_decouple / --no_vu_decouple 控制。

  ★ [改動四] subprocess_timeout 自動調整
  ─────────────────────────────────────────────────────────────────────
    num_sample=5000 時每次評估 ~142s，timeout 從 600s 降至 360s，
    避免真正 hang 的子行程等太久。

  v10.1 保留（不變）：
    - parallel subprocess pool（8-GPU 並行）
    - AESOQPSOOptimizer batch_evaluate_fn 介面
    - worker_eval.py 子行程隔離（CUDA pinned memory 問題根本修正）
    - verify_workers_parallel 並行驗證

依賴：
  worker_eval.py    — 必須與本檔案在同一目錄
  qpso_optimizer_ae.py v1.2 — 必須在 PYTHONPATH 可及範圍
  scipy             — Sobol 初始化需要（pip install scipy）
==============================================================================
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

try:
    import cudaq
except ImportError:
    print("[ERROR] 無法 import cudaq。請執行：pip install cuda-quantum-cu12==0.7.1")
    sys.exit(1)

try:
    from qmg.utils import ConditionalWeightsGenerator
except ImportError as e:
    print(f"[ERROR] 無法 import qmg.utils: {e}")
    sys.exit(1)

try:
    from qpso_optimizer_ae import AESOQPSOOptimizer
except ImportError as e:
    print(f"[ERROR] 無法 import qpso_optimizer_ae: {e}")
    sys.exit(1)


# ===========================================================================
# 記憶體工具
# ===========================================================================

def _get_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024
    except Exception:
        pass
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        pass
    return -1.0


def log_memory(logger: logging.Logger, label: str = "") -> float:
    rss = _get_rss_mb()
    if rss >= 0:
        logger.info(f"[MEM] {label}  RSS={rss:.0f} MB")
    return rss


# ===========================================================================
# ★ v10.2 Sobol 初始化工具
# ===========================================================================

def make_sobol_positions(
    n_particles: int,
    n_params:    int,
    logger:      logging.Logger,
) -> np.ndarray:
    """
    使用 scrambled Sobol 序列生成粒子初始位置。

    n_particles 建議為 2 的冪次（64 = 2^6）以保證 Sobol 均勻性保證。
    若非 2 的冪次，會生成最近的 2^k 個點後截取前 n_particles 個，
    並發出警告。

    Args:
        n_particles: 粒子數 M（建議 64）
        n_params:    參數維度 D（= 134）
        logger:      logging.Logger

    Returns:
        positions: np.ndarray shape (n_particles, n_params)，值域 [0,1]
    """
    try:
        from scipy.stats import qmc
    except ImportError:
        logger.error("[Sobol] scipy 未安裝，fallback 到 random 初始化。")
        logger.error("  請執行：pip install scipy --break-system-packages")
        return None

    import math
    k = math.ceil(math.log2(n_particles))
    n_sobol = 2 ** k

    if n_sobol != n_particles:
        logger.warning(
            f"[Sobol] n_particles={n_particles} 非 2 的冪次，"
            f"生成 {n_sobol} 個點後截取前 {n_particles} 個。"
            f"建議設 --particles 64 以取得完整 Sobol 均勻性保證。"
        )

    # scramble=True：Owen scrambling，保持低差異性同時打破維度間相關結構
    # seed=0：完全確定性，不受 --seed 影響
    sampler = qmc.Sobol(d=n_params, scramble=True, seed=0)
    sobol_all = sampler.random(n=n_sobol)         # shape (n_sobol, n_params)
    positions = sobol_all[:n_particles].copy()    # shape (n_particles, n_params)

    # 計算並記錄覆蓋品質
    disc = qmc.discrepancy(positions)
    logger.info(
        f"[Sobol v10.2] 初始化完成  "
        f"n={n_particles}  d={n_params}  "
        f"discrepancy={disc:.4e}  "
        f"(scramble=True, seed=0, 完全確定性)"
    )
    # 各維度覆蓋品質
    per_dim_range = positions.max(axis=0) - positions.min(axis=0)
    logger.info(
        f"[Sobol] 各維度覆蓋範圍  "
        f"mean={per_dim_range.mean():.4f}  "
        f"min={per_dim_range.min():.4f}  "
        f"dims_under_0.5={(per_dim_range < 0.5).sum()}"
    )
    return positions


# ===========================================================================
# 參數解析
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QMG CUDA-Q 0.7.1 + AE-SOQPSO（v10.2 Sobol+OBL+VU-Decouple 版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── 基本參數 ──────────────────────────────────────────────────────────
    p.add_argument("--num_heavy_atom",    type=int,   default=9)
    p.add_argument(
        "--num_sample",        type=int,   default=5000,
        help=(
            "每次量子電路採樣的 shots 數。"
            "v10.2 預設從 10000 降至 5000：birthday paradox 分析顯示 K≈84972，"
            "5000 shots 的理論 uniqueness=0.973（vs 10000 shots 的 0.947），"
            "V×U 理論提升 +0.024，同時與 Chen 2025 對齊。"
        ),
    )
    p.add_argument(
        "--particles",         type=int,   default=64,
        help=(
            "粒子數 M。v10.2 預設從 56 改為 64（=2^6），"
            "確保 Sobol 序列的均勻性保證完整成立。"
        ),
    )
    p.add_argument("--iterations",        type=int,   default=150)  # v10.3(V8): 120 → 150
    p.add_argument("--seed",              type=int,   default=0,
                   help="QPSO 更新的隨機種子（位置更新、Cauchy mutation 用）。"
                        "Sobol 模式下不影響初始化。")

    # ── GPU 並行設定 ──────────────────────────────────────────────────────
    p.add_argument("--n_gpus",    type=int, default=8)
    p.add_argument("--gpu_ids",   type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument(
        "--backend", type=str, default="cudaq_nvidia",
        choices=["cudaq_qpp", "cudaq_nvidia", "cudaq_nvidia_fp64",
                 "cudaq_tensornet", "cudaq_tensornet_mps"],
    )
    p.add_argument(
        "--subprocess_timeout", type=int, default=360,
        help="每個子行程最大秒數。v10.2 預設 360s（5000 shots ~142s，留 2.5x 餘量）",
    )

    # ── ★ v10.2 Sobol 初始化 ──────────────────────────────────────────────
    p.add_argument(
        "--sobol_init",    action="store_true",  default=True,
        help="使用 scrambled Sobol 序列初始化粒子位置（預設開啟）",
    )
    p.add_argument(
        "--no_sobol_init", action="store_false", dest="sobol_init",
        help="關閉 Sobol 初始化，改用 pseudo-random（seed 參數生效）",
    )

    # ── SOQPSO 超參數 ─────────────────────────────────────────────────────
    p.add_argument("--alpha_max",          type=float, default=1.2)
    p.add_argument("--alpha_min",          type=float, default=0.3)  # v10.3(V8): 對齊 optimizer
    p.add_argument("--mutation_prob",      type=float, default=0.15)
    p.add_argument("--stagnation_limit",   type=int,   default=12)
    p.add_argument("--reinit_fraction",    type=float, default=0.25)

    # ── AE-QTS 超參數 ─────────────────────────────────────────────────────
    p.add_argument("--ae_weighting",    action="store_true",  default=True)
    p.add_argument("--no_ae_weighting", action="store_false", dest="ae_weighting")
    p.add_argument("--pair_interval",   type=int,   default=4)  # v10.3(V8): 對齊 optimizer
    p.add_argument("--rotate_factor",   type=float, default=0.015)

    # ── ★ v10.2 OBL ──────────────────────────────────────────────────────
    p.add_argument(
        "--obl",    action="store_true",  default=True,
        help="Phase 0 執行 Opposition-Based Learning（預設開啟）",
    )
    p.add_argument(
        "--no_obl", action="store_false", dest="obl",
        help="關閉 OBL",
    )

    # ── ★ v10.2 V-U 解耦 mbest ───────────────────────────────────────────
    p.add_argument(
        "--vu_decouple",    action="store_true",  default=True,
        help="啟用 V-U 解耦 mbest（預設開啟）",
    )
    p.add_argument(
        "--no_vu_decouple", action="store_false", dest="vu_decouple",
        help="關閉 V-U 解耦 mbest",
    )
    p.add_argument("--w_vu", type=float, default=0.70,
                   help="V-U 解耦 mbest 中標準 U 形加權的權重")
    p.add_argument("--w_v",  type=float, default=0.15,
                   help="V-U 解耦 mbest 中 V* 位置的牽引權重")
    p.add_argument("--w_u",  type=float, default=0.15,
                   help="V-U 解耦 mbest 中 U* 位置的牽引權重")
    p.add_argument("--min_u_for_v_track", type=float, default=0.50,
                   help="更新 V* 牽引位置時要求的最低 uniqueness 門檻")
    p.add_argument("--min_v_for_u_track", type=float, default=0.50,
                   help="更新 U* 牽引位置時要求的最低 validity 門檻")

    # ── ★ v10.3(V8) mode collapse 防護/回收 ──────────────────────────────
    p.add_argument("--mode_collapse_u_thresh", type=float, default=0.20,
                   help="uniqueness 低於此值的粒子視為 mode collapse："
                        "不更新 pbest，並於下一迭代開頭重置至 gbest 鄰域")

    # ── ★ v10.4 HBA/HBD 量測（opt-in，純記錄，不改變 V×U 最適化）──────────
    #   對齊 qiskit 參考 log（chemistry_constraint 4HBA/3HBD）的量測。
    #   兩者皆為 None 時功能關閉，執行流程與 v10.3 完全相同（向後相容）。
    #   任一設定即開啟：worker 會計算平均 HBA/HBD，主行程逐迭代以
    #   qiskit 相容格式記錄「最佳粒子」的 V×U / HBA / HBD，並另存 CSV。
    #   ⚠ HBA/HBD 僅為量測指標，不進入 fitness、不改變 QPSO 演算法。
    p.add_argument("--hba_target", type=float, default=None,
                   help="HBA 量測目標（參考 log 為 4）。設定即開啟 HBA/HBD 量測記錄。")
    p.add_argument("--hbd_target", type=float, default=None,
                   help="HBD 量測目標（參考 log 為 3）。設定即開啟 HBA/HBD 量測記錄。")

    # ── 輸出設定 ──────────────────────────────────────────────────────────
    p.add_argument("--task_name", type=str,
                   default="unconditional_9_ae_v6_sobol_obl")
    p.add_argument("--data_dir",  type=str, default="results_v6")
    return p.parse_args()


# ===========================================================================
# Logger
# ===========================================================================

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("SobolOBLQPSOLogger")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s,%(msecs)03d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for h in [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ===========================================================================
# GPU 資訊輸出
# ===========================================================================

def log_gpu_info(logger: logging.Logger, gpu_ids: list) -> None:
    try:
        import subprocess as _sp
        out = _sp.check_output(
            "nvidia-smi --query-gpu=index,name,memory.total,driver_version "
            "--format=csv,noheader",
            shell=True, stderr=_sp.DEVNULL,
        ).decode().strip()
        for line in out.splitlines():
            logger.info(f"  GPU: {line.strip()}")
    except Exception:
        logger.info("  GPU info: 無法取得")
    try:
        import re
        ver_str = cudaq.__version__
        match = re.search(r'(\d+\.\d+\.\d+)', ver_str)
        logger.info(f"  CUDA-Q: {match.group(1) if match else ver_str}")
    except Exception:
        pass
    logger.info(f"  分配 GPU IDs: {gpu_ids}")
    logger.info(f"  並行數 N_GPUS: {len(gpu_ids)}")


# ===========================================================================
# ★ v10.4 新增：HBA/HBD 量測記錄器（純記錄，完全不觸碰 optimizer / V×U）
# ===========================================================================

class HBAHBDRecorder:
    """
    以 qiskit 參考 log 相容格式記錄每次評估批次的 HBA / HBD 量測。

    設計原則（對應使用者需求「不更改架構，只加必要記錄」）：
      - optimizer（qpso_optimizer_ae.py）完全不動；仍只收到 (V, U)。
      - 本記錄器掛在評估函式（分子生成的唯一位置）之後，
        利用 worker 額外回傳的 arr[2]=HBA、arr[3]=HBD 做記錄。
      - 逐「批次」對齊 optimizer 的呼叫序列，還原 qiskit 的
        「Iteration number: N」逐代編號：
            batch mode + OBL : call0=Iter0, call1=OBL, call2..=Iter1..T
            batch mode 無 OBL: call0=Iter0, call1..=Iter1..T
      - 每代回報「該批次 V×U 最佳粒子」的 V×U / HBA / HBD（對應
        qiskit 每代單一候選電路的量測），並另存逐代 CSV。

    ⚠ HBA/HBD 不進入 fitness、不影響 QPSO 更新，僅供驗證用途。
    """

    def __init__(self, args, logger, obl_enabled: bool):
        self.logger      = logger
        self.hba_target  = args.hba_target
        self.hbd_target  = args.hbd_target
        self.obl_enabled = obl_enabled
        self.call_idx    = 0
        self.csv_path    = os.path.join(
            args.data_dir, f"{args.task_name}_hbahbd.csv"
        )
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write(
                "iter_label,phase,best_particle,"
                "product_validity_uniqueness,HBA,HBD,"
                "batch_mean_HBA,batch_mean_HBD\n"
            )

    def _hba_tag(self) -> str:
        return f" (close to {self.hba_target:g})" if self.hba_target is not None else ""

    def _hbd_tag(self) -> str:
        return f" (close to {self.hbd_target:g})" if self.hbd_target is not None else ""

    def report_batch(self, vu_list, hbahbd_list) -> None:
        """
        vu_list:     list[(v, u)]        —— 與傳給 optimizer 的內容一致
        hbahbd_list: list[(hba, hbd)]    —— worker 回傳的量測欄位
        """
        idx = self.call_idx
        self.call_idx += 1

        # ── 還原 qiskit 逐代編號 ──
        if idx == 0:
            phase, iter_label = "phase0", 0
        elif self.obl_enabled and idx == 1:
            phase, iter_label = "obl", -1          # OBL 批次，不佔用迭代編號
        else:
            phase = "iter"
            iter_label = idx - (1 if self.obl_enabled else 0)

        M = len(vu_list)
        if M == 0:
            return

        # ── 該批次 V×U 最佳粒子（對應 qiskit 每代單一候選）──
        vu_scores = [float(v) * float(u) for (v, u) in vu_list]
        best_i    = int(np.argmax(vu_scores))
        best_vu   = vu_scores[best_i]
        best_hba, best_hbd = hbahbd_list[best_i]

        # ── 批次內有效粒子（HBA 或 HBD > 0）的平均，作為分布觀察值 ──
        valid = [(h, d) for (h, d) in hbahbd_list if (h > 0 or d > 0)]
        if valid:
            mean_hba = sum(h for h, _ in valid) / len(valid)
            mean_hbd = sum(d for _, d in valid) / len(valid)
        else:
            mean_hba, mean_hbd = 0.0, 0.0

        # ── qiskit 相容格式輸出（best particle）──
        if phase == "obl":
            self.logger.info("[HBA/HBD 量測] OBL 批次（不計入迭代編號）")
            self.logger.info(f"product_validity_uniqueness (maximize): {best_vu:.3f}")
            self.logger.info(f"HBA{self._hba_tag()}: {best_hba:.3f}")
            self.logger.info(f"HBD{self._hbd_tag()}: {best_hbd:.3f}")
        else:
            self.logger.info(f"[HBA/HBD 量測] Iteration number: {iter_label}")
            self.logger.info(f"product_validity_uniqueness (maximize): {best_vu:.3f}")
            self.logger.info(f"HBA{self._hba_tag()}: {best_hba:.3f}")
            self.logger.info(f"HBD{self._hbd_tag()}: {best_hbd:.3f}")
            self.logger.info(
                f"  [批次分布] mean_HBA={mean_hba:.3f}  mean_HBD={mean_hbd:.3f}  "
                f"(best particle #{best_i}, valid {len(valid)}/{M})"
            )

        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{iter_label},{phase},{best_i},"
                f"{best_vu:.6f},{best_hba:.4f},{best_hbd:.4f},"
                f"{mean_hba:.4f},{mean_hbd:.4f}\n"
            )


# ===========================================================================
# ★ v10.1 保留：parallel subprocess batch evaluate function
# ===========================================================================

def make_parallel_batch_evaluate_fn(
    args:          argparse.Namespace,
    cwg:           ConditionalWeightsGenerator,
    logger:        logging.Logger,
    worker_script: str,
    gpu_ids:       list,
    recorder:      "HBAHBDRecorder" = None,   # ★ v10.4：None 時行為與舊版相同
) -> callable:
    """
    parallel subprocess 批次評估函式（v10.1 不變）。

    每輪同時啟動 min(n_gpus, remaining) 個子行程，
    每個子行程：
      1. 父行程預設 CUDA_VISIBLE_DEVICES=<gpu_id>（在 CUDA 初始化前）
      2. 執行 worker_eval.py → cudaq.sample() → 輸出 V, U
      3. 子行程結束 → CUDA context 銷毀 → pinned memory 完全釋放
    """
    n_gpus     = len(gpu_ids)
    pythonpath = os.environ.get("PYTHONPATH", ".")
    eval_count = [0]

    report_on = recorder is not None            # ★ v10.4 HBA/HBD 量測開關

    def batch_evaluate_fn(positions: np.ndarray) -> list:
        M = positions.shape[0]
        results: list = [(0.0, 0.0)] * M
        hbahbd: list  = [(0.0, 0.0)] * M         # ★ v10.4：與 results 平行的量測欄位
        t_batch_start = time.time()

        n_rounds = (M + n_gpus - 1) // n_gpus
        for round_idx in range(n_rounds):
            round_start = round_idx * n_gpus
            round_end   = min(round_start + n_gpus, M)
            round_pids  = list(range(round_start, round_end))
            round_size  = len(round_pids)

            t_round = time.time()
            procs:        list = []
            weight_paths: list = []

            for local_i, particle_idx in enumerate(round_pids):
                gpu_id_str = str(gpu_ids[local_i % n_gpus])
                uid        = uuid.uuid4().hex[:8]
                wpath      = os.path.join(tempfile.gettempdir(), f"qmg_pw_{uid}.npy")
                rpath      = os.path.join(tempfile.gettempdir(), f"qmg_pr_{uid}.npy")

                w_c = cwg.apply_chemistry_constraint(positions[particle_idx].copy())
                np.save(wpath, w_c)

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_id_str
                env["PYTHONPATH"]           = pythonpath

                cmd = [
                    sys.executable,
                    worker_script,
                    "--weight_path",    wpath,
                    "--result_path",    rpath,
                    "--num_heavy_atom", str(args.num_heavy_atom),
                    "--num_sample",     str(args.num_sample),
                    "--backend",        args.backend,
                ]
                if report_on:                    # ★ v10.4：僅在量測模式時計算 HBA/HBD
                    cmd.append("--report_hbahbd")

                proc = subprocess.Popen(
                    cmd,
                    env    = env,
                    stdout = subprocess.DEVNULL,
                    stderr = subprocess.PIPE,
                )
                procs.append((proc, rpath, particle_idx, gpu_id_str))
                weight_paths.append(wpath)
                eval_count[0] += 1

            for proc, rpath, particle_idx, gpu_id_str in procs:
                try:
                    _, stderr_bytes = proc.communicate(
                        timeout=args.subprocess_timeout
                    )
                    if proc.returncode == 0:
                        arr = np.load(rpath)
                        results[particle_idx] = (float(arr[0]), float(arr[1]))
                        if report_on and len(arr) >= 4:   # ★ v10.4 量測欄位
                            hbahbd[particle_idx] = (float(arr[2]), float(arr[3]))
                    else:
                        msg = stderr_bytes.decode("utf-8", errors="replace")[-400:]
                        logger.warning(
                            f"[parallel] GPU {gpu_id_str} 粒子 {particle_idx} "
                            f"exit={proc.returncode}\n  stderr: {msg}"
                        )
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    logger.warning(
                        f"[parallel] GPU {gpu_id_str} 粒子 {particle_idx} "
                        f"逾時（>{args.subprocess_timeout}s）"
                    )
                except Exception as e:
                    logger.warning(
                        f"[parallel] GPU {gpu_id_str} 粒子 {particle_idx} "
                        f"例外：{e}"
                    )
                finally:
                    try:
                        os.remove(rpath)
                    except FileNotFoundError:
                        pass

            for wp in weight_paths:
                try:
                    os.remove(wp)
                except FileNotFoundError:
                    pass

            round_elapsed  = time.time() - t_round
            valid_in_round = sum(1 for idx in round_pids if results[idx][0] > 0)
            logger.info(
                f"  [parallel 輪次 {round_idx+1}/{n_rounds}] "
                f"粒子 {round_start}..{round_end-1}  "
                f"GPU: {[str(gpu_ids[i % n_gpus]) for i in range(round_size)]}  "
                f"有效:{valid_in_round}/{round_size}  "
                f"本輪:{round_elapsed:.1f}s  "
                f"累計:{time.time()-t_batch_start:.1f}s"
            )

        # ★ v10.4：批次結束後記錄 HBA/HBD（不影響回傳給 optimizer 的 results）
        if report_on:
            try:
                recorder.report_batch(results, hbahbd)
            except Exception as e:
                logger.warning(f"[HBA/HBD 量測] 記錄失敗（不影響最適化）：{e}")

        return results

    return batch_evaluate_fn


# ===========================================================================
# 單 GPU 序列模式（v10.1 保留）
# ===========================================================================

def make_subprocess_evaluate_fn(
    args:          argparse.Namespace,
    cwg:           ConditionalWeightsGenerator,
    logger:        logging.Logger,
    worker_script: str,
    gpu_id:        str,
) -> callable:
    pythonpath = os.environ.get("PYTHONPATH", ".")
    eval_count = [0]

    def evaluate_fn(pos: np.ndarray) -> tuple:
        idx = eval_count[0]
        eval_count[0] += 1
        uid   = uuid.uuid4().hex[:8]
        wpath = os.path.join(tempfile.gettempdir(), f"qmg_w_{uid}.npy")
        rpath = os.path.join(tempfile.gettempdir(), f"qmg_r_{uid}.npy")
        try:
            w_c = cwg.apply_chemistry_constraint(pos.copy())
            np.save(wpath, w_c)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["PYTHONPATH"]           = pythonpath
            cmd = [
                sys.executable, worker_script,
                "--weight_path",    wpath,
                "--result_path",    rpath,
                "--num_heavy_atom", str(args.num_heavy_atom),
                "--num_sample",     str(args.num_sample),
                "--backend",        args.backend,
            ]
            t0  = time.time()
            ret = subprocess.run(
                cmd, env=env,
                timeout=args.subprocess_timeout,
                capture_output=True,
            )
            elapsed = time.time() - t0
            if ret.returncode != 0:
                msg = ret.stderr.decode("utf-8", errors="replace")[-400:]
                logger.warning(f"[single] eval #{idx} 失敗 ({elapsed:.1f}s)\n{msg}")
                return 0.0, 0.0
            arr = np.load(rpath)
            return float(arr[0]), float(arr[1])
        except subprocess.TimeoutExpired:
            logger.warning(f"[single] eval #{idx} 逾時")
            return 0.0, 0.0
        except Exception as e:
            logger.warning(f"[single] eval #{idx} 例外：{e}")
            return 0.0, 0.0
        finally:
            for path in [wpath, rpath]:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

    return evaluate_fn


# ===========================================================================
# 並行 worker 功能驗證（v10.1 保留）
# ===========================================================================

def verify_workers_parallel(
    args:          argparse.Namespace,
    cwg:           ConditionalWeightsGenerator,
    logger:        logging.Logger,
    worker_script: str,
    gpu_ids:       list,
) -> bool:
    logger.info(
        f"[v10.3] 並行功能驗證：同時啟動 {len(gpu_ids)} 個子行程（各 5 shots）..."
    )
    pythonpath = os.environ.get("PYTHONPATH", ".")
    w_test = cwg.generate_conditional_random_weights(random_seed=99)
    procs  = []
    paths  = []
    t0     = time.time()

    for gpu_id_str in [str(g) for g in gpu_ids]:
        uid   = uuid.uuid4().hex[:8]
        wpath = os.path.join(tempfile.gettempdir(), f"qmg_tv_w_{uid}.npy")
        rpath = os.path.join(tempfile.gettempdir(), f"qmg_tv_r_{uid}.npy")
        np.save(wpath, w_test)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id_str
        env["PYTHONPATH"]           = pythonpath
        cmd = [
            sys.executable, worker_script,
            "--weight_path",    wpath,
            "--result_path",    rpath,
            "--num_heavy_atom", str(args.num_heavy_atom),
            "--num_sample",     "5",
            "--backend",        args.backend,
        ]
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        procs.append((proc, rpath, gpu_id_str))
        paths.append(wpath)

    all_ok = True
    for proc, rpath, gpu_id_str in procs:
        try:
            _, stderr_bytes = proc.communicate(timeout=180)
            if proc.returncode == 0:
                arr = np.load(rpath)
                logger.info(f"  GPU {gpu_id_str}: V={arr[0]:.3f}  U={arr[1]:.3f}  ✓")
            else:
                msg = stderr_bytes.decode("utf-8", errors="replace")[-300:]
                logger.error(f"  GPU {gpu_id_str}: 子行程失敗 ✗\n  {msg}")
                all_ok = False
        except subprocess.TimeoutExpired:
            proc.kill()
            logger.error(f"  GPU {gpu_id_str}: 逾時 ✗")
            all_ok = False
        except Exception as e:
            logger.error(f"  GPU {gpu_id_str}: {e} ✗")
            all_ok = False
        finally:
            try:
                os.remove(rpath)
            except FileNotFoundError:
                pass

    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

    elapsed = time.time() - t0
    logger.info(
        f"[v10.3] 並行驗證完成（{elapsed:.1f}s）  "
        f"{'✓ 所有 GPU 正常' if all_ok else '✗ 有 GPU 失敗'}"
    )
    return all_ok


# ===========================================================================
# 主程式
# ===========================================================================

def main() -> None:
    args = parse_args()

    gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
    effective_n_gpus = min(args.n_gpus, len(gpu_ids))
    gpu_ids = gpu_ids[:effective_n_gpus]

    os.makedirs(args.data_dir, exist_ok=True)
    log_path = os.path.join(args.data_dir, f"{args.task_name}.log")
    logger   = setup_logger(log_path)

    # ★ v10.4：HBA/HBD 量測開關（任一 target 設定即開啟；預設關閉＝完全向後相容）
    report_hbahbd = (args.hba_target is not None) or (args.hbd_target is not None)

    # ── 基本資訊記錄 ─────────────────────────────────────────────────────
    logger.info(f"Task name: {args.task_name}")
    if report_hbahbd:
        # 對齊 qiskit 參考 log 的 chemistry_constraint 標頭格式，方便逐項對比。
        # 注意：optimizer 目標仍僅為 product_validity_uniqueness（maximize）；
        #       HBA/HBD 為 measure/report 欄位，不進入 fitness、不改變演算法。
        hba_cond = "None" if args.hba_target is None else f"{args.hba_target:g}"
        hbd_cond = "None" if args.hbd_target is None else f"{args.hbd_target:g}"
        logger.info(f"Task: ['product_validity_uniqueness', 'HBA', 'HBD']")
        logger.info(f"Condition: ['None', '{hba_cond}', '{hbd_cond}']")
        logger.info(f"objective: ['maximize', 'measure', 'measure']")
        logger.info(
            "  ⚠ HBA/HBD 為量測/記錄欄位（非最適化目標）。"
            "QPSO 目標與 v10.3 完全相同：maximize V×U。"
        )
    else:
        logger.info(f"Task: ['validity', 'uniqueness']")
        logger.info(f"Condition: ['None', 'None']")
        logger.info(f"objective: ['maximize', 'maximize']")
    logger.info(f"# of heavy atoms: {args.num_heavy_atom}")
    logger.info(f"# of samples: {args.num_sample}  "
                f"(v10.2 預設 5000，與 Chen 2025 對齊，birthday paradox 修正)")
    logger.info(f"smarts: None")
    logger.info(f"disable_connectivity_position: []")
    logger.info(f"CUDA-Q backend: {args.backend}")
    logger.info(
        f"[v10.3] 初始化策略: "
        f"{'Sobol scrambled (seed=0, 確定性)' if args.sobol_init else f'pseudo-random (seed={args.seed})'}"
    )
    logger.info(
        f"[v10.3] OBL Phase 0: {'✓ 開啟' if args.obl else '✗ 關閉'}"
    )
    logger.info(
        f"[v10.3] V-U 解耦 mbest: "
        f"{'✓ 開啟 (w_vu={:.2f}, w_v={:.2f}, w_u={:.2f}, U_gate={:.2f}, V_gate={:.2f})'.format(args.w_vu, args.w_v, args.w_u, args.min_u_for_v_track, args.min_v_for_u_track) if args.vu_decouple else '✗ 關閉'}"
    )
    logger.info(
        f"[v10.1→v10.2] 評估模式: parallel subprocess pool  "
        f"N_GPUS={effective_n_gpus}  GPU_IDs={gpu_ids}"
    )
    logger.info(f"[v10.3] subprocess_timeout: {args.subprocess_timeout}s")
    log_gpu_info(logger, gpu_ids)
    log_memory(logger, "啟動時")

    # ── 確認 worker_eval.py 存在 ──────────────────────────────────────────
    script_dir    = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(script_dir, "worker_eval.py")
    if not os.path.exists(worker_script):
        logger.error(
            f"[ERROR] worker_eval.py 不存在：{worker_script}\n"
            f"  請確認 worker_eval.py 與 run_qpso_qmg_cudaq.py 在同一目錄。"
        )
        sys.exit(1)
    logger.info(f"  worker_eval.py: {worker_script} ✓")

    # ── 初始化 ConditionalWeightsGenerator ───────────────────────────────
    cwg = ConditionalWeightsGenerator(
        args.num_heavy_atom,
        smarts=None,
        disable_connectivity_position=[],
    )
    n_flexible = int((cwg.parameters_indicator == 0.0).sum())
    logger.info(f"Number of flexible parameters: {n_flexible}")
    assert n_flexible == cwg.length_all_weight_vector

    # ── 預估時間 ──────────────────────────────────────────────────────────
    # num_sample=5000 時每次評估約 142s（V3 的 284s 一半）
    sec_per_eval    = 142
    rounds_per_iter = (args.particles + effective_n_gpus - 1) // effective_n_gpus
    # OBL 多一個批次
    obl_batches     = 1 if args.obl else 0
    total_batches   = (args.iterations + 1) + obl_batches
    est_h           = total_batches * rounds_per_iter * sec_per_eval / 3600
    total_evals     = args.particles * (args.iterations + 1) + (args.particles if args.obl else 0)
    logger.info(
        f"[v10.2 config] M={args.particles}  T={args.iterations}  "
        f"total_evals≈{total_evals}  "
        f"每批次 {rounds_per_iter} 輪 × {effective_n_gpus} GPU  "
        f"預估：{est_h:.1f}h  "
        f"(num_sample={args.num_sample}，~{sec_per_eval}s/eval)"
    )

    # ── 並行 worker 功能驗證 ──────────────────────────────────────────────
    if not verify_workers_parallel(args, cwg, logger, worker_script, gpu_ids):
        logger.error(
            "[ERROR] 並行驗證失敗。請先確認單 GPU 正常：\n"
            "  python run_qpso_qmg_cudaq.py --n_gpus 1 --gpu_ids 0 "
            "--particles 8 --iterations 1 --num_sample 100"
        )
        sys.exit(1)
    log_memory(logger, "並行驗證後")

    # ── 建立評估函式 ──────────────────────────────────────────────────────
    if effective_n_gpus == 1:
        evaluate_fn       = make_subprocess_evaluate_fn(
            args=args, cwg=cwg, logger=logger,
            worker_script=worker_script,
            gpu_id=str(gpu_ids[0]),
        )
        batch_evaluate_fn = None
        logger.info(f"[v10.3] 使用 單GPU 序列模式（GPU {gpu_ids[0]}）")
    else:
        evaluate_fn       = None
        # ★ v10.4：僅在量測模式（8-GPU 並行）建立 HBA/HBD 記錄器
        recorder = None
        if report_hbahbd:
            recorder = HBAHBDRecorder(args, logger, obl_enabled=args.obl)
            logger.info(
                f"[v10.4] HBA/HBD 量測記錄：✓ 開啟  "
                f"(HBA target={args.hba_target}, HBD target={args.hbd_target})  "
                f"CSV={recorder.csv_path}"
            )
        batch_evaluate_fn = make_parallel_batch_evaluate_fn(
            args=args, cwg=cwg, logger=logger,
            worker_script=worker_script,
            gpu_ids=gpu_ids,
            recorder=recorder,
        )
        logger.info(
            f"[v10.3] 使用 {effective_n_gpus}-GPU 並行模式  GPU IDs: {gpu_ids}"
        )

    # ── 建立 AESOQPSOOptimizer v1.5 (V8) ──────────────────────────────────
    optimizer = AESOQPSOOptimizer(
        n_params           = n_flexible,
        n_particles        = args.particles,
        max_iterations     = args.iterations,
        logger             = logger,
        evaluate_fn        = evaluate_fn,
        batch_evaluate_fn  = batch_evaluate_fn,
        seed               = args.seed,
        alpha_max          = args.alpha_max,
        alpha_min          = args.alpha_min,
        data_dir           = args.data_dir,
        task_name          = args.task_name,
        stagnation_limit   = args.stagnation_limit,
        reinit_fraction    = args.reinit_fraction,
        mutation_prob      = args.mutation_prob,
        ae_weighting       = args.ae_weighting,
        pair_interval      = args.pair_interval,
        rotate_factor      = args.rotate_factor,
        # ★ v1.2 新增參數
        obl                = args.obl,
        vu_decouple        = args.vu_decouple,
        w_vu               = args.w_vu,
        w_v                = args.w_v,
        w_u                = args.w_u,
        min_u_for_v_track  = args.min_u_for_v_track,
        min_v_for_u_track  = args.min_v_for_u_track,
        # ★ v10.3(V8) 新增參數
        mode_collapse_u_thresh = args.mode_collapse_u_thresh,
    )

    # ── ★ v10.2：Sobol 初始化覆寫粒子位置 ────────────────────────────────
    if args.sobol_init:
        sobol_pos = make_sobol_positions(args.particles, n_flexible, logger)
        if sobol_pos is not None:
            optimizer.positions = sobol_pos.copy()
            optimizer.pbest     = sobol_pos.copy()
            # pbest_fit 保持 -inf → Phase 0 評估後正常建立
            logger.info(
                f"[Sobol v10.2] 粒子初始位置已覆寫  "
                f"shape={optimizer.positions.shape}  "
                f"range=[{optimizer.positions.min():.4f}, "
                f"{optimizer.positions.max():.4f}]"
            )
        else:
            logger.warning(
                "[Sobol] scipy 安裝失敗，使用 pseudo-random 初始化（seed={}）".format(args.seed)
            )
    else:
        logger.info(
            f"[v10.3] 使用 pseudo-random 初始化  seed={args.seed}"
        )

    # ── 執行優化 ──────────────────────────────────────────────────────────
    log_memory(logger, "優化開始前")
    try:
        best_params, best_fitness = optimizer.optimize()
    except Exception as e:
        logger.error(f"[ERROR] optimizer.optimize() 異常：{e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        log_memory(logger, "優化結束後")

    # ── 儲存最佳參數 ──────────────────────────────────────────────────────
    best_npy = os.path.join(args.data_dir, f"{args.task_name}_best_params.npy")
    np.save(best_npy, best_params)
    logger.info(f"最佳參數已儲存: {best_npy}")
    logger.info(
        f"最終結果: V×U={best_fitness:.6f}  "
        + ("✓ 超越 BO 基線 0.8834!" if best_fitness > 0.8834
           else f"✗ 未超越 — 差距 {0.8834 - best_fitness:.4f}")
    )
    log_memory(logger, "程序結束前")


if __name__ == "__main__":
    main()
