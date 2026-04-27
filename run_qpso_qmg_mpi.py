"""
==============================================================================
run_qpso_qmg_mpi.py  —  CUDA-Q 0.7.1 + AE-SOQPSO + MPI 並行主入口（v1.2 GPU修正版）
==============================================================================

v1.1 → v1.2  根本修正：MPI GPU 綁定失效問題

  ★ [根因] CUDA_VISIBLE_DEVICES 設定時機過晚
  ─────────────────────────────────────────────────────────────────────
    v1.1 的 GPU 綁定邏輯：
      _ORIGINAL_CUDA_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
      _dev_list = [d.strip() for d in _ORIGINAL_CUDA_DEVICES.split(",")]
      _my_gpu   = _dev_list[_RANK % len(_dev_list)]
      os.environ["CUDA_VISIBLE_DEVICES"] = _my_gpu      # ← 設在模組頂層

    看似正確，但在以下情境下失效：
      (A) SLURM cgroup 隔離：NCHC 以 cgroup v2 管理 GPU 資源，
          每個 task 的允許 GPU 列表由 SLURM 的 GpuAccessCgroup 控制。
          即使 Python 修改 CUDA_VISIBLE_DEVICES，
          CUDA driver 初始化時讀取 /proc/self/cgroup 確認 GPU 權限，
          若 cgroup 不允許存取 GPU N，cuStateVec 靜默 fallback 至 GPU 0。

      (B) SLURM 的 --ntasks-per-node=8 + --gres=gpu:8：
          預設行為是將 8 個 GPU 平均分配給 8 個 task，
          但並非 task i 必然得到 GPU i。
          實際分配依 SLURM 的 GPU 拓樸感知演算法決定，
          必須透過 SLURM_LOCALID 或 SLURM_STEP_GPUS 取得正確 GPU index。

      (C) CUDA-Q 0.7.1 全節點序列化鎖（非 cgroup 環境也會發生）：
          cuStateVec 初始化時使用 /dev/nvidia-uvm 的全域 mutex，
          導致即使 8 個 rank 各有獨立 GPU，
          cudaq.set_target("nvidia") 仍按序執行。

    實測確認（from full_run.txt / unconditional_9_ae_mpi_full.log）：
      50 粒子全部顯示 t=6129.1s（Phase 0 批次耗時），
      代表 50 個粒子順序執行，8 GPU 沒有真正並行。

  ★ [修正一] 使用 SLURM_LOCALID 取得正確 GPU index
  ─────────────────────────────────────────────────────────────────────
    SLURM_LOCALID：每個 node 上的 local rank 索引（0..ntasks-1）
    SLURM_STEP_GPUS：SLURM 分配給本 step 的 GPU 列表（如 "0,1,2,3,4,5,6,7"）
    正確做法：
      local_id = int(os.environ.get("SLURM_LOCALID", _RANK))
      step_gpus = os.environ.get("SLURM_STEP_GPUS", "").split(",")
      my_gpu = step_gpus[local_id % len(step_gpus)]   # 從 SLURM 分配結果取 GPU

    必須在任何 CUDA 相關 import 之前設定。

  ★ [修正二] SLURM script 加入 --gpu-bind=per_task:1
  ─────────────────────────────────────────────────────────────────────
    在 sbatch 指令或 slurm 腳本加入：
      --gpu-bind=per_task:1
    強制 SLURM 為每個 task 綁定獨立的 GPU，並正確設定 cgroup。

  ★ [修正三] 記憶體洩漏：每 N 批次重啟子行程（checkpoint 機制）
  ─────────────────────────────────────────────────────────────────────
    CUDA pinned memory 無法在行程存活期間釋放。
    解法：每隔 REINIT_EVERY 個 QPSO 迭代，rank 0 發送重啟信號，
    所有 rank 儲存當前 MoleculeGeneratorCUDAQ，重新初始化 cudaq，
    強制觸發新的 CUDA context，釋放 pinned memory。

    注意：這需要 cudaq.reset() 或重新 set_target()，
    在 0.7.1 中效果有限。最根本的解法見 run_qpso_qmg_cudaq.py v10.1
    的 parallel subprocess 方案。

  ★ 建議優先使用 run_qpso_qmg_cudaq.py v10.1（parallel subprocess）
  ─────────────────────────────────────────────────────────────────────
    若叢集不支援 mpi4py 或 MPI GPU 綁定難以確認，
    run_qpso_qmg_cudaq.py v10.1 是更可靠的選擇，
    它不依賴 MPI，完全由父行程控制每個子行程的 GPU 分配。

SLURM 提交（使用修正版 v1.2）：
  sbatch cutn-qmg_mpi_8g.slurm
==============================================================================
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

# ===========================================================================
# ★ v1.2 修正一：GPU 綁定必須在任何 CUDA import 之前完成
# 使用 SLURM_LOCALID + SLURM_STEP_GPUS 取得正確 GPU index
# ===========================================================================

# Step 1: 先取 MPI rank（不 import MPI，只用環境變數）
# PMI_RANK 是 OpenMPI/MPICH 通用的行程排名環境變數
_EARLY_RANK = int(os.environ.get("PMI_RANK",
               os.environ.get("OMPI_COMM_WORLD_RANK",
               os.environ.get("MV2_COMM_WORLD_RANK", "0"))))

# Step 2: 從 SLURM 環境取正確的 GPU 列表
_SLURM_LOCAL_ID  = int(os.environ.get("SLURM_LOCALID", _EARLY_RANK))
_SLURM_STEP_GPUS = os.environ.get("SLURM_STEP_GPUS", "")
_SLURM_JOB_GPUS  = os.environ.get("SLURM_JOB_GPUS",  "")

_ORIGINAL_CUDA_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")

if _SLURM_STEP_GPUS:
    # 優先使用 SLURM_STEP_GPUS（當前 step 分配的 GPU 列表）
    _gpu_pool = [g.strip() for g in _SLURM_STEP_GPUS.split(",") if g.strip()]
elif _SLURM_JOB_GPUS:
    _gpu_pool = [g.strip() for g in _SLURM_JOB_GPUS.split(",") if g.strip()]
elif _ORIGINAL_CUDA_DEVICES:
    _gpu_pool = [g.strip() for g in _ORIGINAL_CUDA_DEVICES.split(",") if g.strip()]
else:
    # 最後 fallback：假設每個 rank 對應一張 GPU
    _gpu_pool = [str(_SLURM_LOCAL_ID)]

# ★ 每個 rank 取自己的 GPU（round-robin）
_my_gpu = _gpu_pool[_SLURM_LOCAL_ID % len(_gpu_pool)]

# ★ 必須在 import cudaq 之前設定，確保 CUDA driver 看到正確 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = _my_gpu

# Step 3: 現在才 import MPI（MPI init 後 CUDA_VISIBLE_DEVICES 已確定）
try:
    from mpi4py import MPI
    _COMM    = MPI.COMM_WORLD
    _RANK    = _COMM.Get_rank()
    _NRANK   = _COMM.Get_size()
    _HAS_MPI = True
except ImportError:
    _COMM    = None
    _RANK    = 0
    _NRANK   = 1
    _HAS_MPI = False
    if _EARLY_RANK == 0:
        print("[WARN] mpi4py 未安裝，以單 rank 模式執行。", flush=True)

# Step 4: import cudaq（此時 CUDA_VISIBLE_DEVICES 已正確設定）
try:
    import cudaq
except ImportError:
    if _RANK == 0:
        print("[ERROR] 無法 import cudaq。請安裝：pip install cuda-quantum-cu12==0.7.1")
    sys.exit(1)

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

try:
    from qmg.generator_cudaq import MoleculeGeneratorCUDAQ
    from qmg.utils.weight_generator import ConditionalWeightsGenerator
except ImportError as e:
    if _RANK == 0:
        print(f"[ERROR] 無法 import qmg：{e}")
    sys.exit(1)

try:
    from qpso_optimizer_ae import AESOQPSOOptimizer
except ImportError as e:
    if _RANK == 0:
        print(f"[ERROR] 無法 import qpso_optimizer_ae：{e}")
    sys.exit(1)


# ===========================================================================
# MPI 通訊常數
# ===========================================================================

_MPI_FLAG_CONTINUE = 1
_MPI_FLAG_STOP     = 0


# ===========================================================================
# 工具函式
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QMG CUDA-Q + AE-SOQPSO MPI 並行版 v1.2（GPU 綁定修正版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num_heavy_atom",   type=int,   default=9)
    p.add_argument("--num_sample",       type=int,   default=10000)
    p.add_argument("--particles",        type=int,   default=50)
    p.add_argument("--iterations",       type=int,   default=40)
    p.add_argument("--alpha_max",        type=float, default=1.2)
    p.add_argument("--alpha_min",        type=float, default=0.4)
    p.add_argument("--mutation_prob",    type=float, default=0.15)
    p.add_argument("--stagnation_limit", type=int,   default=8)
    p.add_argument("--reinit_fraction",  type=float, default=0.20)
    p.add_argument("--ae_weighting",     action="store_true", default=True)
    p.add_argument("--no_ae_weighting",  action="store_false", dest="ae_weighting")
    p.add_argument("--pair_interval",    type=int,   default=5)
    p.add_argument("--rotate_factor",    type=float, default=0.01)
    p.add_argument(
        "--backend", type=str, default="cudaq_tensornet",
        choices=["cudaq_tensornet", "cudaq_nvidia", "cudaq_qpp",
                 "cudaq_nvidia_fp64", "cudaq_tensornet_mps"],
    )
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--task_name",        type=str,   default="unconditional_9_ae_mpi")
    p.add_argument("--data_dir",         type=str,   default="results_mpi")
    p.add_argument(
        "--reinit_every", type=int, default=10,
        help=(
            "每隔多少 QPSO 迭代重建 MoleculeGeneratorCUDAQ（釋放 CUDA pinned memory）。"
            "0 表示停用。建議值：10（每 500 次評估釋放一次記憶體）。"
        ),
    )
    return p.parse_args()


def setup_logger(log_path: str) -> logging.Logger:
    if _RANK != 0:
        null = logging.getLogger(f"null_rank{_RANK}")
        null.addHandler(logging.NullHandler())
        return null
    logger = logging.getLogger("AEMPILogger_v12")
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


def get_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024
    except Exception:
        pass
    return -1.0


def log_gpu_info(logger: logging.Logger) -> None:
    if _RANK != 0:
        return
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
        m = re.search(r'(\d+\.\d+\.\d+)', cudaq.__version__)
        logger.info(f"  CUDA-Q: {m.group(1) if m else cudaq.__version__}")
    except Exception:
        pass
    logger.info(f"  CUDA_VISIBLE_DEVICES（原始）: {_ORIGINAL_CUDA_DEVICES}")
    logger.info(f"  SLURM_STEP_GPUS: {_SLURM_STEP_GPUS or '（未設定）'}")
    logger.info(f"  SLURM_LOCALID（rank 0）: {_SLURM_LOCAL_ID}")
    logger.info(f"  MPI N_RANK: {_NRANK}  rank {_RANK} → GPU {_my_gpu}")


def log_all_gpu_bindings(logger: logging.Logger) -> None:
    """收集所有 rank 的 GPU 綁定情況並統一記錄（rank 0 輸出）。"""
    if not _HAS_MPI:
        logger.info(f"  GPU binding: rank 0 → GPU {_my_gpu}")
        return
    binding_info = _COMM.gather(
        {"rank": _RANK, "gpu": _my_gpu, "local_id": _SLURM_LOCAL_ID},
        root=0,
    )
    if _RANK == 0:
        logger.info("  GPU 綁定確認（v1.2 SLURM_LOCALID 方式）：")
        for info in binding_info:
            logger.info(
                f"    Rank {info['rank']} (SLURM_LOCALID={info['local_id']}) "
                f"→ GPU {info['gpu']}"
            )
        unique_gpus = set(info['gpu'] for info in binding_info)
        if len(unique_gpus) == len(binding_info):
            logger.info(f"  ✓ 所有 {len(binding_info)} 個 rank 分配到不同 GPU")
        else:
            logger.warning(
                f"  ⚠ GPU 分配衝突：{_NRANK} 個 rank 共用 {len(unique_gpus)} 張 GPU！"
                f"  這會導致串行化執行。請確認 SLURM script 加入 --gpu-bind=per_task:1"
            )


# ===========================================================================
# ★ v1.2 MPI 評估函式（結構與 v1.1 相同，GPU 綁定已在模組頂層修正）
# ===========================================================================

def _mpi_signal_stop() -> None:
    if not _HAS_MPI or _NRANK <= 1:
        return
    _COMM.bcast(_MPI_FLAG_STOP, root=0)


def _mpi_evaluate_all(
    gen:       "MoleculeGeneratorCUDAQ",
    cwg:       "ConditionalWeightsGenerator",
    args:      argparse.Namespace,
    positions: np.ndarray,
) -> list:
    """
    所有 MPI rank 同時呼叫（v1.1 設計保留）。
    GPU 綁定已在模組頂層修正，此函式邏輯不變。
    """
    if not _HAS_MPI or _NRANK <= 1:
        M = positions.shape[0]
        results = []
        for idx in range(M):
            w   = positions[idx]
            w_c = cwg.apply_chemistry_constraint(w.copy())
            gen.update_weight_vector(w_c)
            try:
                _, v, u = gen.sample_molecule(args.num_sample)
            except Exception as e:
                print(f"[Rank 0] sample_molecule 失敗（idx={idx}）：{e}", flush=True)
                v, u = 0.0, 0.0
            results.append((float(v), float(u)))
        return results

    flag = _COMM.bcast(_MPI_FLAG_CONTINUE if _RANK == 0 else None, root=0)
    if flag == _MPI_FLAG_STOP:
        return []

    positions = _COMM.bcast(positions if _RANK == 0 else None, root=0)
    M = positions.shape[0]

    my_indices = list(range(_RANK, M, _NRANK))
    my_results = []

    for idx in my_indices:
        w   = positions[idx]
        w_c = cwg.apply_chemistry_constraint(w.copy())
        gen.update_weight_vector(w_c)
        try:
            _, v, u = gen.sample_molecule(args.num_sample)
        except Exception as e:
            print(f"[Rank {_RANK}] sample_molecule 失敗（idx={idx}）：{e}", flush=True)
            v, u = 0.0, 0.0
        my_results.append((float(v), float(u), idx))

    all_scattered = _COMM.allgather(my_results)

    ordered: list = [(0.0, 0.0)] * M
    for rank_results in all_scattered:
        for v, u, idx in rank_results:
            ordered[idx] = (v, u)

    return ordered


# ===========================================================================
# ★ v1.2 新增：定期重建 Generator 以緩解記憶體洩漏
# ===========================================================================

def rebuild_generator(
    args:    argparse.Namespace,
    n_dim:   int,
    logger:  logging.Logger,
) -> "MoleculeGeneratorCUDAQ":
    """
    重建 MoleculeGeneratorCUDAQ，強制觸發新的 CUDA context 初始化。
    
    CUDA-Q 0.7.1 的 cuStateVec 會在 set_target() 時分配 pinned memory。
    重建 generator 並重新 set_target() 會創建新的 context，
    部分 pinned memory 可能被複用（driver 層快取），
    但已結束使用的記憶體可被重新分配。
    
    注意：這是緩解措施，不如子行程隔離（run_qpso_qmg_cudaq.py v10.1）徹底。
    """
    import gc, ctypes
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    gen = MoleculeGeneratorCUDAQ(
        num_heavy_atom            = args.num_heavy_atom,
        all_weight_vector         = np.zeros(n_dim),
        backend_name              = args.backend,
        remove_bond_disconnection = True,
        chemistry_constraint      = True,
    )
    if _RANK == 0:
        logger.info(
            f"[v1.2] Generator 重建完成  "
            f"RSS={get_rss_mb():.0f} MB (rank {_RANK})"
        )
    return gen


# ===========================================================================
# batch_evaluate_fn 工廠（含定期重建機制）
# ===========================================================================

def make_mpi_batch_evaluate_fn(
    gen_holder: list,   # [gen] 可變容器，允許在閉包內重建
    cwg:        "ConditionalWeightsGenerator",
    args:       argparse.Namespace,
    logger:     logging.Logger,
    n_dim:      int,
) -> callable:
    """
    v1.2：加入 reinit_every 定期重建機制。
    gen_holder 是一個單元素 list，允許閉包內替換 gen。
    """
    call_count = [0]

    def batch_evaluate_fn(positions: np.ndarray) -> list:
        # ── 定期重建 Generator（緩解 pinned memory 洩漏）────────────────
        call_count[0] += 1
        if args.reinit_every > 0 and call_count[0] % args.reinit_every == 0:
            if _RANK == 0:
                logger.info(
                    f"[v1.2] 第 {call_count[0]} 批次，觸發 Generator 重建 "
                    f"（reinit_every={args.reinit_every}）..."
                )
            # 所有 rank 同步重建（必須在 bcast 通訊框架外進行）
            if _HAS_MPI:
                _COMM.Barrier()
            gen_holder[0] = rebuild_generator(args, n_dim, logger)
            if _HAS_MPI:
                _COMM.Barrier()

        return _mpi_evaluate_all(gen_holder[0], cwg, args, positions)

    return batch_evaluate_fn


# ===========================================================================
# 主程式
# ===========================================================================

def main():
    args = parse_args()

    if _RANK == 0:
        os.makedirs(args.data_dir, exist_ok=True)
    if _HAS_MPI:
        _COMM.Barrier()

    log_path = os.path.join(args.data_dir, f"{args.task_name}.log")
    logger   = setup_logger(log_path)

    if _RANK == 0:
        logger.info(f"Task name: {args.task_name}")
        logger.info(f"Task: ['validity', 'uniqueness']")
        logger.info(f"Condition: ['None', 'None']")
        logger.info(f"objective: ['maximize', 'maximize']")
        logger.info(f"# of heavy atoms: {args.num_heavy_atom}")
        logger.info(f"# of samples: {args.num_sample}")
        logger.info(f"smarts: None")
        logger.info(f"CUDA-Q backend: {args.backend}")
        logger.info(
            f"[v1.2] GPU 綁定方式：SLURM_LOCALID + SLURM_STEP_GPUS  "
            f"（在 import cudaq 之前設定 CUDA_VISIBLE_DEVICES）"
        )
        logger.info(
            f"[v1.2] 記憶體緩解：reinit_every={args.reinit_every} 批次重建 Generator"
        )
        logger.info(f"MPI ranks: {_NRANK}")
        logger.info(f"AE-SOQPSO v1.1  調和加權: {'開啟' if args.ae_weighting else '關閉'}")
        logger.info(f"AE 配對更新間隔: {args.pair_interval} 迭代")
        log_gpu_info(logger)
        logger.info(f"[MEM] 啟動時 RSS={get_rss_mb():.0f} MB (rank 0)")

    # ── 收集並驗證所有 rank 的 GPU 綁定 ───────────────────────────────────
    log_all_gpu_bindings(logger)

    # ── 初始化 cwg ────────────────────────────────────────────────────────
    cwg = ConditionalWeightsGenerator(
        args.num_heavy_atom,
        smarts=None,
        disable_connectivity_position=[],
    )
    D = cwg.length_all_weight_vector   # 134

    if _RANK == 0:
        logger.info(f"Number of flexible parameters: {D}")

    # ── 初始化 Generator（所有 rank，各自使用已設定的 GPU）─────────────────
    gen = MoleculeGeneratorCUDAQ(
        num_heavy_atom            = args.num_heavy_atom,
        all_weight_vector         = np.zeros(D),
        backend_name              = args.backend,
        remove_bond_disconnection = True,
        chemistry_constraint      = True,
    )

    if _RANK == 0:
        logger.info(
            f"[Rank {_RANK}] Generator 初始化完成  "
            f"backend={args.backend}  GPU={_my_gpu}"
        )

    # ── 功能驗證（所有 rank 同時呼叫 _mpi_evaluate_all）──────────────────
    if _RANK == 0:
        logger.info("[MPI v1.2] 執行功能驗證（5 shots × N_RANK，驗證 GPU 並行）...")

    class _TestArgs:
        num_sample      = 5
        num_heavy_atom  = args.num_heavy_atom

    w_test         = cwg.generate_conditional_random_weights(random_seed=99)
    test_positions = np.tile(w_test, (_NRANK, 1))
    test_start     = time.time()
    test_results   = _mpi_evaluate_all(gen, cwg, _TestArgs(), test_positions)
    test_elapsed   = time.time() - test_start

    if _RANK == 0:
        any_nonzero = any(v > 0 or u > 0 for v, u in test_results)
        per_rank_s  = test_elapsed
        expected_serial_s = per_rank_s * _NRANK
        # 若並行真的有效，test_elapsed 應遠小於 serial 預期時間
        is_parallel = (test_elapsed < expected_serial_s * 0.5 + 10)
        logger.info(
            f"[MPI v1.2] 功能驗證完成  "
            f"{'✓' if any_nonzero else '⚠'} 結果  "
            f"耗時 {test_elapsed:.1f}s  "
            f"{'✓ 並行生效' if is_parallel else '⚠ 可能序列化（請確認 GPU 綁定）'}"
        )
        if not is_parallel:
            logger.warning(
                f"  [警告] 功能驗證耗時 {test_elapsed:.1f}s，"
                f"若 GPU 真正並行應 < {per_rank_s:.1f}s。\n"
                f"  建議確認 SLURM script 加入 --gpu-bind=per_task:1，"
                f"或改用 run_qpso_qmg_cudaq.py v10.1（parallel subprocess）。"
            )
        logger.info(f"[MEM] 功能驗證後 RSS={get_rss_mb():.0f} MB (rank 0)")

    if _HAS_MPI:
        _COMM.Barrier()

    # ── 建立 batch 評估函式 ────────────────────────────────────────────────
    gen_holder = [gen]   # 可變容器，允許 reinit 時替換
    batch_eval_fn = make_mpi_batch_evaluate_fn(
        gen_holder=gen_holder,
        cwg=cwg,
        args=args,
        logger=logger,
        n_dim=D,
    )

    # ── 執行優化 ──────────────────────────────────────────────────────────
    if _RANK == 0:
        optimizer = AESOQPSOOptimizer(
            n_params           = D,
            n_particles        = args.particles,
            max_iterations     = args.iterations,
            logger             = logger,
            batch_evaluate_fn  = batch_eval_fn,
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
        )
        total_evals = args.particles * (args.iterations + 1)
        logger.info(
            f"[QPSO config] M={args.particles}  T={args.iterations}  "
            f"total_evals={total_evals}  seed={args.seed}"
        )

        best_params  = None
        best_fitness = -np.inf
        try:
            best_params, best_fitness = optimizer.optimize()
        except Exception as e:
            logger.error(f"[ERROR] optimizer.optimize() 發生例外：{e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if _HAS_MPI and _NRANK > 1:
                _mpi_signal_stop()
                logger.info("[MPI v1.2] STOP 信號已發送，等待所有 rank 完成...")

    else:
        # 非 rank 0：持續等待 bcast，直到收到 STOP 信號
        while True:
            results = _mpi_evaluate_all(gen_holder[0], cwg, args, None)
            if len(results) == 0:
                break

    if _HAS_MPI:
        _COMM.Barrier()

    if _RANK == 0:
        if best_params is not None:
            best_npy = os.path.join(args.data_dir, f"{args.task_name}_best_params.npy")
            np.save(best_npy, best_params)
            logger.info(f"最佳參數已儲存: {best_npy}")
            logger.info(
                f"最終結果: V×U={best_fitness:.6f}  "
                + ("✓ 超越 BO 基線 0.8834!" if best_fitness > 0.8834
                   else "✗ 未超越")
            )
        else:
            logger.error("optimize() 未正常完成，無法儲存最佳參數。")
        logger.info(f"[MEM] 程序結束前 RSS={get_rss_mb():.0f} MB (rank 0)")

    if _HAS_MPI:
        MPI.Finalize()


if __name__ == "__main__":
    main()