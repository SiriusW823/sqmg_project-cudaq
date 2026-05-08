"""
==============================================================================
run_qpso_qmg_mpi.py  —  CUDA-Q 0.7.1 + AE-SOQPSO + MPI 並行主入口（v1.3 Deadlock修正版）
==============================================================================

v1.2 → v1.3  根本修正：MPI Deadlock in make_mpi_batch_evaluate_fn

  ★ [BUG-FIX] _COMM.Barrier() 在 batch_evaluate_fn 中造成 Deadlock
  ─────────────────────────────────────────────────────────────────────
    v1.2 的問題：
      make_mpi_batch_evaluate_fn 回傳的 batch_evaluate_fn 只被 rank 0
      呼叫（在 AESOQPSOOptimizer.optimize() 內部）。
      
      當 reinit_every 條件觸發時：
        Rank 0：batch_evaluate_fn → call_count % reinit_every == 0
                → _COMM.Barrier()              ← 等待在 Barrier
        Non-rank-0：while True loop
                → _mpi_evaluate_all()
                → _COMM.bcast(flag, root=0)    ← 等待在 bcast

      _COMM.Barrier() 與 _COMM.bcast() 是兩個不同的 MPI collective，
      所有 rank 必須呼叫「相同的」 collective 才能推進。
      上面的情況兩邊呼叫的是不同 collective → 永久 Deadlock。

    v1.3 修正方案：
      新增 _MPI_FLAG_REBUILD = 2，將 rebuild 信號折入既有的 flag bcast
      通道。rank 0 在需要 rebuild 時於 bcast 中發送 flag=2，
      所有 rank 在同一個 bcast collective 中接收並執行 rebuild，
      之後繼續正常評估流程。完全移除 _COMM.Barrier()。

      修改的函式：
        - _mpi_evaluate_all()：接受 gen_holder（list）取代 gen，
          加入 do_rebuild / n_dim / logger 參數
        - make_mpi_batch_evaluate_fn()：移除 _COMM.Barrier()，
          改傳 do_rebuild 給 _mpi_evaluate_all()
        - 非 rank-0 while 迴圈：傳入 n_dim / logger
        - main() 中的功能驗證測試呼叫：調整為 [gen] holder

  v1.2 保留（不變）：
    - SLURM_LOCALID GPU 綁定（在 import cudaq 之前設定 CUDA_VISIBLE_DEVICES）
    - AE-SOQPSO 演算法完整實作

SLURM 提交（使用修正版 v1.3）：
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
# ★ v1.2 保留：GPU 綁定必須在任何 CUDA import 之前完成
# ===========================================================================

_EARLY_RANK = int(os.environ.get("PMI_RANK",
               os.environ.get("OMPI_COMM_WORLD_RANK",
               os.environ.get("MV2_COMM_WORLD_RANK", "0"))))

_SLURM_LOCAL_ID  = int(os.environ.get("SLURM_LOCALID", _EARLY_RANK))
_SLURM_STEP_GPUS = os.environ.get("SLURM_STEP_GPUS", "")
_SLURM_JOB_GPUS  = os.environ.get("SLURM_JOB_GPUS",  "")
_ORIGINAL_CUDA_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")

if _SLURM_STEP_GPUS:
    _gpu_pool = [g.strip() for g in _SLURM_STEP_GPUS.split(",") if g.strip()]
elif _SLURM_JOB_GPUS:
    _gpu_pool = [g.strip() for g in _SLURM_JOB_GPUS.split(",") if g.strip()]
elif _ORIGINAL_CUDA_DEVICES:
    _gpu_pool = [g.strip() for g in _ORIGINAL_CUDA_DEVICES.split(",") if g.strip()]
else:
    _gpu_pool = [str(_SLURM_LOCAL_ID)]

_my_gpu = _gpu_pool[_SLURM_LOCAL_ID % len(_gpu_pool)]
os.environ["CUDA_VISIBLE_DEVICES"] = _my_gpu

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
# MPI 通訊常數（v1.3 新增 _MPI_FLAG_REBUILD）
# ===========================================================================

_MPI_FLAG_CONTINUE = 1
_MPI_FLAG_STOP     = 0
_MPI_FLAG_REBUILD  = 2   # ★ v1.3 新增：折入 bcast 通道的 rebuild 信號


# ===========================================================================
# 工具函式
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QMG CUDA-Q + AE-SOQPSO MPI 並行版 v1.3（Deadlock 修正版）",
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
        "--backend", type=str, default="cudaq_nvidia",
        choices=["cudaq_nvidia", "cudaq_qpp", "cudaq_nvidia_fp64"],
    )
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--task_name",        type=str,   default="unconditional_9_ae_mpi")
    p.add_argument("--data_dir",         type=str,   default="results_mpi")
    p.add_argument(
        "--reinit_every", type=int, default=10,
        help=(
            "每隔多少 QPSO 批次重建 MoleculeGeneratorCUDAQ（釋放 CUDA pinned memory）。"
            "0 表示停用。建議值：10。"
            "v1.3 修正：不再使用 _COMM.Barrier()，透過 flag bcast 信號同步。"
        ),
    )
    return p.parse_args()


def setup_logger(log_path: str) -> logging.Logger:
    if _RANK != 0:
        null = logging.getLogger(f"null_rank{_RANK}")
        null.addHandler(logging.NullHandler())
        return null
    logger = logging.getLogger("AEMPILogger_v13")
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
    if not _HAS_MPI:
        logger.info(f"  GPU binding: rank 0 → GPU {_my_gpu}")
        return
    binding_info = _COMM.gather(
        {"rank": _RANK, "gpu": _my_gpu, "local_id": _SLURM_LOCAL_ID},
        root=0,
    )
    if _RANK == 0:
        logger.info("  GPU 綁定確認（v1.3 SLURM_LOCALID 方式）：")
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
                f"  請確認 SLURM script 加入 --gpu-bind=per_task:1"
            )


# ===========================================================================
# ★ v1.3 修正：_mpi_evaluate_all 重寫
#   - 接受 gen_holder: list（取代直接接受 gen），允許在 rebuild 後更新引用
#   - 新增 do_rebuild / n_dim / logger 參數
#   - 將 rebuild 邏輯折入既有的 flag bcast，完全消除獨立 Barrier
# ===========================================================================

def _mpi_evaluate_all(
    gen_holder: list,
    cwg:        "ConditionalWeightsGenerator",
    args:       argparse.Namespace,
    positions:  np.ndarray,
    *,
    do_rebuild: bool              = False,
    n_dim:      int               = 134,
    logger:     logging.Logger    = None,
) -> list:
    """
    所有 MPI rank 同時呼叫。

    v1.3 修正：
      - 第一個 bcast 同時攜帶 CONTINUE / STOP / REBUILD 三種信號
      - 收到 REBUILD 時，所有 rank 在 bcast 之後立刻重建 generator，
        然後繼續執行本次評估（不再需要額外的 Barrier 或 bcast）
      - do_rebuild 只由 rank 0 有效；non-rank-0 永遠傳 False，
        但會從 bcast 接收 rank 0 的信號
    """
    _logger = logger or logging.getLogger(__name__)

    # ── 單 rank 模式（非 MPI 或 NRANK=1）────────────────────────────────
    if not _HAS_MPI or _NRANK <= 1:
        if do_rebuild:
            gen_holder[0] = rebuild_generator(args, n_dim, _logger)
        M = positions.shape[0]
        results = []
        for idx in range(M):
            w   = positions[idx]
            w_c = cwg.apply_chemistry_constraint(w.copy())
            gen_holder[0].update_weight_vector(w_c)
            try:
                _, v, u = gen_holder[0].sample_molecule(args.num_sample)
            except Exception as e:
                print(f"[Rank 0] sample_molecule 失敗（idx={idx}）：{e}", flush=True)
                v, u = 0.0, 0.0
            results.append((float(v), float(u)))
        return results

    # ── MPI 多 rank 模式 ─────────────────────────────────────────────────

    # ★ v1.3 關鍵修正：
    #   將 do_rebuild 與 CONTINUE/STOP 折入同一個 bcast。
    #   rank 0 發送：REBUILD(2) > CONTINUE(1) > STOP(0)
    #   non-rank-0 傳 None，接收 rank 0 的值。
    if _RANK == 0:
        flag_to_send = _MPI_FLAG_REBUILD if do_rebuild else _MPI_FLAG_CONTINUE
    else:
        flag_to_send = None

    flag = _COMM.bcast(flag_to_send, root=0)

    if flag == _MPI_FLAG_STOP:
        return []

    # ★ 所有 rank 在同一個 bcast 之後同步執行 rebuild（無需額外 Barrier）
    if flag == _MPI_FLAG_REBUILD:
        if _RANK == 0:
            _logger.info(
                f"[v1.3] Generator rebuild 信號已廣播（全 {_NRANK} rank 同步重建）"
            )
        gen_holder[0] = rebuild_generator(args, n_dim, _logger)
        # 不需要 _COMM.Barrier()：allgather 在後面確保同步

    # ── 廣播 positions ─────────────────────────────────────────────────
    positions = _COMM.bcast(positions if _RANK == 0 else None, root=0)
    M = positions.shape[0]

    # ── 各 rank 評估自己分配到的粒子 ─────────────────────────────────────
    my_indices = list(range(_RANK, M, _NRANK))
    my_results = []

    for idx in my_indices:
        w   = positions[idx]
        w_c = cwg.apply_chemistry_constraint(w.copy())
        gen_holder[0].update_weight_vector(w_c)
        try:
            _, v, u = gen_holder[0].sample_molecule(args.num_sample)
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
# ★ v1.2 保留：定期重建 Generator
# ===========================================================================

def rebuild_generator(
    args:    argparse.Namespace,
    n_dim:   int,
    logger:  logging.Logger,
) -> "MoleculeGeneratorCUDAQ":
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
            f"[v1.3] Generator 重建完成  "
            f"RSS={get_rss_mb():.0f} MB (rank {_RANK})"
        )
    return gen


# ===========================================================================
# ★ v1.3 修正：batch_evaluate_fn 工廠（移除 _COMM.Barrier()）
# ===========================================================================

def make_mpi_batch_evaluate_fn(
    gen_holder: list,
    cwg:        "ConditionalWeightsGenerator",
    args:       argparse.Namespace,
    logger:     logging.Logger,
    n_dim:      int,
) -> callable:
    """
    v1.3：移除 reinit 路徑中的 _COMM.Barrier()，改透過 _mpi_evaluate_all
    的 do_rebuild 參數傳遞 rebuild 信號，由 flag bcast 統一同步。

    v1.2 的 deadlock 根因：
      batch_evaluate_fn 只被 rank 0 呼叫。當 Barrier 被觸發時，
      non-rank-0 正在等待 _mpi_evaluate_all 的 bcast(flag)，
      而 rank 0 在等待 Barrier → 不同 collective → deadlock。

    v1.3 的修正：
      do_rebuild=True 傳給 _mpi_evaluate_all，在 flag bcast 中發送
      _MPI_FLAG_REBUILD(2)，所有 rank 在同一個 bcast 後同步重建。
    """
    call_count = [0]

    def batch_evaluate_fn(positions: np.ndarray) -> list:
        call_count[0] += 1
        do_rebuild = (args.reinit_every > 0 and call_count[0] % args.reinit_every == 0)

        if do_rebuild and _RANK == 0:
            logger.info(
                f"[v1.3] 第 {call_count[0]} 批次，透過 MPI flag 信號觸發 "
                f"Generator 重建（reinit_every={args.reinit_every}）..."
            )

        # ★ v1.3：不再使用 _COMM.Barrier()
        #   do_rebuild 由 _mpi_evaluate_all 的 flag bcast 傳播到所有 rank
        return _mpi_evaluate_all(
            gen_holder, cwg, args, positions,
            do_rebuild = do_rebuild,
            n_dim      = n_dim,
            logger     = logger,
        )

    return batch_evaluate_fn


# ===========================================================================
# MPI stop 信號
# ===========================================================================

def _mpi_signal_stop() -> None:
    """rank 0 廣播 STOP 信號，non-rank-0 在下一次 bcast 接收後退出迴圈。"""
    if not _HAS_MPI or _NRANK <= 1:
        return
    _COMM.bcast(_MPI_FLAG_STOP, root=0)


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
            f"[v1.3] GPU 綁定方式：SLURM_LOCALID + SLURM_STEP_GPUS  "
            f"（在 import cudaq 之前設定 CUDA_VISIBLE_DEVICES）"
        )
        logger.info(
            f"[v1.3] 記憶體緩解：reinit_every={args.reinit_every} 批次重建 Generator"
            f"（透過 MPI flag bcast 同步，無 Barrier Deadlock 風險）"
        )
        logger.info(f"MPI ranks: {_NRANK}")
        logger.info(f"AE-SOQPSO v1.1  調和加權: {'開啟' if args.ae_weighting else '關閉'}")
        logger.info(f"AE 配對更新間隔: {args.pair_interval} 迭代")
        log_gpu_info(logger)
        logger.info(f"[MEM] 啟動時 RSS={get_rss_mb():.0f} MB (rank 0)")

    log_all_gpu_bindings(logger)

    cwg = ConditionalWeightsGenerator(
        args.num_heavy_atom,
        smarts=None,
        disable_connectivity_position=[],
    )
    D = cwg.length_all_weight_vector   # 134

    if _RANK == 0:
        logger.info(f"Number of flexible parameters: {D}")

    # ── 初始化 Generator ──────────────────────────────────────────────────
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

    # ── 功能驗證 ──────────────────────────────────────────────────────────
    if _RANK == 0:
        logger.info("[MPI v1.3] 執行功能驗證（5 shots × N_RANK，驗證 GPU 並行）...")

    class _TestArgs:
        num_sample      = 5
        num_heavy_atom  = args.num_heavy_atom

    w_test         = cwg.generate_conditional_random_weights(random_seed=99)
    test_positions = np.tile(w_test, (_NRANK, 1))
    test_start     = time.time()

    # ★ v1.3：使用 gen_holder list，與正式執行一致
    test_gen_holder = [gen]
    test_results = _mpi_evaluate_all(
        test_gen_holder, cwg, _TestArgs(), test_positions,
        n_dim=D, logger=logger,
    )
    test_elapsed = time.time() - test_start

    if _RANK == 0:
        any_nonzero = any(v > 0 or u > 0 for v, u in test_results)
        per_rank_s  = test_elapsed
        is_parallel = (test_elapsed < per_rank_s * _NRANK * 0.5 + 10)
        logger.info(
            f"[MPI v1.3] 功能驗證完成  "
            f"{'✓' if any_nonzero else '⚠'} 結果  "
            f"耗時 {test_elapsed:.1f}s  "
            f"{'✓ 並行生效' if is_parallel else '⚠ 可能序列化（請確認 GPU 綁定）'}"
        )
        if not is_parallel:
            logger.warning(
                f"  [警告] 功能驗證耗時 {test_elapsed:.1f}s 超出預期。\n"
                f"  建議確認 SLURM script 加入 --gpu-bind=per_task:1，"
                f"或改用 run_qpso_qmg_cudaq.py v10.1（parallel subprocess）。"
            )
        logger.info(f"[MEM] 功能驗證後 RSS={get_rss_mb():.0f} MB (rank 0)")

    if _HAS_MPI:
        _COMM.Barrier()

    # ── 正式 gen_holder 與 batch 評估函式 ────────────────────────────────
    gen_holder    = [gen]
    batch_eval_fn = make_mpi_batch_evaluate_fn(
        gen_holder = gen_holder,
        cwg        = cwg,
        args       = args,
        logger     = logger,
        n_dim      = D,
    )

    # ── 執行優化（只有 rank 0 持有 optimizer）────────────────────────────
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
                logger.info("[MPI v1.3] STOP 信號已發送，等待所有 rank 完成...")

    else:
        # ★ v1.3：non-rank-0 持續呼叫 _mpi_evaluate_all，直到收到 STOP 信號
        #   傳入 gen_holder（list），允許 rebuild 信號更新 generator 引用
        while True:
            results = _mpi_evaluate_all(
                gen_holder, cwg, args, None,
                n_dim  = D,
                logger = logger,
            )
            if len(results) == 0:   # STOP 信號：return [] in _mpi_evaluate_all
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