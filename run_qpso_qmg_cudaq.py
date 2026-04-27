"""
==============================================================================
run_qpso_qmg_cudaq.py  — CUDA-Q 0.7.1 + AE-SOQPSO  (v10.1 parallel-subprocess 版)
==============================================================================

v9.6 → v10.1  根本性架構升級（根因診斷 2026-04-24）：

  ★ 問題一確診：MPI 並行化在 CUDA-Q 0.7.1 + NCHC DGX111 上完全失效
  ─────────────────────────────────────────────────────────────────────
    症狀：50 粒子 × 122.6s/粒子 = 6129s（全串行），8 GPU 並行應為 858s
    根本原因（三層複合）：
      (1) NCHC SLURM 的 GPU 資源綁定方式：--gres=gpu:8 + --ntasks-per-node=8
          並不會自動將每個 MPI task 綁定到不同物理 GPU。
          所有 8 個 rank 可能共享同一物理 GPU，Python 層的
          os.environ["CUDA_VISIBLE_DEVICES"] 在 cgroup 環境下可能被忽略。
      (2) CUDA-Q 0.7.1 cuStateVec 後端對多行程呼叫存在序列化：
          即使不同行程使用不同 GPU，libcustatevec 的 stream 初始化
          可能通過 /dev/nvidia-ctl 產生全節點級別的序列化鎖。
      (3) MPI rank 生命週期橫跨整個實驗，CUDA context 永不銷毀，
          cudaMallocHost pinned memory 只進不出。

    實測記憶體洩漏：
      Phase 0：122.6s/粒子 → Iter 1：133.7s（+8.9%，記憶體壓力上升）
      Iter 2：128.2s → Iter 3：129.9s → OOM Kill（signal 9，rank 1）
      4 次批次 × 50 粒子 × 2.5GB/粒子 = 500GB pinned，超過系統 RAM 上限

  ★ v10.1 解法：parallel subprocess pool（放棄 MPI，使用多行程子程序池）
  ─────────────────────────────────────────────────────────────────────
    核心機制：
      1. 完全棄用 mpi4py，使用 subprocess.Popen 並行啟動子行程
      2. 每個子行程由父行程設定 CUDA_VISIBLE_DEVICES=<gpu_id>，
         繞過 SLURM cgroup 限制，確保每個子行程看到唯一的 GPU
      3. 子行程在啟動時 CUDA 尚未初始化，CUDA_VISIBLE_DEVICES 有效
      4. 評估完成後子行程結束 → CUDA context 銷毀 → pinned memory 釋放
      5. 批次分輪：每輪同時啟動 min(N_GPUS, remaining) 個子行程

    效能預估（N=50 粒子，8 GPU，T=40 迭代）：
      每批次：⌈50/8⌉=7 輪 × 122.6s = 858s ≈ 14 分鐘
      Phase 0：858s（對比 MPI 版 6129s，提速 7.1×）
      完整實驗 (T=40)：41 批次 × 858s = 35,178s ≈ 9.8 小時
      完整實驗 (T=200)：201 批次 × 858s ≈ 47.9 小時（建議改用 T=40）

    記憶體安全：
      每個子行程只執行 1 次 cudaq.sample()，評估後即結束
      主行程記憶體穩定（< 1 GB）
      8 個子行程並行，各佔 ~2.5GB pinned → 8 × 2.5GB = 20GB（可接受）

  ★ 使用 AESOQPSOOptimizer 取代 QMGSOQPSOOptimizer
  ─────────────────────────────────────────────────────────────────────
    AESOQPSOOptimizer（qpso_optimizer_ae.py）：
      - 支援 batch_evaluate_fn 介面（批次評估，必要）
      - AE-QTS v1.1 U 形對稱調和加權 mbest（更好的收斂）
      - 雙目標分解追蹤 V⋆/U⋆
      - 演算法品質優於 QMGSOQPSOOptimizer

依賴：
  worker_eval.py 必須與本檔案放在同一目錄
  qpso_optimizer_ae.py 必須在 PYTHONPATH 可及範圍
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
# 參數解析
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QMG CUDA-Q 0.7.1 + AE-SOQPSO（v10.1 parallel subprocess 版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── 基本參數 ──────────────────────────────────────────────────────────
    p.add_argument("--num_heavy_atom",    type=int,   default=9)
    p.add_argument("--num_sample",        type=int,   default=10000)
    p.add_argument("--particles",         type=int,   default=50)
    p.add_argument("--iterations",        type=int,   default=40)
    p.add_argument("--seed",              type=int,   default=42)

    # ── GPU 並行設定 ──────────────────────────────────────────────────────
    p.add_argument(
        "--n_gpus", type=int, default=8,
        help="並行 GPU 數量。每輪同時啟動此數量的子行程，各佔一張 GPU。"
             "設為 1 等同於 v9.6 單 GPU 序列模式。",
    )
    p.add_argument(
        "--gpu_ids", type=str, default="0,1,2,3,4,5,6,7",
        help="可用 GPU 的 device index 列表，逗號分隔。"
             "子行程依 round-robin 分配。例如：'0,1,2,3,4,5,6,7'",
    )
    p.add_argument(
        "--backend", type=str, default="cudaq_nvidia",
        choices=["cudaq_qpp", "cudaq_nvidia", "cudaq_nvidia_fp64",
                 "cudaq_tensornet", "cudaq_tensornet_mps"],
    )
    p.add_argument(
        "--subprocess_timeout", type=int, default=600,
        help="每個子行程的最大執行秒數",
    )

    # ── SOQPSO 超參數 ─────────────────────────────────────────────────────
    p.add_argument("--alpha_max",          type=float, default=1.2)
    p.add_argument("--alpha_min",          type=float, default=0.4)
    p.add_argument("--mutation_prob",      type=float, default=0.12)
    p.add_argument("--stagnation_limit",   type=int,   default=8)
    p.add_argument("--reinit_fraction",    type=float, default=0.20)

    # ── AE-QTS 超參數 ─────────────────────────────────────────────────────
    p.add_argument("--ae_weighting",       action="store_true",  default=True,
                   help="AE-QTS v1.1 U 形對稱調和加權 mbest（預設開啟）")
    p.add_argument("--no_ae_weighting",    action="store_false", dest="ae_weighting")
    p.add_argument("--pair_interval",      type=int,   default=5,
                   help="AE-QTS 配對更新執行間隔（QPSO 迭代數）")
    p.add_argument("--rotate_factor",      type=float, default=0.01,
                   help="AE-QTS 配對更新步長縮放因子")

    # ── 輸出設定 ──────────────────────────────────────────────────────────
    p.add_argument("--task_name", type=str, default="unconditional_9_qpso")
    p.add_argument("--data_dir",  type=str, default="results_parallel_gpu")
    return p.parse_args()


# ===========================================================================
# Logger
# ===========================================================================

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("ParallelGPUQPSOLogger")
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
# ★ v10.1 核心：parallel subprocess batch evaluate function
# ===========================================================================

def make_parallel_batch_evaluate_fn(
    args:          argparse.Namespace,
    cwg:           ConditionalWeightsGenerator,
    logger:        logging.Logger,
    worker_script: str,
    gpu_ids:       list,            # 每個元素是 GPU device index (str)
) -> callable:
    """
    建立 parallel subprocess 批次評估函式。

    執行流程（以 M=50, N_GPUS=8 為例）：
      Round 1: 同時啟動 8 個子行程評估粒子 0..7
               各子行程 CUDA_VISIBLE_DEVICES = 0, 1, ..., 7
               等待所有 8 個子行程完成（~122.6s）
               子行程結束 → CUDA context 銷毀 → pinned memory 釋放
      Round 2: 粒子 8..15，同上
      ...
      Round 7: 粒子 48..49（只有 2 個子行程）
      Total:   7 rounds × 122.6s = 858s ≈ 14 分鐘

    相比 MPI 方案：
      MPI（失效）：50 × 122.6s = 6130s（所有 8 ranks 在同一物理 GPU 上序列化）
      本方案：7 × 122.6s = 858s（8 GPU 真正並行，7.1× 加速）

    記憶體安全性：
      每個子行程只執行 1 次 cudaq.sample()，結束即釋放 ~2.5GB pinned memory
      同時最多 8 個子行程並行 → 最多 8 × 2.5GB = 20GB pinned（V100 16GB × 8 可容納）
    """
    n_gpus     = len(gpu_ids)
    pythonpath = os.environ.get("PYTHONPATH", ".")
    eval_count = [0]

    def batch_evaluate_fn(positions: np.ndarray) -> list:
        M = positions.shape[0]
        results: list = [(0.0, 0.0)] * M
        t_batch_start = time.time()

        n_rounds = (M + n_gpus - 1) // n_gpus
        for round_idx in range(n_rounds):
            round_start = round_idx * n_gpus
            round_end   = min(round_start + n_gpus, M)
            round_pids  = list(range(round_start, round_end))
            round_size  = len(round_pids)

            t_round = time.time()
            procs: list      = []   # (proc, result_path, particle_idx, gpu_id_str)
            weight_paths: list = []

            # ── 同時啟動 round_size 個子行程 ──────────────────────────────
            for local_i, particle_idx in enumerate(round_pids):
                gpu_id_str = str(gpu_ids[local_i % n_gpus])
                uid        = uuid.uuid4().hex[:8]
                wpath      = os.path.join(tempfile.gettempdir(), f"qmg_pw_{uid}.npy")
                rpath      = os.path.join(tempfile.gettempdir(), f"qmg_pr_{uid}.npy")

                # chemistry constraint 在主行程套用（與 v9.6 一致）
                w_c = cwg.apply_chemistry_constraint(positions[particle_idx].copy())
                np.save(wpath, w_c)

                # ★ 關鍵：由父行程設定 CUDA_VISIBLE_DEVICES，
                #   確保子行程在 CUDA 初始化前已看到正確的 GPU
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

                # 非阻塞啟動（Popen 立即返回，子行程在背景執行）
                proc = subprocess.Popen(
                    cmd,
                    env    = env,
                    stdout = subprocess.DEVNULL,
                    stderr = subprocess.PIPE,
                )
                procs.append((proc, rpath, particle_idx, gpu_id_str))
                weight_paths.append(wpath)
                eval_count[0] += 1

            # ── 等待本輪所有子行程完成 ─────────────────────────────────────
            for proc, rpath, particle_idx, gpu_id_str in procs:
                try:
                    _, stderr_bytes = proc.communicate(
                        timeout=args.subprocess_timeout
                    )
                    if proc.returncode == 0:
                        arr = np.load(rpath)
                        results[particle_idx] = (float(arr[0]), float(arr[1]))
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
                        f"逾時（>{args.subprocess_timeout}s），回傳 V=0 U=0"
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

            # ── 清理 weight 暫存檔 ─────────────────────────────────────────
            for wp in weight_paths:
                try:
                    os.remove(wp)
                except FileNotFoundError:
                    pass

            # ── 本輪統計 ───────────────────────────────────────────────────
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

        return results

    return batch_evaluate_fn


# ===========================================================================
# 單 GPU 序列模式（n_gpus=1，向後相容 v9.6）
# ===========================================================================

def make_subprocess_evaluate_fn(
    args:          argparse.Namespace,
    cwg:           ConditionalWeightsGenerator,
    logger:        logging.Logger,
    worker_script: str,
    gpu_id:        str,
) -> callable:
    """
    單粒子、單 GPU 的序列評估（v9.6 模式，n_gpus=1 時使用）。
    保留供單 GPU 測試或向後相容使用。
    """
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
            for p in [wpath, rpath]:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

    return evaluate_fn


# ===========================================================================
# worker 功能驗證（並行啟動 n_gpus 個測試子行程）
# ===========================================================================

def verify_workers_parallel(
    args:          argparse.Namespace,
    cwg:           ConditionalWeightsGenerator,
    logger:        logging.Logger,
    worker_script: str,
    gpu_ids:       list,
) -> bool:
    """
    同時啟動 n_gpus 個 worker 子行程（各 5 shots），驗證並行功能。
    返回 True 表示所有 GPU 均正常。
    """
    logger.info(
        f"[v10.1] 並行功能驗證：同時啟動 {len(gpu_ids)} 個子行程（各 5 shots）..."
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

    all_ok  = True
    results = {}
    for proc, rpath, gpu_id_str in procs:
        try:
            _, stderr_bytes = proc.communicate(timeout=180)
            if proc.returncode == 0:
                arr = np.load(rpath)
                results[gpu_id_str] = (float(arr[0]), float(arr[1]))
                logger.info(
                    f"  GPU {gpu_id_str}: V={arr[0]:.3f}  U={arr[1]:.3f}  ✓"
                )
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
        f"[v10.1] 並行驗證完成（{elapsed:.1f}s）  "
        f"{'✓ 所有 GPU 正常' if all_ok else '✗ 有 GPU 失敗，請確認環境'}"
    )

    # ── 診斷：若時間與串行（n_gpus × 單次時間）接近，代表沒有真正並行 ──
    expected_parallel = elapsed
    if len(gpu_ids) > 1 and all_ok:
        # 5 shots 理應非常快（< 5s），但實際時間可能更長（PTX 編譯、初始化）
        # 此處只記錄，讓用戶自行判斷
        logger.info(
            f"  並行驗證耗時 {elapsed:.1f}s（{len(gpu_ids)} GPU 同時運行）"
        )
        if elapsed > 60 * len(gpu_ids):
            logger.warning(
                f"  ⚠ 耗時異常（{elapsed:.1f}s > {60*len(gpu_ids)}s），"
                f"可能存在 GPU 序列化問題（如所有子行程使用同一 GPU）。"
                f"建議確認 CUDA_VISIBLE_DEVICES 是否在子行程中生效。"
            )

    return all_ok


# ===========================================================================
# 主程式
# ===========================================================================

def main() -> None:
    args = parse_args()

    # ── 解析 GPU ID 列表 ──────────────────────────────────────────────────
    gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
    if len(gpu_ids) < args.n_gpus:
        print(
            f"[WARNING] --gpu_ids 包含 {len(gpu_ids)} 個 GPU，"
            f"但 --n_gpus={args.n_gpus}。"
            f"使用 round-robin 分配，部分 GPU 會處理多個粒子。"
        )
    # 統一 n_gpus 為實際可用 GPU 數量（最多）
    effective_n_gpus = min(args.n_gpus, len(gpu_ids))
    gpu_ids = gpu_ids[:effective_n_gpus]

    # ── 設定輸出目錄與 Logger ─────────────────────────────────────────────
    os.makedirs(args.data_dir, exist_ok=True)
    log_path = os.path.join(args.data_dir, f"{args.task_name}.log")
    logger   = setup_logger(log_path)

    # ── 基本資訊記錄 ─────────────────────────────────────────────────────
    logger.info(f"Task name: {args.task_name}")
    logger.info(f"Task: ['validity', 'uniqueness']")
    logger.info(f"Condition: ['None', 'None']")
    logger.info(f"objective: ['maximize', 'maximize']")
    logger.info(f"# of heavy atoms: {args.num_heavy_atom}")
    logger.info(f"# of samples: {args.num_sample}")
    logger.info(f"smarts: None")
    logger.info(f"disable_connectivity_position: []")
    logger.info(f"CUDA-Q backend: {args.backend}")
    logger.info(
        f"[v10.1] 評估模式: parallel subprocess pool  "
        f"N_GPUS={effective_n_gpus}  GPU_IDs={gpu_ids}"
    )
    logger.info(f"[v10.1] subprocess_timeout: {args.subprocess_timeout}s")
    logger.info(
        f"[v10.1] 根因修正: MPI 序列化（CUDA-Q 0.7.1 全節點 CUDA context 鎖）"
        f"→ 子行程隔離（每評估獨立 CUDA context + 記憶體完全釋放）"
    )
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
    assert n_flexible == cwg.length_all_weight_vector, (
        f"[BUG] n_flexible={n_flexible} != {cwg.length_all_weight_vector}"
    )

    total_evals = args.particles * (args.iterations + 1)
    batches_per_iter = (args.particles + effective_n_gpus - 1) // effective_n_gpus
    est_time_per_batch = 122.6  # seconds，基於 V100 實測
    est_total_h = total_evals / args.particles * batches_per_iter * est_time_per_batch / 3600
    logger.info(
        f"[parallel config] M={args.particles}  T={args.iterations}  "
        f"total_evals={total_evals}  "
        f"每批次 {batches_per_iter} 輪 × {effective_n_gpus} GPU 並行  "
        f"預估總時間：{est_total_h:.1f}h"
    )

    # ── 並行 worker 功能驗證 ──────────────────────────────────────────────
    if not verify_workers_parallel(args, cwg, logger, worker_script, gpu_ids):
        logger.error(
            "[ERROR] 並行驗證失敗。請檢查：\n"
            "  1. worker_eval.py 是否正確\n"
            "  2. CUDA-Q 是否正確安裝\n"
            "  3. GPU 是否可用（nvidia-smi）\n"
            "  建議先用 --n_gpus 1 --gpu_ids 0 確認單 GPU 正常"
        )
        sys.exit(1)
    log_memory(logger, "並行驗證後")

    # ── 建立評估函式 ──────────────────────────────────────────────────────
    if effective_n_gpus == 1:
        # 單 GPU：使用向後相容的 evaluate_fn（傳給 AESOQPSOOptimizer 的 evaluate_fn）
        evaluate_fn = make_subprocess_evaluate_fn(
            args=args, cwg=cwg, logger=logger,
            worker_script=worker_script,
            gpu_id=str(gpu_ids[0]),
        )
        batch_evaluate_fn = None
        logger.info(f"[v10.1] 使用 單GPU 序列模式（GPU {gpu_ids[0]}）")
    else:
        # 多 GPU：使用 parallel batch evaluate
        evaluate_fn       = None
        batch_evaluate_fn = make_parallel_batch_evaluate_fn(
            args=args, cwg=cwg, logger=logger,
            worker_script=worker_script,
            gpu_ids=gpu_ids,
        )
        logger.info(
            f"[v10.1] 使用 {effective_n_gpus}-GPU 並行模式  "
            f"GPU IDs: {gpu_ids}"
        )

    # ── 建立 AESOQPSOOptimizer ────────────────────────────────────────────
    optimizer = AESOQPSOOptimizer(
        n_params           = n_flexible,
        n_particles        = args.particles,
        max_iterations     = args.iterations,
        logger             = logger,
        evaluate_fn        = evaluate_fn,        # None 當 n_gpus > 1
        batch_evaluate_fn  = batch_evaluate_fn,  # None 當 n_gpus == 1
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
           else f"✗ 未超越 — 差距 {0.8834 - best_fitness:.4f}，"
                f"建議增加 --iterations 或 --particles")
    )
    log_memory(logger, "程序結束前")


if __name__ == "__main__":
    main()