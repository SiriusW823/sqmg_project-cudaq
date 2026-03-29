"""
==============================================================================
run_qpso_qmg_cudaq.py  ─ CUDA-Q + SOQPSO 主入口
==============================================================================

此腳本整合：
  - CUDA-Q 量子電路（DynamicCircuitBuilderCUDAQ）
  - SOQPSO 優化器（QMGSOQPSOOptimizer，與 Qiskit 版完全共用）
  - 結果格式與 unconditional_9.log 對齊

典型執行方式（單 GPU）：
  python run_qpso_qmg_cudaq.py \\
      --num_heavy_atom 9  --num_sample 10000 \\
      --particles 50      --iterations 200   \\
      --backend cudaq_nvidia                 \\
      --alpha_max 1.5     --alpha_min 0.5    \\
      --seed 42           --task_name unconditional_9_cudaq_v100 \\
      --data_dir results_cudaq_v100          \\
      2>&1 | tee results_cudaq_v100/run.log

支援 backend：
  cudaq_qpp         → CPU（無 GPU 時 fallback）
  cudaq_nvidia      → 單 GPU，cuStateVec FP32（V100 推薦）
  cudaq_nvidia_fp64 → 單 GPU，cuStateVec FP64
==============================================================================
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

# ── RDKit 靜音 ────────────────────────────────────────────────────────────────
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

# ── CUDA-Q ───────────────────────────────────────────────────────────────────
try:
    import cudaq
except ImportError:
    print("[ERROR] 無法 import cudaq。請執行：")
    print("  pip install cuda-quantum-cu11   # CUDA 11.x")
    print("  pip install cuda-quantum-cu12   # CUDA 12.x")
    sys.exit(1)

# ── QMG 套件 ─────────────────────────────────────────────────────────────────
try:
    from qmg.utils import ConditionalWeightsGenerator
except ImportError as e:
    print(f"[ERROR] 無法 import qmg: {e}")
    print("  請確認在專案根目錄（sqmg_project-cudaq/）執行，且 qmg/ 資料夾存在。")
    sys.exit(1)

# ── CUDA-Q 生成器（修正版） ────────────────────────────────────────────────────
try:
    from qmg.generator_cudaq import MoleculeGeneratorCUDAQ
except ImportError as e:
    print(f"[ERROR] 無法 import generator_cudaq: {e}")
    sys.exit(1)

# ── SOQPSO 優化器（Qiskit/CUDA-Q 共用，不需修改） ──────────────────────────────
try:
    from qpso_optimizer_qmg import QMGSOQPSOOptimizer
except ImportError as e:
    print(f"[ERROR] 無法 import qpso_optimizer_qmg: {e}")
    sys.exit(1)


# ===========================================================================
# 命令列參數
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QMG CUDA-Q + SOQPSO 分子生成優化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 電路參數
    p.add_argument("--num_heavy_atom",   type=int,   default=9,
                   help="重原子數（目前支援 N=9）")
    p.add_argument("--num_sample",       type=int,   default=10000,
                   help="每次電路評估的 shots 數")
    # QPSO 參數
    p.add_argument("--particles",        type=int,   default=5,
                   help="QPSO 粒子數 M")
    p.add_argument("--iterations",       type=int,   default=120,
                   help="QPSO 迭代數 T（總 evals = M×(T+1)）")
    p.add_argument("--alpha_max",        type=float, default=1.2)
    p.add_argument("--alpha_min",        type=float, default=0.4)
    p.add_argument("--mutation_prob",    type=float, default=0.12,
                   help="Cauchy 變異機率")
    p.add_argument("--stagnation_limit", type=int,   default=8,
                   help="停滯偵測門檻（QPSO 迭代次數）")
    p.add_argument("--seed",             type=int,   default=42)
    # Backend
    p.add_argument("--backend",          type=str,   default="cudaq_nvidia",
                   choices=["cudaq_qpp", "cudaq_nvidia", "cudaq_nvidia_fp64"],
                   help="CUDA-Q 模擬後端")
    # 輸出
    p.add_argument("--task_name",        type=str,
                   default="unconditional_9_cudaq")
    p.add_argument("--data_dir",         type=str,
                   default="results_cudaq")
    return p.parse_args()


# ===========================================================================
# Logger（格式對齊 unconditional_9.log）
# ===========================================================================

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("CUDAQQPSOLogger")
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
# GPU 診斷資訊
# ===========================================================================

def log_gpu_info(logger: logging.Logger) -> None:
    try:
        import subprocess
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=index,name,memory.total,driver_version "
            "--format=csv,noheader",
            shell=True, stderr=subprocess.DEVNULL,
        ).decode().strip()
        for line in out.splitlines():
            logger.info(f"  GPU: {line.strip()}")
    except Exception:
        logger.info("  GPU info: 無法取得（nvidia-smi 不可用）")

    try:
        logger.info(f"  CUDA-Q version: {cudaq.__version__}")
    except AttributeError:
        pass

    try:
        avail = [str(t) for t in cudaq.get_targets()]
        logger.info(f"  Available targets: {avail}")
    except Exception:
        pass


# ===========================================================================
# 主流程
# ===========================================================================

def main() -> None:
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    log_path = os.path.join(args.data_dir, f"{args.task_name}.log")
    logger   = setup_logger(log_path)

    # ── 啟動資訊（格式對齊 unconditional_9.log）──────────────────────────────
    logger.info(f"Task name: {args.task_name}")
    logger.info(f"Task: ['validity', 'uniqueness']")
    logger.info(f"Condition: ['None', 'None']")
    logger.info(f"objective: ['maximize', 'maximize']")
    logger.info(f"# of heavy atoms: {args.num_heavy_atom}")
    logger.info(f"# of samples: {args.num_sample}")
    logger.info(f"smarts: None")
    logger.info(f"disable_connectivity_position: None")
    logger.info(f"CUDA-Q backend: {args.backend}")
    log_gpu_info(logger)

    # ── ConditionalWeightsGenerator（N=9, smarts=None → 全部 134 參數可自由優化）
    cwg = ConditionalWeightsGenerator(
        args.num_heavy_atom,
        smarts                       = None,
        disable_connectivity_position = None,
    )
    n_flexible = int((cwg.parameters_indicator == 0.0).sum())
    logger.info(f"Number of flexible parameters: {n_flexible}")

    total_evals = args.particles * (args.iterations + 1)
    logger.info(
        f"[CUDAQ-QPSO config] "
        f"M={args.particles}  T={args.iterations}  "
        f"total_evals={total_evals}  seed={args.seed}  "
        f"backend={args.backend}"
    )

    # ── CUDA-Q 生成器（一次性建立，避免重複 JIT 編譯）────────────────────────
    logger.info("[CUDAQ] 初始化 MoleculeGeneratorCUDAQ（首次 JIT 可能需 10-30s）...")
    t_init = time.time()
    generator = MoleculeGeneratorCUDAQ(
        num_heavy_atom            = args.num_heavy_atom,
        backend_name              = args.backend,
        remove_bond_disconnection = True,
        chemistry_constraint      = True,
    )
    logger.info(f"[CUDAQ] 初始化完成，耗時 {time.time()-t_init:.1f}s")

    # ── Evaluate function（QPSO → 量子電路評估）───────────────────────────────
    def evaluate_fn(pos: np.ndarray) -> tuple[float, float]:
        """
        pos: QPSO 粒子位置，shape=(n_flexible,)，值域 [0,1]
        →   apply_chemistry_constraint 後直接作為電路 weight vector
        回傳 (validity, uniqueness)
        """
        w = cwg.apply_chemistry_constraint(pos.copy())
        generator.update_weight_vector(w)
        _, validity, uniqueness = generator.sample_molecule(args.num_sample)
        return float(validity), float(uniqueness)

    # ── SOQPSO 優化（與 Qiskit 版完全共用，僅 evaluate_fn 不同）─────────────
    optimizer = QMGSOQPSOOptimizer(
        n_params          = n_flexible,
        n_particles       = args.particles,
        max_iterations    = args.iterations,
        evaluate_fn       = evaluate_fn,
        logger            = logger,
        seed              = args.seed,
        alpha_max         = args.alpha_max,
        alpha_min         = args.alpha_min,
        data_dir          = args.data_dir,
        task_name         = args.task_name,
        stagnation_limit  = args.stagnation_limit,
        mutation_prob     = args.mutation_prob,
    )

    best_params, best_fitness = optimizer.optimize()

    # ── 儲存最佳參數 ─────────────────────────────────────────────────────────
    best_npy = os.path.join(args.data_dir, f"{args.task_name}_best_params.npy")
    np.save(best_npy, best_params)
    logger.info(f"最佳參數已儲存: {best_npy}")
    logger.info(
        f"最終結果: V×U={best_fitness:.6f}  "
        + ("✓ 超越 BO 基線 0.8834!" if best_fitness > 0.8834
           else "✗ 未超越 — 建議增加 --particles 或 --iterations")
    )


if __name__ == "__main__":
    main()