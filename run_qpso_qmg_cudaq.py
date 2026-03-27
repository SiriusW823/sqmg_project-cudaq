"""
==============================================================================
run_qpso_qmg_cudaq.py — CUDA-Q 版 QMG + SOQPSO 主入口
對應 Qiskit 版本：run_qpso_qmg.py
==============================================================================

與 Qiskit 版的差異（僅三處）：
  1. MoleculeGenerator    →  MoleculeGeneratorCUDAQ（from generator_cudaq）
  2. backend_name 預設    →  "cudaq_qpp"（CPU 模擬，對應 qiskit_aer）
                              如有 NVIDIA GPU 可改 "cudaq_custatevec"
  3. 新增 --backend 參數
  其餘所有邏輯、QPSO 超參數、Logger 格式完全不變。

執行方式：
  python run_qpso_qmg_cudaq.py                          # 預設 CPU
  python run_qpso_qmg_cudaq.py --backend cudaq_nvidia   # GPU
  python run_qpso_qmg_cudaq.py --particles 8 --iterations 80
  python run_qpso_qmg_cudaq.py --seed 123
==============================================================================
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    pass

try:
    from weight_generator import ConditionalWeightsGenerator
except ImportError as e:
    print(f"[ERROR] 無法匯入 weight_generator: {e}")
    sys.exit(1)

try:
    from generator_cudaq import MoleculeGeneratorCUDAQ as MoleculeGenerator
except ImportError as e:
    print(f"[ERROR] 無法匯入 generator_cudaq: {e}")
    print("  請確認 generator_cudaq.py 與 build_dynamic_circuit_cudaq.py 在同目錄。")
    sys.exit(1)

from qpso_optimizer_qmg import QMGSOQPSOOptimizer


# ===========================================================================
# 命令列參數
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="QMG + SOQPSO（CUDA-Q 版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--num_heavy_atom',   type=int,   default=9)
    parser.add_argument('--num_sample',       type=int,   default=10000)
    parser.add_argument('--particles',        type=int,   default=5)
    parser.add_argument('--iterations',       type=int,   default=120)
    parser.add_argument('--alpha_max',        type=float, default=1.2)
    parser.add_argument('--alpha_min',        type=float, default=0.4)
    parser.add_argument('--mutation_prob',    type=float, default=0.12)
    parser.add_argument('--stagnation_limit', type=int,   default=8)
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--task_name',  type=str,
                        default='unconditional_9_qpso_cudaq')
    parser.add_argument('--data_dir',   type=str,
                        default='results_qpso_cudaq')
    # CUDA-Q 新增
    parser.add_argument(
        '--backend', type=str, default='cudaq_qpp',
        choices=['cudaq_qpp', 'cudaq_custatevec', 'cudaq_nvidia', 'cudaq_mqpu'],
        help=(
            'CUDA-Q 模擬 backend。\n'
            '  cudaq_qpp        : 本地 CPU（對應 qiskit_aer）\n'
            '  cudaq_custatevec : NVIDIA GPU（需 CUDA + cuStateVec）\n'
            '  cudaq_nvidia     : NVIDIA GPU（別名）\n'
            '  cudaq_mqpu       : 多 GPU 並行'
        ),
    )
    return parser.parse_args()


# ===========================================================================
# Logger（格式與 Qiskit 版完全相同）
# ===========================================================================

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger('QMGQPSOCUDAQLogger')
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter(
        '%(asctime)s,%(msecs)03d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ===========================================================================
# 主流程
# ===========================================================================

def main():
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    log_path = os.path.join(args.data_dir, f"{args.task_name}.log")
    logger   = setup_logger(log_path)

    # ── 啟動資訊（對齊 unconditional_9.log header）──────────────────────
    logger.info(f"Task name: {args.task_name}")
    logger.info(f"Task: ['validity', 'uniqueness']")
    logger.info(f"Condition: ['None', 'None']")
    logger.info(f"objective: ['maximize', 'maximize']")
    logger.info(f"# of heavy atoms: {args.num_heavy_atom}")
    logger.info(f"# of samples: {args.num_sample}")
    logger.info(f"smarts: None")
    logger.info(f"disable_connectivity_position: None")
    import cudaq as _cudaq
    logger.info(f"CUDA-Q backend: {args.backend}")
    logger.info(f"CUDA-Q version: {_cudaq.__version__}")

    # ── ConditionalWeightsGenerator（與 Qiskit 版完全相同）──────────────
    cwg = ConditionalWeightsGenerator(
        args.num_heavy_atom,
        smarts=None,
        disable_connectivity_position=None,
    )
    n_flexible = int((cwg.parameters_indicator == 0.).sum())
    logger.info(f"Number of flexible parameters: {n_flexible}")

    full_weight_vector_base = np.zeros(cwg.length_all_weight_vector)

    logger.info(
        f"[QPSO config] M={args.particles}  T={args.iterations}  "
        f"total_evals={args.particles*(args.iterations+1)}  "
        f"seed={args.seed}  backend={args.backend}"
    )

    # ── 適應度函式（與 Qiskit 版邏輯完全相同，只換 MoleculeGenerator）──
    def evaluate_fn(partial_inputs: np.ndarray) -> tuple:
        inputs = full_weight_vector_base.copy()
        inputs[cwg.parameters_indicator == 0.] = np.clip(partial_inputs, 0.0, 1.0)
        inputs = cwg.apply_chemistry_constraint(inputs)
        mg = MoleculeGenerator(
            args.num_heavy_atom,
            all_weight_vector    = inputs,
            backend_name         = args.backend,  # ← CUDA-Q backend
            dynamic_circuit      = True,
            chemistry_constraint = True,
        )
        smiles_dict, validity, uniqueness = mg.sample_molecule(args.num_sample)
        return float(validity), float(uniqueness)

    # ── SOQPSO（完全與 Qiskit 版相同）──────────────────────────────────
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

    best_npy = os.path.join(args.data_dir, f"{args.task_name}_best_params.npy")
    np.save(best_npy, best_params)
    logger.info(f"最佳參數已儲存: {best_npy}")
    logger.info(
        f"最終結果: V×U={best_fitness:.6f}  "
        + ("✓ 超越 BO 基線 0.8834!" if best_fitness > 0.8834
           else "請嘗試增加 --particles 或 --iterations")
    )


if __name__ == '__main__':
    main()
