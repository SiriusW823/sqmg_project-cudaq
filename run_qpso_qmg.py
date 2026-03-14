"""
==============================================================================
run_qpso_qmg.py — QMG 量子電路 + SOQPSO 主入口
==============================================================================

使用 Chen et al. 2025 QMG 電路（原版 Qiskit 動態電路），
將 Bayesian Optimization（GPEI）替換為 SOQPSO，
在相同計算預算下超越 unconditional_9.log 的基線。

基線（unconditional_9.log）最佳結果：
  Best V×U = 0.8834  (V=0.955, U=0.925)，出現於第 355 次評估
  需要 ~300 次評估才收斂至 V×U > 0.80 穩定域

實驗設置（完全對齊 unconditional_9.log）：
  num_heavy_atom = 9
  num_sample     = 10000
  task           = validity + uniqueness
  objective      = maximize × maximize
  smarts         = None（無條件生成）
  chemistry_constraint = ON

QPSO 計算預算設計：
  M=5 粒子 × 120 迭代 = 605 total evals（含 Phase 0）
  每個粒子評估 = 一個 "Iteration number: X"（對齊 BO log 格式）
  預期：QPSO 群體智慧在 ~100 次評估內收斂至 V×U ≈ 0.85+
        相比 BO 需要 ~200 次評估才能穩定達到此水準

執行方式：
  python run_qpso_qmg.py                    # 預設參數
  python run_qpso_qmg.py --particles 8 --iterations 80  # 調整粒子數
  python run_qpso_qmg.py --seed 123         # 不同隨機種子
==============================================================================
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

# RDKit 靜音（必須在 qmg 匯入前）
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    pass

# QMG 套件匯入（qmg/ 目錄必須在當前目錄或 PYTHONPATH 中）
try:
    from qmg.generator import MoleculeGenerator
    from qmg.utils     import ConditionalWeightsGenerator
except ImportError as e:
    print(f"[ERROR] 無法匯入 qmg 套件: {e}")
    print("  請確認 qmg/ 目錄存在於當前目錄，或已正確設定 PYTHONPATH。")
    sys.exit(1)

from qpso_optimizer_qmg import QMGSOQPSOOptimizer


# ===========================================================================
# 命令列參數
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="QMG + SOQPSO: 超越 BO 基線的量子分子生成實驗",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 對齊 unconditional_9.log 的核心參數
    parser.add_argument('--num_heavy_atom', type=int,   default=9,
                        help='重原子數（對應 log 的 # of heavy atoms: 9）')
    parser.add_argument('--num_sample',     type=int,   default=10000,
                        help='每次電路取樣的 shots（對應 log 的 # of samples: 10000）')
    # QPSO 超參數
    parser.add_argument('--particles',      type=int,   default=5,
                        help='粒子數 M（預設 5，×120+5 = 605 total evals）')
    parser.add_argument('--iterations',     type=int,   default=120,
                        help='QPSO 最大迭代次數 T')
    parser.add_argument('--alpha_max',      type=float, default=1.2)
    parser.add_argument('--alpha_min',      type=float, default=0.4)
    parser.add_argument('--mutation_prob',  type=float, default=0.12,
                        help='Cauchy 變異機率（per particle per iter）')
    parser.add_argument('--stagnation_limit', type=int, default=8,
                        help='停滯偵測門檻（QPSO 迭代數）')
    parser.add_argument('--seed',           type=int,   default=42)
    # 輸出設置
    parser.add_argument('--task_name',  type=str, default='unconditional_9_qpso',
                        help='實驗名稱，用於 log/CSV 檔名（對齊 BO 命名慣例）')
    parser.add_argument('--data_dir',   type=str, default='results_qpso',
                        help='結果輸出目錄')
    return parser.parse_args()


# ===========================================================================
# Logger（時間戳格式完全對齊 unconditional_9.log）
# ===========================================================================

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger('QMGQPSOLogger')
    logger.setLevel(logging.INFO)
    # 防止重複 handler（重複呼叫時）
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        '%(asctime)s,%(msecs)03d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # File handler
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    # Console handler
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

    # ── 記錄啟動資訊（完全對齊 unconditional_9.log header 格式）────
    logger.info(f"Task name: {args.task_name}")
    logger.info(f"Task: ['validity', 'uniqueness']")
    logger.info(f"Condition: ['None', 'None']")
    logger.info(f"objective: ['maximize', 'maximize']")
    logger.info(f"# of heavy atoms: {args.num_heavy_atom}")
    logger.info(f"# of samples: {args.num_sample}")
    logger.info(f"smarts: None")
    logger.info(f"disable_connectivity_position: None")

    # CUDA 是否可用（BoTorch GPU 用；Aer 電路模擬用 CPU）
    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except ImportError:
        use_cuda = False
    logger.info(f"Using cuda: {use_cuda}")

    # ── 初始化 ConditionalWeightsGenerator ───────────────────────
    # 無條件生成（smarts=None）：parameters_indicator 全為 0，
    # 即 ALL parameters 均為 flexible → n_flexible = length_all_weight_vector = 134
    cwg = ConditionalWeightsGenerator(
        args.num_heavy_atom,
        smarts=None,
        disable_connectivity_position=None,
    )
    n_flexible = int((cwg.parameters_indicator == 0.).sum())
    logger.info(f"Number of flexible parameters: {n_flexible}")

    # fixed part（smarts=None 時全為 0）
    full_weight_vector_base = np.zeros(cwg.length_all_weight_vector)

    logger.info(
        f"[QPSO config] M={args.particles}  T={args.iterations}  "
        f"total_evals={args.particles*(args.iterations+1)}  seed={args.seed}"
    )

    # ── 適應度函式（與 constrained_bo.py 的 evaluate() 等效）───────
    def evaluate_fn(partial_inputs: np.ndarray) -> tuple:
        """
        輸入：QPSO 粒子位置（134 維，[0,1]）
        輸出：(validity, uniqueness)

        流程（與 BO 的 evaluate() 完全一致）：
          1. 將 partial_inputs clip 至 [0,1]
          2. 填入 full_weight_vector 的 flexible 位置
          3. 套用 chemistry constraint（對應 BO 預設行為）
          4. 建立 MoleculeGenerator 並呼叫 sample_molecule
        """
        inputs = full_weight_vector_base.copy()
        inputs[cwg.parameters_indicator == 0.] = np.clip(partial_inputs, 0.0, 1.0)
        inputs = cwg.apply_chemistry_constraint(inputs)

        mg = MoleculeGenerator(
            args.num_heavy_atom,
            all_weight_vector=inputs,
            backend_name='qiskit_aer',   # 本地 Aer 模擬（非 IBM hardware）
            dynamic_circuit=True,         # 動態電路（N=9 用 20 qubits，非 90）
            chemistry_constraint=True,
        )
        smiles_dict, validity, uniqueness = mg.sample_molecule(args.num_sample)
        return float(validity), float(uniqueness)

    # ── 初始化並執行 SOQPSO ──────────────────────────────────────
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

    # ── 儲存最佳參數 ─────────────────────────────────────────────
    best_npy = os.path.join(args.data_dir, f"{args.task_name}_best_params.npy")
    np.save(best_npy, best_params)
    logger.info(f"最佳參數已儲存: {best_npy}")
    logger.info(
        f"最終結果: V×U={best_fitness:.6f}  "
        f"{'✓ 超越 BO 基線 0.8834!' if best_fitness > 0.8834 else '請嘗試增加 --particles 或 --iterations'}"
    )


if __name__ == '__main__':
    main()
