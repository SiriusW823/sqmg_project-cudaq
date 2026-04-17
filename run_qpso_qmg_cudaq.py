"""
==============================================================================
run_qpso_qmg_cudaq.py  ─ CUDA-Q 0.7.1 + SOQPSO 主入口（v9.4 記憶體修正版）
==============================================================================
v9.3 → v9.4 記憶體修正清單：

  [MEM-1] evaluate_fn 加入週期性 generator 重建
          CUDA-Q 0.7.1 即使 del result + gc.collect() 後，C++ 端仍有
          累積的 JIT cache / MLIR module 狀態殘留。
          每 REINIT_EVERY 次評估重建一次 MoleculeGeneratorCUDAQ，
          確保 C++ 端累積狀態被完全清除。
          預設 REINIT_EVERY=50（對應約 5000 秒 = 83 分鐘）。

  [MEM-2] evaluate_fn 每次評估後呼叫 gc.collect() + _free_cpp_heap()
          避免 pybind11 C++ 端記憶體在單次評估內洩漏。

  [MEM-3] 加入 log_memory() 監控 RSS
          每個 QPSO iteration 記錄一次 RSS，
          方便確認記憶體增長是否已被控制。

  [MEM-4] 加入 --reinit_every CLI 參數
          允許在不修改程式碼的情況下調整重建頻率。

原 v9.1 修正清單（保留）：
  [FIX-1] 移除 --backend choices 中的 tensornet
  [FIX-2] evaluate_fn 加入 validity=0 的警告日誌
  [FIX-3] 其餘邏輯不變
==============================================================================
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import logging
import os
import re
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
    print("  pip install cuda-quantum-cu11==0.7.1  # CUDA 11.x")
    print("  pip install cuda-quantum-cu12==0.7.1  # CUDA 12.x")
    sys.exit(1)

# ── QMG 套件 ─────────────────────────────────────────────────────────────────
try:
    from qmg.utils import ConditionalWeightsGenerator
except ImportError as e:
    print(f"[ERROR] 無法 import qmg.utils: {e}")
    print("  請確認 qmg/utils/__init__.py 存在且包含正確 import。")
    sys.exit(1)

try:
    from qmg.generator_cudaq import MoleculeGeneratorCUDAQ, _free_cpp_heap
except ImportError as e:
    print(f"[ERROR] 無法 import qmg.generator_cudaq: {e}")
    sys.exit(1)

# ── SOQPSO 優化器 ─────────────────────────────────────────────────────────────
try:
    from qpso_optimizer_qmg import QMGSOQPSOOptimizer
except ImportError as e:
    print(f"[ERROR] 無法 import qpso_optimizer_qmg: {e}")
    sys.exit(1)


# ===========================================================================
# 記憶體監控工具（v9.4 新增）
# ===========================================================================

def _get_rss_mb() -> float:
    """取得目前程序的 RSS（Resident Set Size），單位 MB。"""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        # psutil 不可用時，改讀 /proc/self/status
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024  # kB → MB
        except Exception:
            pass
        return -1.0


def log_memory(logger: logging.Logger, label: str = "") -> float:
    """記錄 RSS 並回傳 MB 值。"""
    rss = _get_rss_mb()
    if rss >= 0:
        logger.info(f"[MEM] {label}  RSS={rss:.0f} MB")
    return rss


# ===========================================================================
# 命令列參數
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QMG CUDA-Q 0.7.1 + SOQPSO 分子生成優化（v9.4 記憶體修正版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 電路參數
    p.add_argument("--num_heavy_atom",   type=int,   default=9,
                   help="重原子數（目前支援 N=9）")
    p.add_argument("--num_sample",       type=int,   default=10000,
                   help="每次電路評估的 shots 數")
    # QPSO 參數
    p.add_argument("--particles",        type=int,   default=50,
                   help="QPSO 粒子數 M")
    p.add_argument("--iterations",       type=int,   default=200,
                   help="QPSO 迭代數 T（總 evals = M×(T+1)）")
    p.add_argument("--alpha_max",        type=float, default=1.5,
                   help="收斂係數上界")
    p.add_argument("--alpha_min",        type=float, default=0.5,
                   help="收斂係數下界")
    p.add_argument("--mutation_prob",    type=float, default=0.12,
                   help="Cauchy 變異機率")
    p.add_argument("--stagnation_limit", type=int,   default=8,
                   help="停滯偵測門檻（QPSO 迭代次數）")
    p.add_argument("--seed",             type=int,   default=42,
                   help="隨機種子")
    # Backend
    p.add_argument(
        "--backend", type=str, default="cudaq_nvidia",
        choices=["cudaq_qpp", "cudaq_nvidia", "cudaq_nvidia_fp64"],
        help=(
            "CUDA-Q 模擬後端。\n"
            "  cudaq_nvidia     : V100 GPU（推薦）\n"
            "  cudaq_nvidia_fp64: V100 GPU（雙精度，較慢）\n"
            "  cudaq_qpp        : CPU（僅供除錯，~90s/eval）\n"
            "  注意：tensornet 對動態電路不相容，已移除。"
        ),
    )
    # 記憶體管理參數（v9.4 新增）
    p.add_argument(
        "--reinit_every", type=int, default=50,
        help=(
            "每隔多少次評估重建一次 MoleculeGeneratorCUDAQ，"
            "以釋放 CUDA-Q C++ 端累積的 JIT cache 狀態。"
            "設為 0 表示停用（不建議，可能導致 OOM）。"
        ),
    )
    # 輸出
    p.add_argument("--task_name", type=str,
                   default="unconditional_9_qpso",
                   help="實驗名稱（用於 log/csv/npy 檔名）")
    p.add_argument("--data_dir",  type=str,
                   default="results_dgx1_gpu_final",
                   help="結果輸出目錄")
    return p.parse_args()


# ===========================================================================
# Logger
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
        ver_str = cudaq.__version__
        match = re.search(r'(\d+\.\d+\.\d+)', ver_str)
        short_ver = match.group(1) if match else ver_str
        logger.info(f"  CUDA-Q version: {short_ver} (full: {ver_str[:60]})")
    except AttributeError:
        logger.info("  CUDA-Q version: unknown")

    try:
        avail = [str(t) for t in cudaq.get_targets()]
        logger.info(f"  Available targets: {avail}")
    except Exception:
        pass


# ===========================================================================
# Generator 工廠（v9.4：供週期性重建使用）
# ===========================================================================

def _build_generator(args: argparse.Namespace,
                     logger: logging.Logger) -> MoleculeGeneratorCUDAQ:
    """建立（或重建）MoleculeGeneratorCUDAQ 實例。"""
    logger.info(
        "[CUDAQ] 初始化 MoleculeGeneratorCUDAQ"
        "（首次 JIT 編譯可能需 10~60s）..."
    )
    t0 = time.time()
    gen = MoleculeGeneratorCUDAQ(
        num_heavy_atom            = args.num_heavy_atom,
        backend_name              = args.backend,
        remove_bond_disconnection = True,
        chemistry_constraint      = True,
    )
    logger.info(f"[CUDAQ] 初始化完成，耗時 {time.time() - t0:.1f}s")
    return gen


# ===========================================================================
# 主流程
# ===========================================================================

def main() -> None:
    args = parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    log_path = os.path.join(args.data_dir, f"{args.task_name}.log")
    logger   = setup_logger(log_path)

    # ── 啟動資訊 ─────────────────────────────────────────────────────────
    logger.info(f"Task name: {args.task_name}")
    logger.info(f"Task: ['validity', 'uniqueness']")
    logger.info(f"Condition: ['None', 'None']")
    logger.info(f"objective: ['maximize', 'maximize']")
    logger.info(f"# of heavy atoms: {args.num_heavy_atom}")
    logger.info(f"# of samples: {args.num_sample}")
    logger.info(f"smarts: None")
    logger.info(f"disable_connectivity_position: []")
    logger.info(f"CUDA-Q backend: {args.backend}")
    logger.info(f"[v9.4] reinit_every: {args.reinit_every} evals")
    log_gpu_info(logger)
    log_memory(logger, "啟動時")

    # ── ConditionalWeightsGenerator ──────────────────────────────────────
    cwg = ConditionalWeightsGenerator(
        args.num_heavy_atom,
        smarts=None,
        disable_connectivity_position=[],
    )
    n_flexible = int((cwg.parameters_indicator == 0.0).sum())
    logger.info(f"Number of flexible parameters: {n_flexible}")

    assert n_flexible == cwg.length_all_weight_vector, (
        f"[BUG] n_flexible={n_flexible} != "
        f"length_all_weight_vector={cwg.length_all_weight_vector}"
    )

    total_evals = args.particles * (args.iterations + 1)
    logger.info(
        f"[CUDAQ-QPSO config] "
        f"M={args.particles}  T={args.iterations}  "
        f"total_evals={total_evals}  seed={args.seed}  "
        f"backend={args.backend}"
    )

    # ── 初始建立 generator ────────────────────────────────────────────────
    generator = _build_generator(args, logger)
    log_memory(logger, "Generator 初始化後")

    # ── Evaluate function ─────────────────────────────────────────────────
    # 使用 list 包裝以允許閉包中重新賦值（避免 Python 3 nonlocal 需求）
    _state = {
        "generator":   generator,
        "eval_count":  0,
    }

    def evaluate_fn(pos: np.ndarray) -> tuple:
        """
        pos : shape=(n_flexible=134,)，值域 [0,1]。
        apply_chemistry_constraint 套用後作為完整 134-dim weight vector 送入電路。
        """
        if len(pos) != n_flexible:
            raise ValueError(
                f"[evaluate_fn] pos 維度錯誤：got {len(pos)}, expected {n_flexible}"
            )

        eval_cnt = _state["eval_count"]

        # ★ v9.4 MEM-1：週期性重建 generator，清除 C++ JIT cache 累積狀態
        if args.reinit_every > 0 and eval_cnt > 0 and eval_cnt % args.reinit_every == 0:
            rss_before = _get_rss_mb()
            logger.info(
                f"[MEM] 週期性重建 generator（eval #{eval_cnt}）  "
                f"RSS={rss_before:.0f} MB → 開始清理..."
            )
            # 明確刪除舊 generator，觸發 C++ 析構
            del _state["generator"]
            gc.collect()
            _free_cpp_heap()
            rss_mid = _get_rss_mb()
            logger.info(f"[MEM] 舊 generator 已釋放  RSS={rss_mid:.0f} MB")

            # 重建
            _state["generator"] = _build_generator(args, logger)
            rss_after = _get_rss_mb()
            logger.info(
                f"[MEM] 重建完成  RSS={rss_after:.0f} MB  "
                f"（釋放了 {rss_before - rss_after:.0f} MB）"
            )

        _state["eval_count"] += 1

        # ── 化學約束套用 ───────────────────────────────────────────────
        w = cwg.apply_chemistry_constraint(pos.copy())
        _state["generator"].update_weight_vector(w)

        # ── 電路採樣（記憶體釋放已在 sample_molecule 內部完成）────────
        _, validity, uniqueness = _state["generator"].sample_molecule(args.num_sample)

        # ★ v9.4 MEM-2：每次評估後額外呼叫 gc + malloc_trim
        #   sample_molecule 內部已呼叫，這裡是雙重保險
        gc.collect()
        _free_cpp_heap()

        # ★ v9.4 FIX-2：validity=0 警告
        if validity == 0.0:
            logger.warning(
                "[evaluate_fn] validity=0.0 — 可能原因：\n"
                "  1. 命名暫存器未正確識別（確認 v9.1 build_dynamic_circuit_cudaq.py）\n"
                "  2. 所有 shots 均產生無效分子（優化早期正常現象）\n"
                "  3. raw_counts 為空（檢查 generator 初始化日誌）"
            )

        return float(validity), float(uniqueness)

    # ── SOQPSO 優化 ───────────────────────────────────────────────────────
    # 包裝 optimizer 以便在每個 iteration 結束後記錄記憶體
    class MemoryAwareOptimizer(QMGSOQPSOOptimizer):
        """繼承 QMGSOQPSOOptimizer，在每個 iteration 後記錄 RSS。"""

        def optimize(self):
            # Phase 0：初始評估
            total_evals = self.M * (self.T + 1)
            self.logger.info("=" * 65)
            self.logger.info("SOQPSO 量子粒子群優化啟動（v9.4 記憶體監控版）")
            self.logger.info(f"  粒子數 M            : {self.M}")
            self.logger.info(f"  參數維度 D           : {self.D}")
            self.logger.info(f"  最大迭代 T           : {self.T}")
            self.logger.info(f"  總評估次數           : {total_evals}")
            self.logger.info(f"  α 排程              : [{self.alpha_min}, {self.alpha_max}] cosine")
            self.logger.info(f"  停滯門檻             : {self.stagnation_limit} QPSO iters")
            self.logger.info(f"  Cauchy 變異機率      : {self.mutation_prob:.0%}")
            self.logger.info(f"  BO 基線 (Best V×U)  : 0.8834")
            self.logger.info("=" * 65)

            self.logger.info("[Phase 0] 初始粒子評估（隨機初始化）...")
            for i in range(self.M):
                v, u, f = self._eval_particle(
                    self.positions[i], self._global_eval_cnt, 0, i, self._get_alpha(0)
                )
                self._global_eval_cnt += 1
                self._update_pbest(i, f)
                self._update_gbest(i, f, v, u)
            self._prev_best = self.gbest_fit
            log_memory(logger, f"Phase 0 結束（{self.M} evals）")

            # 主迭代
            import math as _math
            for t in range(self.T):
                alpha = self._get_alpha(t)
                mbest = np.mean(self.pbest, axis=0)
                gbest = self.gbest_pos if self.gbest_pos is not None else self.positions[0]

                iter_fits = []
                for i in range(self.M):
                    self.positions[i] = self._update_pos(
                        self.positions[i], self.pbest[i], gbest, mbest, alpha
                    )
                    if self.rng.random() < self.mutation_prob:
                        self.positions[i] = self._cauchy_mutation(self.positions[i])
                        self._total_mutations += 1

                    v, u, f = self._eval_particle(
                        self.positions[i], self._global_eval_cnt, t + 1, i, alpha
                    )
                    self._global_eval_cnt += 1
                    iter_fits.append(f)
                    self._update_pbest(i, f)
                    self._update_gbest(i, f, v, u)

                self._update_stagnation(self.gbest_fit)
                self._maybe_reinit()

                mean_fit = float(np.mean(iter_fits))
                max_fit  = float(np.max(iter_fits))
                self.history.append({
                    'qpso_iter':        t + 1,
                    'n_evals':          self._global_eval_cnt,
                    'gbest_fitness':    self.gbest_fit,
                    'gbest_validity':   self.gbest_val,
                    'gbest_uniqueness': self.gbest_uniq,
                    'mean_fitness':     mean_fit,
                    'max_fitness':      max_fit,
                    'alpha':            alpha,
                })
                self.logger.info(
                    f"  [QPSO Iter {t+1:3d}/{self.T}] "
                    f"α={alpha:.3f}  "
                    f"gbest={self.gbest_fit:.4f} (V={self.gbest_val:.3f} U={self.gbest_uniq:.3f})  "
                    f"mean={mean_fit:.4f}  max={max_fit:.4f}  "
                    f"stag={self._stag_counter}  evals={self._global_eval_cnt}"
                )
                # ★ v9.4 MEM-3：每個 iteration 記錄 RSS
                log_memory(logger, f"Iter {t+1}/{self.T}")

            # 最終摘要
            self.logger.info("=" * 65)
            self.logger.info("SOQPSO 優化完成")
            self.logger.info(f"  Best V×U   : {self.gbest_fit:.6f}")
            self.logger.info(f"  Best V     : {self.gbest_val:.4f}")
            self.logger.info(f"  Best U     : {self.gbest_uniq:.4f}")
            self.logger.info(
                f"  BO Baseline: 0.8834  "
                + ("✓ 超越基線!" if self.gbest_fit > 0.8834
                   else "✗ 未超越 — 建議增加粒子數或迭代次數")
            )
            self.logger.info(f"  Total evals: {self._global_eval_cnt}")
            self.logger.info(f"  Reinits    : {self._total_reinits}")
            self.logger.info(f"  Mutations  : {self._total_mutations}")
            self.logger.info("=" * 65)
            log_memory(logger, "優化完成")

            best = self.gbest_pos.copy() if self.gbest_pos is not None else np.zeros(self.D)
            return best, self.gbest_fit

    optimizer = MemoryAwareOptimizer(
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

    # ── 儲存最佳參數 ─────────────────────────────────────────────────────
    best_npy = os.path.join(args.data_dir, f"{args.task_name}_best_params.npy")
    np.save(best_npy, best_params)
    logger.info(f"最佳參數已儲存: {best_npy}")
    logger.info(
        f"最終結果: V×U={best_fitness:.6f}  "
        + ("✓ 超越 BO 基線 0.8834!" if best_fitness > 0.8834
           else "✗ 未超越 — 建議增加 --particles 或 --iterations")
    )
    log_memory(logger, "程序結束前")


if __name__ == "__main__":
    main()