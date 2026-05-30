"""
==============================================================================
qpso_optimizer_ae.py
AE-SOQPSO v1.2 — Amplitude-Ensemble Single-Objective Quantum PSO
融合 AE-QTS（arXiv:2311.12867v2）的調和加權與配對更新機制
支援單粒子與批次兩種評估模式

v1.1 → v1.2 新增功能：

  ★ [NEW] Opposition-Based Learning (OBL) Phase 0
      在 Phase 0 評估完成後，額外評估每個粒子的對立粒子 x' = 1 - x。
      若對立粒子的 fitness 更好則替換，讓初始探索等效覆蓋翻倍。
      僅支援 batch_evaluate_fn 模式（parallel evaluator）。
      由 obl=True 控制，可透過 --no_obl 關閉。

  ★ [NEW] V-U 解耦 mbest（V-U Decoupled mbest）
      在標準 U 形調和加權 mbest 基礎上，額外加入：
        - _best_v_pos：歷史最高 validity 對應的粒子位置
        - _best_u_pos：歷史最高 uniqueness 對應的粒子位置
      加權比例：w_vu=0.70, w_v=0.15, w_u=0.15
      解決高維空間中 V 最優解與 U 最優解分離的聯合最優問題。
      由 vu_decouple=True 控制，可透過 --no_vu_decouple 關閉。

  v1.1 修正保留（不變）：
    - U 形對稱調和加權 mbest（對照 AE-QTS Algorithm 3）
    - AE-QTS Best-Worst 配對更新（兩端→attractor，步長 rotate_factor/k）
    - 停滯偵測與重初始化
    - 批次評估介面（parallel evaluator 支援）
    - 雙目標解耦監控（V_best_ever, U_best_ever）

參考文獻：
  [1] Tseng et al., AE-QTS, arXiv:2311.12867v2, 2024
  [2] Sun et al., QPSO, CEC 2012
  [3] Chen et al., QMG, JCTC 2025
  [4] Xiao et al., SQMG, arXiv:2604.13877v1, 2026
  [5] Tizhoosh, H.R., Opposition-Based Learning, ISDA 2005
==============================================================================
"""
from __future__ import annotations

import csv
import logging
import math
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class AESOQPSOOptimizer:
    """
    Amplitude-Ensemble Single-Objective QPSO（v1.2 OBL + V-U Decouple 版）。

    評估模式（擇一傳入）：
      evaluate_fn:       (pos: np.ndarray[D]) → (validity, uniqueness)
      batch_evaluate_fn: (positions: np.ndarray[M,D]) → list[(v,u)]

    新增參數：
      obl:          是否在 Phase 0 執行 Opposition-Based Learning（僅 batch 模式）
      vu_decouple:  是否啟用 V-U 解耦 mbest
      w_vu:         標準 U 形加權 mbest 的權重（預設 0.70）
      w_v:          V* 位置的牽引權重（預設 0.15）
      w_u:          U* 位置的牽引權重（預設 0.15）
    """

    def __init__(
        self,
        n_params:            int,
        n_particles:         int,
        max_iterations:      int,
        logger:              logging.Logger,
        evaluate_fn:         Optional[Callable[[np.ndarray], Tuple[float, float]]] = None,
        batch_evaluate_fn:   Optional[Callable[[np.ndarray], List[Tuple[float, float]]]] = None,
        seed:                int   = 42,
        alpha_max:           float = 1.2,
        alpha_min:           float = 0.4,
        data_dir:            str   = "results_ae_qpso",
        task_name:           str   = "unconditional_9_ae_qpso",
        stagnation_limit:    int   = 12,
        reinit_fraction:     float = 0.25,
        mutation_prob:       float = 0.15,
        mutation_scale:      float = 0.15,
        alpha_perturb_std:   float = 0.04,
        alpha_stag_boost:    float = 0.20,
        ae_weighting:        bool  = True,
        pair_interval:       int   = 5,
        rotate_factor:       float = 0.015,
        # ★ v1.2 新增
        obl:                 bool  = True,
        vu_decouple:         bool  = True,
        w_vu:                float = 0.70,
        w_v:                 float = 0.15,
        w_u:                 float = 0.15,
    ):
        if evaluate_fn is None and batch_evaluate_fn is None:
            raise ValueError("必須提供 evaluate_fn 或 batch_evaluate_fn 其中一個。")

        self.D                 = n_params
        self.M                 = n_particles
        self.T                 = max_iterations
        self.logger            = logger
        self.evaluate_fn       = evaluate_fn
        self.batch_evaluate_fn = batch_evaluate_fn
        self.alpha_max         = alpha_max
        self.alpha_min         = alpha_min
        self.data_dir          = data_dir
        self.task_name         = task_name
        self.stagnation_limit  = stagnation_limit
        self.reinit_fraction   = reinit_fraction
        self.mutation_prob     = mutation_prob
        self.mutation_scale    = mutation_scale
        self.alpha_perturb_std = alpha_perturb_std
        self.alpha_stag_boost  = alpha_stag_boost
        self.ae_weighting      = ae_weighting
        self.pair_interval     = pair_interval
        self.rotate_factor     = rotate_factor
        # ★ v1.2
        self.obl               = obl
        self.vu_decouple       = vu_decouple
        self.w_vu              = w_vu
        self.w_v               = w_v
        self.w_u               = w_u

        self.lb = np.zeros(self.D, dtype=np.float64)
        self.ub = np.ones(self.D,  dtype=np.float64)
        self._mut_range = mutation_scale * (self.ub - self.lb)

        self.rng       = np.random.default_rng(seed)
        self.positions = self._rand_pos(self.M)
        self.pbest     = self.positions.copy()
        self.pbest_fit = np.full(self.M, -np.inf)

        self.gbest_pos  = None
        self.gbest_fit  = -np.inf
        self.gbest_val  = 0.0
        self.gbest_uniq = 0.0
        self._best_v_ever = 0.0
        self._best_u_ever = 0.0

        # ★ v1.2：V* 和 U* 位置追蹤
        self._best_v_pos: Optional[np.ndarray] = None
        self._best_u_pos: Optional[np.ndarray] = None

        self.history:            List[Dict] = []
        self._stag_counter       = 0
        self._prev_best          = -np.inf
        self._global_eval_cnt    = 0
        self._total_reinits      = 0
        self._total_mutations    = 0
        self._total_ae_updates   = 0
        self._total_obl_replaced = 0   # ★ v1.2

        os.makedirs(data_dir, exist_ok=True)
        self._csv_path = os.path.join(data_dir, f"{task_name}.csv")
        self._init_csv()

    # ================================================================
    # 基礎工具
    # ================================================================

    def _rand_pos(self, n: int) -> np.ndarray:
        return self.lb + self.rng.random((n, self.D)) * (self.ub - self.lb)

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lb, self.ub)

    def _get_alpha(self, t: int) -> float:
        progress = t / max(self.T - 1, 1)
        base = (self.alpha_min
                + 0.5 * (self.alpha_max - self.alpha_min)
                * (1.0 + math.cos(math.pi * progress)))
        perturb = self.rng.normal(0.0, self.alpha_perturb_std)
        boost   = self.alpha_stag_boost if self._stag_counter >= self.stagnation_limit else 0.0
        return float(np.clip(base + perturb + boost,
                             self.alpha_min,
                             self.alpha_max + self.alpha_stag_boost))

    def _update_pos_single(self, x, pbest_i, gbest, mbest, alpha) -> np.ndarray:
        """Delta 勢阱位置更新（Sun et al. 2012 Eq.12）。"""
        phi       = self.rng.uniform(0.0, 1.0, size=self.D)
        attractor = phi * pbest_i + (1.0 - phi) * gbest
        u         = np.maximum(self.rng.uniform(0.0, 1.0, size=self.D), 1e-10)
        step      = alpha * np.abs(mbest - x) * np.log(1.0 / u)
        sign      = np.where(self.rng.uniform(0.0, 1.0, size=self.D) < 0.5, 1.0, -1.0)
        return self._clip(attractor + sign * step)

    def _cauchy_mutation(self, x: np.ndarray) -> np.ndarray:
        """Cauchy 重尾變異（SOQPSO 自身機制，非 AE 配對）。"""
        x_mut = x.copy()
        n_dim = max(1, int(self.D * self.rng.uniform(0.15, 0.35)))
        dims  = self.rng.choice(self.D, size=n_dim, replace=False)
        noise = self.rng.standard_cauchy(size=n_dim) * self._mut_range[dims]
        x_mut[dims] += noise
        return self._clip(x_mut)

    # ================================================================
    # ★ v1.2 Opposition-Based Learning
    # ================================================================

    def _run_obl_phase0(self) -> None:
        """
        Phase 0 結束後評估對立粒子 x' = clip(1 - x, 0, 1)。
        若對立粒子 fitness 更好則替換。
        僅在 batch_evaluate_fn 模式下執行。
        """
        if self.batch_evaluate_fn is None:
            self.logger.info("[OBL] 跳過（僅支援 batch_evaluate_fn 模式）")
            return

        self.logger.info(
            f"[OBL v1.2] Phase 0 對立粒子評估（{self.M} 個對立位置）..."
        )
        obl_positions = self._clip(1.0 - self.positions)
        t_obl = time.time()
        obl_results = self.batch_evaluate_fn(obl_positions)
        elapsed_obl = time.time() - t_obl

        n_replaced = 0
        alpha0 = self._get_alpha(0)
        for i, (v, u) in enumerate(obl_results):
            f_obl = float(v) * float(u)
            # 記錄 OBL 評估（particle_id = M + i 以示區別）
            self._log_eval(
                self._global_eval_cnt, 0, self.M + i,
                v, u, f_obl, alpha0, elapsed_obl
            )
            self._global_eval_cnt += 1

            if f_obl > self.pbest_fit[i]:
                self.positions[i]  = obl_positions[i].copy()
                self.pbest[i]      = obl_positions[i].copy()
                self.pbest_fit[i]  = f_obl
                self._update_gbest(i, f_obl, v, u)
                n_replaced += 1

        self._total_obl_replaced += n_replaced
        self.logger.info(
            f"[OBL v1.2] 完成（{elapsed_obl:.1f}s），"
            f"替換 {n_replaced}/{self.M} 個粒子，"
            f"gbest={self.gbest_fit:.4f}"
        )

    # ================================================================
    # ★ v1.2 AE-QTS + V-U Decoupled mbest
    # ================================================================

    def _ae_weighted_mbest(self) -> np.ndarray:
        """
        ★ v1.2：AE-QTS U 形對稱調和加權 mbest + V-U 解耦牽引。

        標準部分（v1.1 不變）：
          第 k 對 (best_k, worst_k) 各貢獻 1/k，形成 U 形加權分佈。

        V-U 解耦牽引（v1.2，由 vu_decouple 控制）：
          mbest = (w_vu * mbest_standard
                 + w_v  * _best_v_pos
                 + w_u  * _best_u_pos) / total_w
        """
        sorted_idx = np.argsort(self.pbest_fit)[::-1]
        half = self.M // 2
        weights = np.zeros(self.M, dtype=np.float64)

        for k in range(1, half + 1):
            w_k = 1.0 / k
            weights[k - 1]      += w_k
            weights[self.M - k] += w_k

        total = weights.sum()
        weights = weights / total if total > 0 else np.full(self.M, 1.0 / self.M)

        mbest_standard = np.sum(
            self.pbest[sorted_idx] * weights[:, np.newaxis], axis=0
        )

        if not self.vu_decouple:
            return mbest_standard

        components = [self.w_vu * mbest_standard]
        active_w   = self.w_vu

        if self._best_v_pos is not None:
            components.append(self.w_v * self._best_v_pos)
            active_w += self.w_v
        if self._best_u_pos is not None:
            components.append(self.w_u * self._best_u_pos)
            active_w += self.w_u

        return sum(components) / active_w

    def _ae_paired_update(self, alpha: float):
        """
        AE-QTS Best-Worst 配對更新（v1.1 不變）。
        Best 和 Worst 均向各自的 local attractor 移動，幅度 rotate_factor/k。
        """
        if self.gbest_pos is None:
            return

        half     = self.M // 2
        sorted_i = np.argsort(self.pbest_fit)[::-1]

        for k in range(1, half + 1):
            best_idx  = sorted_i[k - 1]
            worst_idx = sorted_i[self.M - k]
            step = self.rotate_factor / k

            for idx in (best_idx, worst_idx):
                phi       = self.rng.uniform(0.0, 1.0, size=self.D)
                attractor = phi * self.pbest[idx] + (1.0 - phi) * self.gbest_pos
                direction = attractor - self.positions[idx]
                self.positions[idx] = self._clip(
                    self.positions[idx] + step * direction
                )

        self._total_ae_updates += 1

    # ================================================================
    # 停滯偵測與重初始化
    # ================================================================

    def _update_stagnation(self, score: float):
        if score > self._prev_best + 1e-8:
            self._stag_counter = 0
        else:
            self._stag_counter += 1
        self._prev_best = score

    def _maybe_reinit(self):
        if self._stag_counter < self.stagnation_limit:
            return
        n_reinit  = max(1, int(self.M * self.reinit_fraction))
        worst_idx = np.argsort(self.pbest_fit)[:n_reinit]
        for off, idx in enumerate(worst_idx):
            if off < n_reinit // 2 or self.gbest_pos is None:
                new_pos = self._rand_pos(1)[0]
            else:
                noise   = self.rng.normal(0.0, 0.10, size=self.D) * (self.ub - self.lb)
                new_pos = self._clip(self.gbest_pos + noise)
            self.positions[idx] = new_pos
            self.pbest[idx]     = new_pos.copy()
            self.pbest_fit[idx] = -np.inf
        self._stag_counter   = 0
        self._total_reinits += 1
        self.logger.info(
            f"  [停滯偵測] 重初始化 {n_reinit} 個粒子"
            f"（累計第 {self._total_reinits} 次）"
        )

    # ================================================================
    # pbest / gbest 更新
    # ================================================================

    def _update_pbest(self, i: int, fit: float):
        if fit > self.pbest_fit[i]:
            self.pbest[i]     = self.positions[i].copy()
            self.pbest_fit[i] = fit

    def _update_gbest(self, i: int, fit: float, val: float, uniq: float):
        if fit > self.gbest_fit:
            self.gbest_pos  = self.positions[i].copy()
            self.gbest_fit  = fit
            self.gbest_val  = val
            self.gbest_uniq = uniq
            self.logger.info(
                f"  🔥 New gbest!  V={val:.4f}  U={uniq:.4f}  V×U={fit:.4f}"
                f"{'  ✓ 超越 BO 基線 0.8834!' if fit > 0.8834 else ''}"
            )
        # ★ v1.2：追蹤 V* 和 U* 的粒子位置
        if val > self._best_v_ever:
            self._best_v_ever = val
            self._best_v_pos  = self.positions[i].copy()
        if uniq > self._best_u_ever:
            self._best_u_ever = uniq
            self._best_u_pos  = self.positions[i].copy()

    # ================================================================
    # CSV 記錄
    # ================================================================

    _CSV_FIELDS = [
        'eval_index', 'qpso_iter', 'particle',
        'validity', 'uniqueness', 'fitness',
        'gbest_fitness', 'best_v_ever', 'best_u_ever',
        'alpha', 'stagnation', 'elapsed_s',
    ]

    def _init_csv(self):
        with open(self._csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=self._CSV_FIELDS).writeheader()

    def _write_csv(self, row: dict):
        with open(self._csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=self._CSV_FIELDS).writerow(row)

    def _log_eval(
        self, eval_label: int, qpso_iter: int, particle_id: int,
        v: float, u: float, f: float, alpha: float, elapsed: float,
    ):
        self.logger.info(f"Iteration number: {eval_label}")
        self.logger.info(f"validity (maximize): {v:.3f}")
        self.logger.info(f"uniqueness (maximize): {u:.3f}")
        self.logger.info(
            f"  [AE-QPSO v1.2] iter={qpso_iter}  p={particle_id}  "
            f"fit={f:.4f}  gbest={self.gbest_fit:.4f}  "
            f"stag={self._stag_counter}  α={alpha:.4f}  t={elapsed:.1f}s  "
            f"[V⋆={self._best_v_ever:.3f} U⋆={self._best_u_ever:.3f}]"
        )
        self._write_csv({
            'eval_index':    eval_label,
            'qpso_iter':     qpso_iter,
            'particle':      particle_id,
            'validity':      round(v, 4),
            'uniqueness':    round(u, 4),
            'fitness':       round(f, 6),
            'gbest_fitness': round(self.gbest_fit, 6),
            'best_v_ever':   round(self._best_v_ever, 4),
            'best_u_ever':   round(self._best_u_ever, 4),
            'alpha':         round(alpha, 4),
            'stagnation':    self._stag_counter,
            'elapsed_s':     round(elapsed, 1),
        })

    # ================================================================
    # 主優化迴圈
    # ================================================================

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        執行 AE-SOQPSO 優化（v1.2）。

        Phase 0：評估 M 個初始粒子；若 obl=True 額外評估 M 個對立粒子。
        主迭代：位置更新 → AE 配對更新（每 pair_interval）→ 評估 → 停滯偵測。
        """
        use_batch   = self.batch_evaluate_fn is not None
        obl_evals   = self.M if (self.obl and use_batch) else 0
        total_evals = self.M * (self.T + 1) + obl_evals

        self.logger.info("=" * 70)
        self.logger.info("AE-SOQPSO（v1.2 OBL + V-U Decouple）優化啟動")
        self.logger.info(f"  粒子數 M                : {self.M}")
        self.logger.info(f"  參數維度 D               : {self.D}")
        self.logger.info(f"  最大迭代 T               : {self.T}")
        self.logger.info(f"  總評估次數（含 OBL）      : {total_evals}")
        self.logger.info(f"  評估模式                 : {'批次（parallel evaluator）' if use_batch else '逐粒子'}")
        self.logger.info(f"  α 排程                  : [{self.alpha_min}, {self.alpha_max}] cosine")
        self.logger.info(f"  AE 加權 mbest（U形）     : {'✓' if self.ae_weighting else '✗'}")
        self.logger.info(
            f"  V-U 解耦 mbest（v1.2）  : "
            f"{'✓ w_vu={:.2f} w_v={:.2f} w_u={:.2f}'.format(self.w_vu, self.w_v, self.w_u) if self.vu_decouple else '✗'}"
        )
        self.logger.info(f"  OBL Phase 0（v1.2）     : {'✓' if (self.obl and use_batch) else '✗'}")
        self.logger.info(
            f"  AE 配對更新             : "
            f"{'✓ 間隔' + str(self.pair_interval) + '迭代  rotate_factor=' + str(self.rotate_factor) if self.pair_interval > 0 else '✗'}"
        )
        self.logger.info(f"  停滯門檻                 : {self.stagnation_limit} iters")
        self.logger.info(f"  Cauchy 變異機率          : {self.mutation_prob:.0%}")
        self.logger.info(f"  BO 基線（Chen 2025）    : 0.8834 (V=0.955, U=0.925)")
        self.logger.info("=" * 70)

        # ── Phase 0：初始化評估 ────────────────────────────────────────────
        self.logger.info("[Phase 0] 初始粒子評估...")
        t0 = time.time()

        if use_batch:
            batch_results = self.batch_evaluate_fn(self.positions)
            elapsed_batch = time.time() - t0
            for i, (v, u) in enumerate(batch_results):
                f = float(v) * float(u)
                self._update_pbest(i, f)
                self._update_gbest(i, f, v, u)
                self._log_eval(self._global_eval_cnt, 0, i, v, u, f,
                               self._get_alpha(0), elapsed_batch)
                self._global_eval_cnt += 1
        else:
            for i in range(self.M):
                t_i = time.time()
                v, u = self.evaluate_fn(self.positions[i])
                f    = float(v) * float(u)
                elapsed = time.time() - t_i
                self._update_pbest(i, f)
                self._update_gbest(i, f, v, u)
                self._log_eval(self._global_eval_cnt, 0, i, v, u, f,
                               self._get_alpha(0), elapsed)
                self._global_eval_cnt += 1

        # ── ★ v1.2 OBL Phase 0 ───────────────────────────────────────────
        if self.obl and use_batch:
            self._run_obl_phase0()

        self._prev_best = self.gbest_fit

        # ── 主迭代 ────────────────────────────────────────────────────────
        for t in range(self.T):
            alpha = self._get_alpha(t)

            if self.ae_weighting:
                mbest = self._ae_weighted_mbest()
            else:
                mbest = np.mean(self.pbest, axis=0)

            gbest = self.gbest_pos if self.gbest_pos is not None else self.positions[0]

            for i in range(self.M):
                self.positions[i] = self._update_pos_single(
                    self.positions[i], self.pbest[i], gbest, mbest, alpha
                )
                if self.rng.random() < self.mutation_prob:
                    self.positions[i] = self._cauchy_mutation(self.positions[i])
                    self._total_mutations += 1

            if self.pair_interval > 0 and (t + 1) % self.pair_interval == 0:
                self._ae_paired_update(alpha)
                self.logger.info(
                    f"  [AE 配對更新 v1.1] iter={t+1}  累計 {self._total_ae_updates} 次"
                )

            iter_fits = []
            t_iter = time.time()

            if use_batch:
                batch_results = self.batch_evaluate_fn(self.positions)
                elapsed_batch = time.time() - t_iter
                for i, (v, u) in enumerate(batch_results):
                    f = float(v) * float(u)
                    iter_fits.append(f)
                    self._update_pbest(i, f)
                    self._update_gbest(i, f, v, u)
                    self._log_eval(self._global_eval_cnt, t + 1, i, v, u, f,
                                   alpha, elapsed_batch)
                    self._global_eval_cnt += 1
            else:
                for i in range(self.M):
                    t_i = time.time()
                    v, u = self.evaluate_fn(self.positions[i])
                    f    = float(v) * float(u)
                    elapsed = time.time() - t_i
                    iter_fits.append(f)
                    self._update_pbest(i, f)
                    self._update_gbest(i, f, v, u)
                    self._log_eval(self._global_eval_cnt, t + 1, i, v, u, f,
                                   alpha, elapsed)
                    self._global_eval_cnt += 1

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
                'best_v_ever':      self._best_v_ever,
                'best_u_ever':      self._best_u_ever,
                'mean_fitness':     mean_fit,
                'max_fitness':      max_fit,
                'alpha':            alpha,
            })
            self.logger.info(
                f"  [AE-QPSO v1.2 Iter {t+1:3d}/{self.T}] "
                f"α={alpha:.3f}  "
                f"gbest={self.gbest_fit:.4f} (V={self.gbest_val:.3f} U={self.gbest_uniq:.3f})  "
                f"mean={mean_fit:.4f}  max={max_fit:.4f}  "
                f"stag={self._stag_counter}  evals={self._global_eval_cnt}  "
                f"V⋆={self._best_v_ever:.3f}  U⋆={self._best_u_ever:.3f}"
            )

        # ── 最終摘要 ──────────────────────────────────────────────────────
        self.logger.info("=" * 70)
        self.logger.info("AE-SOQPSO 優化完成（v1.2）")
        self.logger.info(f"  Best V×U          : {self.gbest_fit:.6f}")
        self.logger.info(f"  Best V            : {self.gbest_val:.4f}")
        self.logger.info(f"  Best U            : {self.gbest_uniq:.4f}")
        self.logger.info(f"  V_best_ever       : {self._best_v_ever:.4f}")
        self.logger.info(f"  U_best_ever       : {self._best_u_ever:.4f}")
        self.logger.info(
            f"  BO Baseline       : 0.8834  "
            + ("✓ 超越基線!" if self.gbest_fit > 0.8834
               else f"✗ 差距 {0.8834 - self.gbest_fit:.4f}")
        )
        self.logger.info(f"  Total evals       : {self._global_eval_cnt}")
        self.logger.info(f"  Reinits           : {self._total_reinits}")
        self.logger.info(f"  Mutations         : {self._total_mutations}")
        self.logger.info(f"  AE pair updates   : {self._total_ae_updates}")
        self.logger.info(f"  OBL replacements  : {self._total_obl_replaced}")
        self.logger.info("=" * 70)

        best = self.gbest_pos.copy() if self.gbest_pos is not None else np.zeros(self.D)
        return best, self.gbest_fit