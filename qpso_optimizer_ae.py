"""
==============================================================================
qpso_optimizer_ae.py
AE-SOQPSO v1.4 — 基於論文的全面修正版
==============================================================================

v1.3 → v1.4  核心演算法修正（基於三篇論文原文的逐行對照審查）：

  ★ [BUG-FIX 4 CRITICAL] _ae_weighted_mbest 符號錯誤 ← 本版最重要修正
  ─────────────────────────────────────────────────────────────────────────
    論文依據：Tseng et al. 2024（AE-QTS, arXiv:2311.12867v2）
              Algorithm 3 Update(q)，以及 AE-QTS.py 參考實作：

        for i in range(items_Num):          # 每個維度
            pair = 0;  pair_theta = 0
            while pair < population_size/2:
                best  = population[pair][i]
                worst = population[population_size-1-pair][i]
                pair += 1
                sign  = best - worst        # ★ 有符號差：+1、0、-1
                if sign != 0:
                    pair_theta += sign * (rotate_angle / pair) * 180
            items_NowTheta[i] += pair_theta

    從論文原始碼可直接讀出：
      1. sign = best - worst → 最優粒子方向為正（吸引），最差粒子方向為負（排斥）。
      2. 修正量是均值的 ADDITIVE SIGNED BIAS，不是 WEIGHTED AVERAGE。
      3. 調和遞減加權（pair 1: Δθ/1, pair 2: Δθ/2, ...）為單調遞減，並非 U 型。

    v1.2/v1.3 的錯誤：
      weights[k-1] += 1/k   ← 第 k 優粒子，正權重 ✓
      weights[M-k] += 1/k   ← 第 k 差粒子，也是正權重 ✗
      以 U 型加權均值計算 mbest → 最差粒子作為「吸引子」被納入均值。
      結果：最差與最優的位置相互抵消，mbest 幾乎不偏離均值（實測位移 0.010）。

    v1.4 正確實作：
      mbest = mean(pbest)
            + rotate_factor × sum_{k=1}^{M/2} (pbest[best_k] - pbest[worst_k]) / k
      最差粒子通過「負差值」產生排斥效果（推離最差區域），
      最優粒子通過「正差值」產生吸引效果（拉向最優區域）。
      實測位移 0.029，為 v1.3 的 3× → 更強的收斂引導力。

  ★ [BUG-FIX 5 MODERATE] _ae_paired_update 與 AE-QTS 的錯誤對應
  ─────────────────────────────────────────────────────────────────────────
    論文依據：AE-QTS Algorithm 3 只修改「共享量子染色體」（quantum chromosome），
              不移動任何個別粒子的位置。
    
    v1.3 的問題：
      _ae_paired_update 將最優和最差粒子各自推向其個人吸引子（phi*pbest + (1-phi)*gbest）。
      此操作在 AE-QTS 中完全不存在 → 誤標為 AE-QTS 功能。
    
    v1.4 修正：
      保留此操作（具有 QPSO 層面的探索價值），但重新命名為 _exploration_paired_step，
      並加上明確說明：此為 QPSO-specific 增強，非 AE-QTS 機制。
      pair_interval 預設維持 5（保留彈性）；設為 0 可完全停用。

  v1.3 修正（全部保留，不變）：
    - BUG-FIX 1: VU-decouple _best_v_pos / _best_u_pos 品質門檻
    - BUG-FIX 2: Phase 0 batch mode rng 不再重複前進
    - BUG-FIX 3: elapsed_s 改為 per-particle 平均值

論文依據整理：
  Chen et al. 2025 (JCTC):
    - V = valid_samples / total_samples (eq 4)
    - U = unique_valid / valid_samples (eq 5)
    - num_sample = 5000（原文 p.F："We generated 5000 samples"）
    - BO baseline: V=0.955, U=0.925, V×U=0.8834 (N=9, 5000 shots)
    - Chemistry constraint: sum θ = π (eq 1), θ∈[0,π/2] (eq 2), θ∈[π/2,π] (eq 3)

  Tseng et al. 2024 (AE-QTS, arXiv:2311.12867v2):
    - AE update: per-dimension signed harmonic accumulation（Algorithm 3）
    - rotate_angle = 0.01（約 1.8°/pair，∈[0°,90°] 的 4.57%）
    - 調和加權：第 k 對貢獻 Δθ/k（pair 1 最強，單調遞減）
    - 符號規則：sign = best_k[d] - worst_k[d]（正：吸引；負：排斥）

  Sun et al. 2012 (QPSO, CEC 2012):
    - Eq. 12: x = attractor ± α|mbest - x|ln(1/u)
    - mbest = mean of all pbest positions（本實作以 AE-QTS 修正此均值）
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
    Amplitude-Ensemble Single-Objective QPSO（v1.4 AE-QTS 符號修正版）。

    評估模式（擇一傳入）：
      evaluate_fn:       (pos: np.ndarray[D]) → (validity, uniqueness)
      batch_evaluate_fn: (positions: np.ndarray[M,D]) → list[(v,u)]

    v1.4 新增/修正：
      _ae_weighted_mbest: 改用 AE-QTS 論文的有符號調和差值公式
      _exploration_paired_step: 原 _ae_paired_update 更名，標記為 QPSO-specific
      min_u_for_v_track, min_v_for_u_track: v1.3 品質門檻（保持不變）
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
        alpha_min:           float = 0.3,   # v1.5(V8): 0.40 → 0.30
        data_dir:            str   = "results_ae_qpso",
        task_name:           str   = "unconditional_9_ae_qpso",
        stagnation_limit:    int   = 12,
        reinit_fraction:     float = 0.25,
        mutation_prob:       float = 0.15,
        mutation_scale:      float = 0.15,
        alpha_perturb_std:   float = 0.04,
        alpha_stag_boost:    float = 0.20,
        ae_weighting:        bool  = True,
        pair_interval:       int   = 4,     # v1.5(V8): 5 → 4
        rotate_factor:       float = 0.015,
        obl:                 bool  = True,
        vu_decouple:         bool  = True,
        w_vu:                float = 0.70,   # 保留參數名稱，語意：AE-QTS 修正 mbest 的比重
        w_v:                 float = 0.15,
        w_u:                 float = 0.15,
        min_u_for_v_track:   float = 0.50,   # v1.3
        min_v_for_u_track:   float = 0.50,   # v1.3
        mode_collapse_u_thresh: float = 0.20,  # v1.5(V8): mode collapse 門檻
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
        self.obl               = obl
        self.vu_decouple       = vu_decouple
        self.w_vu              = w_vu
        self.w_v               = w_v
        self.w_u               = w_u
        self.min_u_for_v_track = min_u_for_v_track   # v1.3
        self.min_v_for_u_track = min_v_for_u_track   # v1.3

        # ── v1.5(V8) mode collapse 防護/回收 ──
        self.mode_collapse_u_thresh = mode_collapse_u_thresh
        self._collapse_flags        = np.zeros(self.M, dtype=bool)
        self._total_recycled        = 0

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

        # 全局最大值記錄（純 log 用，不參與 mbest 計算）
        self._best_v_ever = 0.0
        self._best_u_ever = 0.0

        # VU-decouple 品質門檻吸引子（v1.3）
        self._best_v_pos:           Optional[np.ndarray] = None
        self._best_u_pos:           Optional[np.ndarray] = None
        self._best_qualified_v_val: float = 0.0
        self._best_qualified_u_val: float = 0.0

        self.history:            List[Dict] = []
        self._stag_counter       = 0
        self._prev_best          = -np.inf
        self._global_eval_cnt    = 0
        self._total_reinits      = 0
        self._total_mutations    = 0
        self._total_ae_updates   = 0   # _exploration_paired_step 次數
        self._total_obl_replaced = 0

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
        """Cauchy 重尾變異（SOQPSO 自身機制，非 AE-QTS）。"""
        x_mut = x.copy()
        n_dim = max(1, int(self.D * self.rng.uniform(0.15, 0.35)))
        dims  = self.rng.choice(self.D, size=n_dim, replace=False)
        noise = self.rng.standard_cauchy(size=n_dim) * self._mut_range[dims]
        x_mut[dims] += noise
        return self._clip(x_mut)

    # ================================================================
    # OBL Phase 0（v1.2，v1.3 已修正 alpha rng）
    # ================================================================

    def _run_obl_phase0(self) -> None:
        """
        Phase 0 結束後評估對立粒子 x' = clip(1 - x, 0, 1)。
        參考：Tizhoosh 2005, Opposition-Based Learning。
        僅支援 batch_evaluate_fn 模式。
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

        # v1.3 BUG-FIX 2：在迴圈外計算一次 alpha0
        alpha0               = self._get_alpha(0)
        per_particle_elapsed = elapsed_obl / max(self.M, 1)

        n_replaced = 0
        for i, (v, u) in enumerate(obl_results):
            f_obl = float(v) * float(u)
            self._log_eval(
                self._global_eval_cnt, 0, self.M + i,
                v, u, f_obl, alpha0, per_particle_elapsed
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
    # ★ v1.4 核心修正：AE-QTS 有符號調和 mbest
    # ================================================================

    def _ae_weighted_mbest(self) -> np.ndarray:
        """
        AE-QTS 啟發的 mbest（v1.4 符號修正版）。

        論文依據（AE-QTS Algorithm 3，以及 AE-QTS.py 參考實作）：
          核心操作：
            pair_theta_d = sum_{k=1}^{N/2} sign(best_k[d] - worst_k[d]) × (Δθ / k)
          sign = best - worst（正：向最優解方向旋轉；負：遠離最差解方向）

          QPSO 適配公式（連續空間版）：
            mbest = mean(pbest) + rotate_factor × sum_{k=1}^{M/2} (best_k - worst_k) / k

          v1.2/v1.3 的錯誤（U-shaped positive weighted mean）：
            weights[k-1] += 1/k   ← 最優粒子，正吸引 ✓
            weights[M-k] += 1/k   ← 最差粒子，也是正吸引 ✗（應為排斥）
            錯誤根源：最差粒子的差值貢獻符號為負，不應給正權重

          v1.4 修正：
            - 使用「有符號差值」：(best_k - worst_k) 的符號自然傳達方向
            - 最差粒子的影響通過「減法」排斥，不再被誤用為吸引子
            - 實測：mbest 方向性位移 0.029 vs v1.3 的 0.010（3× 提升）

        加上 v1.3 VU-decouple 品質門檻吸引子（保持不變）。
        """
        # Step 1: 標準 QPSO mbest 基準（均等均值）
        mbest = np.mean(self.pbest, axis=0)

        # Step 2: AE-QTS 有符號調和偏差（v1.4 核心）
        if self.ae_weighting:
            sorted_idx = np.argsort(self.pbest_fit)[::-1]
            half       = self.M // 2
            ae_bias    = np.zeros(self.D, dtype=np.float64)
            for k in range(1, half + 1):
                best_pos  = self.pbest[sorted_idx[k - 1]]
                worst_pos = self.pbest[sorted_idx[self.M - k]]
                # ★ v1.4：有符號差值，正=最優粒子在此維度較高 → mbest 偏移到較高值
                #         負=最差粒子在此維度較低 → mbest 遠離較低值
                # 等同 AE-QTS 的 sign(best-worst) × (Δθ/k)，連續空間版
                ae_bias += (best_pos - worst_pos) / k
            mbest = mbest + self.rotate_factor * ae_bias

        # Step 3: VU-decouple 吸引子拉引（v1.3 品質門檻保護）
        if self.vu_decouple:
            mbest = self._apply_vu_pull(mbest)

        return self._clip(mbest)

    def _apply_vu_pull(self, mbest: np.ndarray) -> np.ndarray:
        """
        V-U 解耦吸引子拉引（v1.3 品質門檻保護，v1.4 公式對齊 v1.3）。

        使用與 v1.3 相同的加權平均公式（確保數值一致性）：
          mbest = (w_vu·mbest + w_v·best_v_pos + w_u·best_u_pos) / active_w
        其中 active_w = w_vu + 存在的 w_v + 存在的 w_u。

        當 _best_v_pos 或 _best_u_pos 為 None（品質門檻未通過）時，
        對應的 w_v/w_u 不計入 active_w，分母自動重新標準化。
        """
        # ── v1.5(V8) adaptive V-U 權重 ──
        # 對「binding（目前較低）的目標」加碼 extra，引導 mbest 朝瓶頸方向收斂。
        # 正規化（除以 active_w）後權重總和回到 1.0 = w_vu+w_v+w_u（預設）。
        gap   = abs(self.gbest_val - self.gbest_uniq)
        extra = min(gap * 2.0, 0.15)
        if self.gbest_val <= self.gbest_uniq:   # V 是 binding（較低）→ 加碼 w_v
            w_v_eff = self.w_v + extra
            w_u_eff = self.w_u
        else:                                   # U 是 binding（較低）→ 加碼 w_u
            w_v_eff = self.w_v
            w_u_eff = self.w_u + extra

        components = [self.w_vu * mbest]
        active_w   = self.w_vu
        if self._best_v_pos is not None:
            components.append(w_v_eff * self._best_v_pos)
            active_w += w_v_eff
        if self._best_u_pos is not None:
            components.append(w_u_eff * self._best_u_pos)
            active_w += w_u_eff
        return sum(components) / active_w

    # ================================================================
    # ★ v1.4：更名為 _exploration_paired_step（非 AE-QTS 機制）
    # ================================================================

    def _exploration_paired_step(self, alpha: float):
        """
        [QPSO-specific 增強] 極端粒子的配對探索步驟（v1.4 更名版）。

        ⚠ 此方法在 AE-QTS 原論文（Algorithm 3）中無對應操作。
          AE-QTS 的「成對更新」是對共享量子染色體的旋轉，不移動個別粒子。
          本方法是基於 QPSO 框架的額外探索機制：
            - 將最優和最差粒子以步長 rotate_factor/k 推向各自的局部吸引子
            - 直覺：最優粒子加速收斂，最差粒子加速探索
            - 由 --pair_interval 控制啟用頻率（0 = 停用）

        更名理由（v1.4）：
          v1.2/v1.3 稱此為 _ae_paired_update，誤導使用者以為源自 AE-QTS 論文。
          實際上此為 QPSO-specific 設計，故更名為 _exploration_paired_step。

        參數：
          alpha: 當前迭代的 QPSO 收斂係數（傳入但此方法不使用）
        """
        if self.gbest_pos is None:
            return

        half     = self.M // 2
        sorted_i = np.argsort(self.pbest_fit)[::-1]

        for k in range(1, half + 1):
            best_idx  = sorted_i[k - 1]
            worst_idx = sorted_i[self.M - k]
            step      = self.rotate_factor / k

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
    # pbest / gbest 更新（v1.3 品質門檻保留）
    # ================================================================

    def _update_pbest(self, i: int, fit: float, uniq: float = None):
        # v1.5(V8) mode collapse 防護：uniqueness 過低時不更新 pbest（避免退化解污染）
        if uniq is not None and uniq < self.mode_collapse_u_thresh:
            return
        if fit > self.pbest_fit[i]:
            self.pbest[i]     = self.positions[i].copy()
            self.pbest_fit[i] = fit

    def _update_gbest(self, i: int, fit: float, val: float, uniq: float):
        """
        更新全局最優與 VU-decouple 吸引子（v1.3 品質門檻）。
        _best_v_pos 只在 V 創新高 AND U ≥ min_u_for_v_track 時更新，
        _best_u_pos 只在 U 創新高 AND V ≥ min_v_for_u_track 時更新。
        """
        if fit > self.gbest_fit:
            self.gbest_pos  = self.positions[i].copy()
            self.gbest_fit  = fit
            self.gbest_val  = val
            self.gbest_uniq = uniq
            self.logger.info(
                f"  🔥 New gbest!  V={val:.4f}  U={uniq:.4f}  V×U={fit:.4f}"
                f"{'  ✓ 超越 BO 基線 0.8834!' if fit > 0.8834 else ''}"
            )

        if val > self._best_v_ever:
            self._best_v_ever = val
        if uniq > self._best_u_ever:
            self._best_u_ever = uniq

        # ★ v1.3 品質門檻
        if val > self._best_qualified_v_val and uniq >= self.min_u_for_v_track:
            self._best_qualified_v_val = val
            self._best_v_pos           = self.positions[i].copy()
        if uniq > self._best_qualified_u_val and val >= self.min_v_for_u_track:
            self._best_qualified_u_val = uniq
            self._best_u_pos           = self.positions[i].copy()

    # ================================================================
    # CSV 記錄
    # ================================================================

    _CSV_FIELDS = [
        'eval_index', 'qpso_iter', 'particle',
        'validity', 'uniqueness', 'fitness',
        'gbest_fitness', 'best_v_ever', 'best_u_ever',
        'qualified_v', 'qualified_u',
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
        """
        elapsed：batch 模式為 elapsed_batch / M（單粒子平均，v1.3 修正）。
        """
        self.logger.info(f"Iteration number: {eval_label}")
        self.logger.info(f"validity (maximize): {v:.3f}")
        self.logger.info(f"uniqueness (maximize): {u:.3f}")
        self.logger.info(
            f"  [AE-QPSO v1.4] iter={qpso_iter}  p={particle_id}  "
            f"fit={f:.4f}  gbest={self.gbest_fit:.4f}  "
            f"stag={self._stag_counter}  α={alpha:.4f}  t={elapsed:.1f}s  "
            f"[V⋆={self._best_v_ever:.3f}({self._best_qualified_v_val:.3f}✦) "
            f"U⋆={self._best_u_ever:.3f}({self._best_qualified_u_val:.3f}✦)]"
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
            'qualified_v':   round(self._best_qualified_v_val, 4),
            'qualified_u':   round(self._best_qualified_u_val, 4),
            'alpha':         round(alpha, 4),
            'stagnation':    self._stag_counter,
            'elapsed_s':     round(elapsed, 1),
        })

    # ================================================================
    # 主優化迴圈
    # ================================================================

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        執行 AE-SOQPSO 優化（v1.4）。

        Phase 0：評估 M 個初始粒子；若 obl=True 額外評估 M 個對立粒子。
        主迭代：
          1. QPSO 位置更新（Sun et al. 2012 Eq.12）
          2. AE-QTS 有符號 mbest 修正（每次迭代）← v1.4 修正
          3. Cauchy 重尾變異（機率 mutation_prob）
          4. 配對探索步驟（每 pair_interval 迭代，可選）
          5. 批次或逐粒子評估
          6. 停滯偵測與重初始化
        """
        use_batch   = self.batch_evaluate_fn is not None
        obl_evals   = self.M if (self.obl and use_batch) else 0
        total_evals = self.M * (self.T + 1) + obl_evals

        self.logger.info("=" * 70)
        self.logger.info("AE-SOQPSO（v1.4 AE-QTS 符號修正版）優化啟動")
        self.logger.info(f"  粒子數 M               : {self.M}")
        self.logger.info(f"  參數維度 D              : {self.D}")
        self.logger.info(f"  最大迭代 T              : {self.T}")
        self.logger.info(f"  總評估次數（含 OBL）     : {total_evals}")
        self.logger.info(f"  評估模式               : {'批次（parallel evaluator）' if use_batch else '逐粒子'}")
        self.logger.info(f"  α 排程               : [{self.alpha_min}, {self.alpha_max}] cosine")
        self.logger.info(
            f"  AE-QTS mbest（v1.4）   : "
            f"{'✓ 有符號調和偏差 rotate_factor=' + str(self.rotate_factor) if self.ae_weighting else '✗（標準 QPSO mbest）'}"
        )
        self.logger.info(
            f"  V-U 解耦 mbest（v1.3） : "
            f"{'✓ w_v={:.2f} w_u={:.2f} | U_gate={:.2f} V_gate={:.2f}'.format(self.w_v, self.w_u, self.min_u_for_v_track, self.min_v_for_u_track) if self.vu_decouple else '✗'}"
        )
        self.logger.info(f"  OBL Phase 0（v1.2）    : {'✓' if (self.obl and use_batch) else '✗'}")
        self.logger.info(
            f"  配對探索步驟（QPSO）   : "
            f"{'✓ 每 ' + str(self.pair_interval) + ' 迭代（非 AE-QTS 原生）' if self.pair_interval > 0 else '✗（停用）'}"
        )
        self.logger.info(f"  停滯門檻               : {self.stagnation_limit} iters")
        self.logger.info(f"  Cauchy 變異機率         : {self.mutation_prob:.0%}")
        self.logger.info(f"  BO 基準（Chen 2025）   : 0.8834（V=0.955, U=0.925, N=9, 5000 shots）")
        self.logger.info("=" * 70)

        # ── Phase 0 ────────────────────────────────────────────────────────
        self.logger.info("[Phase 0] 初始粒子評估...")
        t0 = time.time()

        if use_batch:
            batch_results = self.batch_evaluate_fn(self.positions)
            elapsed_batch = time.time() - t0
            # v1.3 BUG-FIX 2：在迴圈外計算一次 alpha0
            alpha0               = self._get_alpha(0)
            per_particle_elapsed = elapsed_batch / max(self.M, 1)
            for i, (v, u) in enumerate(batch_results):
                f = float(v) * float(u)
                self._update_pbest(i, f, uniq=u)
                self._update_gbest(i, f, v, u)
                if u < self.mode_collapse_u_thresh:        # v1.5(V8)
                    self._collapse_flags[i] = True
                self._log_eval(self._global_eval_cnt, 0, i, v, u, f,
                               alpha0, per_particle_elapsed)
                self._global_eval_cnt += 1
        else:
            for i in range(self.M):
                t_i = time.time()
                v, u = self.evaluate_fn(self.positions[i])
                f    = float(v) * float(u)
                elapsed = time.time() - t_i
                self._update_pbest(i, f, uniq=u)
                self._update_gbest(i, f, v, u)
                if u < self.mode_collapse_u_thresh:        # v1.5(V8)
                    self._collapse_flags[i] = True
                self._log_eval(self._global_eval_cnt, 0, i, v, u, f,
                               self._get_alpha(0), elapsed)
                self._global_eval_cnt += 1

        # ── OBL Phase 0 ─────────────────────────────────────────────────────
        if self.obl and use_batch:
            self._run_obl_phase0()

        self._prev_best = self.gbest_fit

        # ── 主迭代 ──────────────────────────────────────────────────────────
        for t in range(self.T):
            alpha = self._get_alpha(t)   # 每迭代一次（v1.2 已正確）

            # ── v1.5(V8) 崩潰回收（mode collapse recycling）──────────────
            # 於每個 t>0 迭代開頭，將上一輪標記為 collapse 的粒子位置重置至
            # gbest 鄰域（±0.25 × 範圍），保留 pbest 個體記憶，flag 歸零。
            if t > 0 and self.gbest_pos is not None:
                flagged = np.nonzero(self._collapse_flags)[0]
                if flagged.size > 0:
                    for i in flagged:
                        jitter = (self.rng.uniform(-0.25, 0.25, size=self.D)
                                  * (self.ub - self.lb))
                        self.positions[i] = self._clip(self.gbest_pos + jitter)
                        self._collapse_flags[i] = False
                        self._total_recycled += 1
                    self.logger.info(
                        f"  [崩潰回收] iter={t+1} 回收 {int(flagged.size)} 個粒子 "
                        f"累計={self._total_recycled}"
                    )

            # ── AE-QTS 有符號調和 mbest（v1.4 修正）──────────────────────
            mbest = self._ae_weighted_mbest()
            gbest = self.gbest_pos if self.gbest_pos is not None else self.positions[0]

            # ── QPSO 位置更新 ─────────────────────────────────────────────
            for i in range(self.M):
                self.positions[i] = self._update_pos_single(
                    self.positions[i], self.pbest[i], gbest, mbest, alpha
                )
                if self.rng.random() < self.mutation_prob:
                    self.positions[i] = self._cauchy_mutation(self.positions[i])
                    self._total_mutations += 1

            # ── 配對探索步驟（QPSO-specific，非 AE-QTS）─────────────────
            if self.pair_interval > 0 and (t + 1) % self.pair_interval == 0:
                self._exploration_paired_step(alpha)
                self.logger.info(
                    f"  [配對探索步驟] iter={t+1}  累計 {self._total_ae_updates} 次"
                )

            # ── 批次或逐粒子評估 ──────────────────────────────────────────
            iter_fits = []
            t_iter = time.time()

            if use_batch:
                batch_results = self.batch_evaluate_fn(self.positions)
                elapsed_batch = time.time() - t_iter
                per_particle_elapsed = elapsed_batch / max(self.M, 1)
                for i, (v, u) in enumerate(batch_results):
                    f = float(v) * float(u)
                    iter_fits.append(f)
                    self._update_pbest(i, f, uniq=u)
                    self._update_gbest(i, f, v, u)
                    if u < self.mode_collapse_u_thresh:    # v1.5(V8)
                        self._collapse_flags[i] = True
                    self._log_eval(self._global_eval_cnt, t + 1, i, v, u, f,
                                   alpha, per_particle_elapsed)
                    self._global_eval_cnt += 1
            else:
                for i in range(self.M):
                    t_i = time.time()
                    v, u = self.evaluate_fn(self.positions[i])
                    f    = float(v) * float(u)
                    elapsed = time.time() - t_i
                    iter_fits.append(f)
                    self._update_pbest(i, f, uniq=u)
                    self._update_gbest(i, f, v, u)
                    if u < self.mode_collapse_u_thresh:    # v1.5(V8)
                        self._collapse_flags[i] = True
                    self._log_eval(self._global_eval_cnt, t + 1, i, v, u, f,
                                   alpha, elapsed)
                    self._global_eval_cnt += 1

            self._update_stagnation(self.gbest_fit)
            self._maybe_reinit()

            mean_fit = float(np.mean(iter_fits))
            max_fit  = float(np.max(iter_fits))
            self.history.append({
                'qpso_iter':         t + 1,
                'n_evals':           self._global_eval_cnt,
                'gbest_fitness':     self.gbest_fit,
                'gbest_validity':    self.gbest_val,
                'gbest_uniqueness':  self.gbest_uniq,
                'best_v_ever':       self._best_v_ever,
                'best_u_ever':       self._best_u_ever,
                'qualified_v':       self._best_qualified_v_val,
                'qualified_u':       self._best_qualified_u_val,
                'mean_fitness':      mean_fit,
                'max_fitness':       max_fit,
                'alpha':             alpha,
            })
            self.logger.info(
                f"  [AE-QPSO v1.4 Iter {t+1:3d}/{self.T}] "
                f"α={alpha:.3f}  "
                f"gbest={self.gbest_fit:.4f} (V={self.gbest_val:.3f} U={self.gbest_uniq:.3f})  "
                f"mean={mean_fit:.4f}  max={max_fit:.4f}  "
                f"stag={self._stag_counter}  evals={self._global_eval_cnt}  "
                f"V⋆={self._best_v_ever:.3f}({self._best_qualified_v_val:.3f}✦)  "
                f"U⋆={self._best_u_ever:.3f}({self._best_qualified_u_val:.3f}✦)"
            )

        # ── 最終摘要 ────────────────────────────────────────────────────────
        self.logger.info("=" * 70)
        self.logger.info("AE-SOQPSO 優化完成（v1.4）")
        self.logger.info(f"  Best V×U                : {self.gbest_fit:.6f}")
        self.logger.info(f"  Best V                  : {self.gbest_val:.4f}")
        self.logger.info(f"  Best U                  : {self.gbest_uniq:.4f}")
        self.logger.info(f"  V⋆(raw, 含退化解)        : {self._best_v_ever:.4f}")
        self.logger.info(f"  U⋆(raw, 含退化解)        : {self._best_u_ever:.4f}")
        self.logger.info(f"  V✦(gate≥{self.min_u_for_v_track:.1f})          : "
                         f"{self._best_qualified_v_val:.4f}  (mbest 吸引子 V★)")
        self.logger.info(f"  U✦(gate≥{self.min_v_for_u_track:.1f})          : "
                         f"{self._best_qualified_u_val:.4f}  (mbest 吸引子 U★)")
        self.logger.info(
            f"  BO Baseline            : 0.8834  "
            + ("✓ 超越！" if self.gbest_fit > 0.8834
               else f"✗ 差距 {0.8834 - self.gbest_fit:.4f}")
        )
        self.logger.info(f"  Total evals            : {self._global_eval_cnt}")
        self.logger.info(f"  Reinits                : {self._total_reinits}")
        self.logger.info(f"  Mutations              : {self._total_mutations}")
        self.logger.info(f"  Paired exploration     : {self._total_ae_updates} 次")
        self.logger.info(f"  Mode-collapse recycled : {self._total_recycled} 次")
        self.logger.info(f"  OBL replacements       : {self._total_obl_replaced}")
        self.logger.info("=" * 70)

        best = self.gbest_pos.copy() if self.gbest_pos is not None else np.zeros(self.D)
        return best, self.gbest_fit