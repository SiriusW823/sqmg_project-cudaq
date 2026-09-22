"""
==============================================================================
evaluator.py — 目標函數評估器（單節點 / 多節點），v12.0
==============================================================================

把「一批參數向量 → 一批 (validity, uniqueness)」這件事抽成獨立模組，
讓所有最適化器共用同一套評估管線。這是 v12.0 重構的核心：
在此之前這段邏輯埋在 run_qpso_qmg_cudaq.py 裡，只有 RR-QPSO 用得到。

兩種派工模式
------------
`local` —— 單節點：每 GPU 一個 worker_eval.py 子行程，輪流吃完整批。
`slurm` —— 多節點：在 sbatch 配額內，每輪一次
           `srun -N n --ntasks-per-node=1 --gres=gpu:8 node_agent.py`，
           並行度 = nodes × gpus_per_node。

多節點模式的兩個已驗證要點（2026-07-25 實測）：
  1. 交換目錄必須在共享檔案系統（beegfs），不可用 /tmp（節點本地）。
  2. SLURM 會偶發 `Job credential expired` 讓整個 step launch 失敗，
     重試即可成功。因此對「結果檔不存在」的粒子會自動重派；
     worker 真的算失敗時會寫出 [0,0,0,0]（檔案存在）→ 不重試，
     這個區分讓重試只修基礎設施問題，不會掩蓋真實的計算失敗。

放置位置：evaluator.py（專案根目錄）
==============================================================================
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Callable, List, Tuple

import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# 單節點：常駐 worker pool（v12.1，預設）
# ===========================================================================

class _WorkerPool:
    """
    每 GPU 一個常駐 `persistent_worker.py`，透過 stdin/stdout 派任務。

    相對於「每次評估開一個新行程」，省下每次 ~24s 的
    Python + cudaq import + GPU target 驗證 + generator 建構成本
    （實測 33.8 s/eval → 約 10 s/eval）。

    worker 若死掉會自動重啟一次；再失敗則該粒子以 0 計入，
    與既有的容錯語意一致。
    """

    def __init__(self, gpu_ids, logger, num_heavy_atom, num_sample,
                 backend, report_hbahbd, timeout, smiles_log_dir=None,
                 recycle_every=10):
        self.gpu_ids   = [str(g) for g in gpu_ids]
        self.logger    = logger
        self.nha       = num_heavy_atom
        self.ns        = num_sample
        self.backend   = backend
        self.hbahbd    = report_hbahbd
        self.timeout   = timeout
        self.procs     = [None] * len(self.gpu_ids)
        # ★ 多樣性研究：每個 worker slot 寫自己的檔，避免並行寫入交錯。
        #   聯集在分析時跨檔計算，worker 之間不需協調。
        self.smiles_log_dir = smiles_log_dir
        # ★ worker 回收（見 maybe_recycle）
        self.recycle_every = recycle_every
        self._batches = 0

    def maybe_recycle(self):
        """跑滿 recycle_every 個批次就把所有 worker 重開。

        ★ 為何需要：worker 存活越久越容易卡死
        ------------------------------------
        在 150 個 run 的 log 上量到的失敗率，依 **該 run 自身的年齡** 分桶：

            0h → 0.0%   1h → 1.7%   2h → 17.9%   3h → 22.2%

        對照「當下叢集併發數」分桶只有 6.0% vs 9.9%，幾乎沒有變化——
        所以不是叢集爭用，是**單一 worker 用久了會退化**。
        worker 的 Python 主迴圈本身沒有累積任何狀態，洩漏在
        `gen.sample_molecule` 底下的 CUDA-Q 0.7.1 裡；那是釘死的相依
        （只有 0.7.1 有 sm_70/V100 的 SASS），改不了。

        所以繞過它：在進入退化區之前就重開。批次約 110s，
        recycle_every=10 讓 worker 壽命約 18 分鐘，穩穩落在 0% 區間。
        啟動成本 20–40s 攤提到 10 個批次上，約 3% 的額外開銷。
        """
        self._batches += 1
        if self.recycle_every <= 0 or self._batches % self.recycle_every:
            return
        self.logger.info(
            f"  [pool] 已跑 {self._batches} 個批次，回收全部 worker"
            f"（避免長壽 worker 退化）")
        for i, p in enumerate(self.procs):
            if p is None:
                continue
            try:
                p.stdin.close()
                p.wait(timeout=10)
            except Exception:                              # noqa: BLE001
                try:
                    p.kill()
                except Exception:
                    pass
            self.procs[i] = None
        self.ensure()

    def _spawn(self, i: int):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_ids[i]
        env["PYTHONPATH"] = REPO_DIR + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"

        # ★ worker 必須用「有 cuda-quantum 的直譯器」啟動，而它未必等於目前這個。
        #   Ax/BoTorch 需要 torch + CUDA 13，與 cudaq 0.7.1 的 CUDA 12 衝突，
        #   因此 Ax 跑在獨立的 ax-bo 環境；此時 sys.executable 是 ax-bo 的 python，
        #   而它沒有 cudaq。用 QMG_WORKER_PYTHON 指定實際要用的直譯器。
        worker_py = os.environ.get("QMG_WORKER_PYTHON") or sys.executable
        cmd = [worker_py, os.path.join(REPO_DIR, "persistent_worker.py"),
               "--num_heavy_atom", str(self.nha),
               "--num_sample", str(self.ns),
               "--backend", self.backend]
        if self.hbahbd:
            cmd.append("--report_hbahbd")
        if self.smiles_log_dir:
            os.makedirs(self.smiles_log_dir, exist_ok=True)
            cmd += ["--smiles_log",
                    os.path.join(self.smiles_log_dir, f"slot{i}.smi")]

        p = subprocess.Popen(cmd, env=env, cwd=REPO_DIR,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True, bufsize=1)
        # 等待 ready 訊號（初始化含 cudaq import 與 GPU 驗證，可能要 20-40s）
        t0 = time.time()
        while time.time() - t0 < 300:
            line = p.stdout.readline()
            if not line:
                break
            try:
                if json.loads(line.strip()).get("ready"):
                    self.procs[i] = p
                    return True
            except json.JSONDecodeError:
                continue          # 忽略 generator 的初始化訊息
        try:
            p.kill()
        except Exception:
            pass
        return False

    def ensure(self):
        for i in range(len(self.gpu_ids)):
            if self.procs[i] is None or self.procs[i].poll() is not None:
                if not self._spawn(i):
                    self.logger.warning(f"[pool] GPU {self.gpu_ids[i]} worker 啟動失敗")

    def run_chunk(self, tasks):
        """tasks: [(slot_i, wpath, rpath, seed)]，同時最多 len(gpu_ids) 個。"""
        sent = []
        for slot, wpath, rpath, seed in tasks:
            p = self.procs[slot]
            if p is None or p.poll() is not None:
                continue
            try:
                p.stdin.write(json.dumps({"w": wpath, "r": rpath, "seed": seed}) + "\n")
                p.stdin.flush()
                sent.append(slot)
            except Exception as e:                        # noqa: BLE001
                self.logger.warning(f"[pool] 送任務失敗（slot {slot}）：{e}")

        for slot in sent:
            p = self.procs[slot]
            try:
                line = p.stdout.readline()
                if not line:
                    self.logger.warning(f"[pool] slot {slot} worker 無回應，將重啟")
                    try:
                        p.kill()
                    except Exception:
                        pass
                    self.procs[slot] = None
                    continue
                r = json.loads(line.strip())
                if not r.get("ok"):
                    self.logger.warning(f"[pool] slot {slot} 評估失敗：{r.get('err','')[:200]}")
            except Exception as e:                        # noqa: BLE001
                self.logger.warning(f"[pool] slot {slot} 讀取失敗：{e}")
                self.procs[slot] = None

    def close(self):
        for p in self.procs:
            if p is not None:
                try:
                    p.stdin.close(); p.wait(timeout=10)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass


def make_pooled_evaluator(
    cwg,
    logger:         logging.Logger,
    gpu_ids:        List[str],
    num_heavy_atom: int = 9,
    num_sample:     int = 5000,
    backend:        str = "cudaq_nvidia",
    timeout:        int = 900,
    report_hbahbd:  bool = False,
    shot_seed:      int = 0,
    smiles_log_dir: str = None,
    retries:        int = 2,
    recycle_every:  int = 10,
) -> Callable[[np.ndarray], List[Tuple[float, float]]]:
    """
    常駐 worker pool 版的單節點評估器（v12.1 預設）。

    `shot_seed` 傳給 `sample_molecule(random_seed=...)`：
      - 0（預設）＝ 與歷史資料一致的確定性目標函數
      - 其他值 ＝ 不同的 shot 序列（用於量測取樣不確定性）
    """
    pool = _WorkerPool(gpu_ids, logger, num_heavy_atom, num_sample,
                       backend, report_hbahbd, timeout, smiles_log_dir,
                       recycle_every)
    if smiles_log_dir:
        logger.info(f"  [pool] 相異 SMILES 記錄已開啟 → {smiles_log_dir}")
    n = len(gpu_ids)
    tmpdir = tempfile.gettempdir()

    def batch_evaluate(positions: np.ndarray) -> List[Tuple[float, ...]]:
        M = positions.shape[0]
        # (V, U, HBA, HBD)；缺件時的預設值，供容錯路徑使用
        out: List[Tuple[float, ...]] = [(0.0, 0.0, 0.0, 0.0)] * M
        t0 = time.time()
        lost = short = retried = 0
        first_err = None
        pool.maybe_recycle()      # 長壽 worker 會退化，先看要不要重開
        pool.ensure()

        for start in range(0, M, n):
            chunk = list(range(start, min(start + n, M)))
            tasks, paths = [], []
            for slot, pidx in enumerate(chunk):
                uid   = uuid.uuid4().hex[:8]
                wpath = os.path.join(tmpdir, f"qmg_w_{uid}.npy")
                rpath = os.path.join(tmpdir, f"qmg_r_{uid}.npy")
                np.save(wpath, cwg.apply_chemistry_constraint(positions[pidx].copy()))
                tasks.append((slot, wpath, rpath, shot_seed))
                paths.append((pidx, wpath, rpath))

            pool.run_chunk(tasks)

            # ── 收結果；結果檔不存在 = 基礎設施失敗，必須重試 ──────────
            # ★ 為何一定要重試（這段是實測代價換來的）
            #   worker 偶爾會卡住不回應，pool 會重啟它，但送進去的那些粒子
            #   永遠不會寫出結果檔，於是被以 0 計入。實測 150 個 run 裡
            #   **146 個（97%）** 出現過這種批次，4,800 個批次中有 460 個
            #   （9.6%）受影響，共 3,846 個粒子評估被誤記為 0。
            #
            #   後果不只是「少算幾個粒子」：0 是一個**錯誤的訊息**，它告訴
            #   最適化器那塊參數空間毫無價值，而它其實從未被評估。更糟的是
            #   它讓整條搜尋軌跡岔開——同一個 seed、同一組組態跑兩次會得到
            #   不同結果，實測全距 0.015~0.027，比本專案追的大多數效應都大。
            #   配對設計假設 seed 固定住 arm 以外的一切，那個假設因此不成立。
            #
            #   模組開頭早就寫明這個區分：檔案不存在 → 基礎設施問題 → 重試；
            #   worker 自己算出 [0,0,0,0]（檔案存在）→ 真實的計算失敗 → 不重試。
            #   pool 路徑先前漏了前半段，這裡補上。
            pending = list(paths)
            for attempt in range(1 + retries):
                missing = []
                for pidx, wpath, rpath in pending:
                    try:
                        arr = np.load(rpath)
                        # worker 寫出的是 [V, U, HBA, HBD]，全部帶回。
                        out[pidx] = tuple(float(x) for x in arr[:4])
                        if len(arr) < 4:
                            short += 1
                    except Exception as e:                  # noqa: BLE001
                        missing.append((pidx, wpath, rpath))
                        if first_err is None:
                            first_err = f"{type(e).__name__}: {str(e)[:120]}"
                if not missing or attempt == retries:
                    pending = missing
                    break
                logger.warning(
                    f"  [pool] {len(missing)} 個粒子的結果檔不存在（基礎設施失敗），"
                    f"重試 {attempt + 1}/{retries}")
                pool.ensure()                     # 先把死掉的 worker 拉回來
                retried += len(missing)
                pool.run_chunk([(s, w, r, shot_seed)
                                for s, (_, w, r) in enumerate(missing)])
                pending = missing

            lost += len(pending)                  # 重試後仍失敗才算真的丟失
            for pidx, wpath, rpath in paths:
                for q in (wpath, rpath):
                    try:
                        os.remove(q)
                    except FileNotFoundError:
                        pass

        # 用索引而非解包：結果現在是 (V, U, HBA, HBD) 四元組
        valid = sum(1 for m in out if m[0] > 0)
        dt = time.time() - t0
        logger.info(f"  [pool] 批次 {M} 個粒子（{n} workers）有效 {valid}/{M}  "
                    f"耗時 {dt:.1f}s  ({dt/max(M,1):.1f} s/eval)"
                    + (f"  重試 {retried}" if retried else ""))
        if lost:
            # 重試過仍拿不到 → 這批次不再是「同一組態必然重現」的，明確標記。
            logger.warning(
                f"  [pool] ★ {lost}/{M} 個粒子重試 {retries} 次後仍無結果檔，"
                f"以 0 計入 —— 本次執行已不具決定性。首個錯誤：{first_err}")
        if short:
            logger.warning(f"  [pool] {short}/{M} 個結果少於 4 個欄位；"
                           f"hbahbd 目標會因此失效（需 V,U,HBA,HBD）。")
        return out

    return batch_evaluate


# ===========================================================================
# 單節點：每 GPU 一個子行程（v12.0 舊版，保留作為對照與後備）
# ===========================================================================

def make_local_evaluator(
    cwg,
    logger:        logging.Logger,
    gpu_ids:       List[str],
    num_heavy_atom: int = 9,
    num_sample:    int = 5000,
    backend:       str = "cudaq_nvidia",
    timeout:       int = 600,
) -> Callable[[np.ndarray], List[Tuple[float, float]]]:
    """單節點批次評估器。並行度 = len(gpu_ids)。"""
    worker = os.path.join(REPO_DIR, "worker_eval.py")
    n_gpus = len(gpu_ids)

    def batch_evaluate(positions: np.ndarray) -> List[Tuple[float, float]]:
        M = positions.shape[0]
        out: List[Tuple[float, ...]] = [(0.0, 0.0, 0.0, 0.0)] * M   # V,U,HBA,HBD
        t0 = time.time()

        for start in range(0, M, n_gpus):
            chunk = list(range(start, min(start + n_gpus, M)))
            procs, wpaths = [], []

            for local_i, pidx in enumerate(chunk):
                uid   = uuid.uuid4().hex[:8]
                wpath = os.path.join(tempfile.gettempdir(), f"qmg_w_{uid}.npy")
                rpath = os.path.join(tempfile.gettempdir(), f"qmg_r_{uid}.npy")
                np.save(wpath, cwg.apply_chemistry_constraint(positions[pidx].copy()))

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[local_i % n_gpus])
                env["PYTHONPATH"] = REPO_DIR + os.pathsep + env.get("PYTHONPATH", "")

                procs.append((subprocess.Popen(
                    [os.environ.get("QMG_WORKER_PYTHON") or sys.executable, worker,
                     "--weight_path", wpath, "--result_path", rpath,
                     "--num_heavy_atom", str(num_heavy_atom),
                     "--num_sample", str(num_sample),
                     "--backend", backend],
                    env=env, cwd=REPO_DIR,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                ), rpath, pidx))
                wpaths.append(wpath)

            for proc, rpath, pidx in procs:
                try:
                    _, err = proc.communicate(timeout=timeout)
                    if proc.returncode == 0:
                        arr = np.load(rpath)
                        out[pidx] = tuple(float(x) for x in arr[:4])   # [V,U,HBA,HBD]
                    else:
                        logger.warning(
                            f"[local] 粒子 {pidx} exit={proc.returncode}: "
                            f"{err.decode('utf-8', errors='replace')[-300:]}")
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                    logger.warning(f"[local] 粒子 {pidx} 逾時 >{timeout}s")
                except Exception as e:                       # noqa: BLE001
                    logger.warning(f"[local] 粒子 {pidx} 例外：{e}")
                finally:
                    for p in (rpath,):
                        try:
                            os.remove(p)
                        except FileNotFoundError:
                            pass
            for w in wpaths:
                try:
                    os.remove(w)
                except FileNotFoundError:
                    pass

        logger.info(f"  [local] 批次 {M} 個粒子完成，耗時 {time.time()-t0:.1f}s")
        return out

    return batch_evaluate


# ===========================================================================
# 多節點：SLURM srun 扇出
# ===========================================================================

def make_slurm_evaluator(
    cwg,
    logger:         logging.Logger,
    job_dir:        str,
    nodes:          int,
    gpus_per_node:  int = 8,
    num_heavy_atom: int = 9,
    num_sample:     int = 5000,
    backend:        str = "cudaq_nvidia",
    timeout:        int = 600,
    srun_overhead:  int = 180,
    srun_retries:   int = 3,
    retry_wait:     int = 20,
    srun_extra:     str = "",
) -> Callable[[np.ndarray], List[Tuple[float, float]]]:
    """多節點批次評估器。並行度 = nodes × gpus_per_node。"""
    agent = os.path.join(REPO_DIR, "node_agent.py")
    G     = nodes * gpus_per_node
    extra = shlex.split(srun_extra) if srun_extra else []
    step_timeout = timeout + srun_overhead

    real = os.path.realpath(job_dir)
    if real.startswith(("/tmp", "/var/tmp", "/dev/shm")):
        raise ValueError(
            f"job_dir 位於節點本地路徑 {real}。多節點模式必須用共享檔案系統"
            f"（本叢集為 beegfs 家目錄），否則父行程讀不到其他節點的結果。")
    os.makedirs(job_dir, exist_ok=True)

    counter = [0]

    def _srun(round_id: int, pending: List[int]) -> None:
        rdir = os.path.join(job_dir, f"round_{round_id}")
        with open(os.path.join(rdir, "manifest.json"), "w") as f:
            json.dump({"slots": list(pending)}, f)

        n_need = min(nodes, (len(pending) + gpus_per_node - 1) // gpus_per_node)
        cmd = ["srun", f"--nodes={n_need}", "--ntasks-per-node=1",
               f"--gres=gpu:{gpus_per_node}", "--kill-on-bad-exit=0", *extra,
               sys.executable, agent,
               "--job_dir", job_dir, "--round", str(round_id),
               "--n_local", str(gpus_per_node), "--repo", REPO_DIR,
               "--num_heavy_atom", str(num_heavy_atom),
               "--num_sample", str(num_sample),
               "--backend", backend, "--timeout", str(timeout)]
        try:
            p = subprocess.run(cmd, cwd=REPO_DIR, timeout=step_timeout,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode != 0:
                logger.warning(
                    f"[slurm] srun 回傳 {p.returncode}\n  "
                    f"{p.stderr.decode('utf-8', errors='replace')[-500:]}")
            for line in p.stdout.decode("utf-8", errors="replace").splitlines():
                if line.strip():
                    logger.info(f"  {line}")
        except subprocess.TimeoutExpired:
            logger.warning(f"[slurm] srun step 逾時 >{step_timeout}s")
        except FileNotFoundError:
            logger.error("[slurm] 找不到 srun。必須在 SLURM 配額內執行。")
            raise

    def batch_evaluate(positions: np.ndarray) -> List[Tuple[float, float]]:
        M = positions.shape[0]
        out: List[Tuple[float, ...]] = [(0.0, 0.0, 0.0, 0.0)] * M   # V,U,HBA,HBD
        t0 = time.time()

        for start in range(0, M, G):
            ids  = list(range(start, min(start + G, M)))
            rid  = counter[0]; counter[0] += 1
            rdir = os.path.join(job_dir, f"round_{rid}")
            os.makedirs(rdir, exist_ok=True)

            for pidx in ids:
                np.save(os.path.join(rdir, f"w_{pidx}.npy"),
                        cwg.apply_chemistry_constraint(positions[pidx].copy()))

            def missing(xs):
                return [p for p in xs
                        if not os.path.exists(os.path.join(rdir, f"r_{p}.npy"))]

            pending = list(ids)
            for attempt in range(srun_retries + 1):
                _srun(rid, pending)
                pending = missing(pending)
                if not pending:
                    break
                if attempt < srun_retries:
                    wait = retry_wait * (attempt + 1)
                    logger.warning(
                        f"[slurm] 仍有 {len(pending)} 個粒子無結果檔"
                        f"（多為 step launch 失敗），{wait}s 後重試 "
                        f"({attempt+1}/{srun_retries})...")
                    time.sleep(wait)
                else:
                    logger.error(
                        f"[slurm] 重試 {srun_retries} 次後仍有 {len(pending)} 個"
                        f"粒子無結果，以 0 計入。")

            for pidx in ids:
                try:
                    arr = np.load(os.path.join(rdir, f"r_{pidx}.npy"))
                    out[pidx] = tuple(float(x) for x in arr[:4])       # [V,U,HBA,HBD]
                except Exception as e:                       # noqa: BLE001
                    logger.warning(f"[slurm] 粒子 {pidx} 結果讀取失敗：{e}")

            try:
                shutil.rmtree(rdir)
            except OSError:
                pass

        # 用索引而非解包：結果現在是 (V, U, HBA, HBD) 四元組
        valid = sum(1 for m in out if m[0] > 0)
        logger.info(
            f"  [slurm] 批次 {M} 個粒子（{nodes}×{gpus_per_node}={G} 並行）"
            f"有效 {valid}/{M}  耗時 {time.time()-t0:.1f}s")
        return out

    return batch_evaluate
