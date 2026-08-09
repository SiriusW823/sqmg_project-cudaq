#!/usr/bin/env python3
"""
==============================================================================
node_agent.py — 多節點派工代理（v11.0 新增）
==============================================================================

角色：
    在 SLURM 配額內，由父行程（run_qpso_qmg_cudaq.py --dispatch slurm）以

        srun -N <nodes> --ntasks-per-node=1 --gres=gpu:<n_local> \
             python node_agent.py --job_dir <shared> --round <R> ...

    每個節點啟動「一個」node_agent。agent 依 SLURM_NODEID 認領自己那一段
    粒子，再用既有的「每 GPU 一個 worker_eval.py 子行程」模式在本地並行。

    亦即維持 v10.x 已驗證的單節點並行架構不變，只在其上加一層節點維度：

        父行程（1）→ node_agent（N 個節點）→ worker_eval（每節點 8 個 GPU）

    總並行度 G = nodes × gpus_per_node，取代原本固定的 8。

為何用 srun 而非 SSH：
    叢集的 node-to-node SSH 被封鎖（permission denied），只能經跳板機；
    但 SLURM 的 srun 在配額內可直接跨節點啟動 step（gpu_scaling_bench.slurm
    已實測可行）。因此多節點派工一律走 srun。

檔案交換（重要）：
    /tmp 是「節點本地」的，父行程讀不到其他節點寫的 /tmp。
    因此所有 weight / result 檔一律放在共享檔案系統（beegfs 家目錄）下的
    --job_dir。父行程負責寫入 weight 與清理，agent 只讀 weight、寫 result。

契約（父行程 ↔ agent）：
    {job_dir}/round_{R}/manifest.json   父行程寫：{"slots": [particle_idx, ...]}
                                        slots 依「全域 slot 序」排列，
                                        node rank r 認領 slots[r*n_local:(r+1)*n_local]
    {job_dir}/round_{R}/w_{pidx}.npy    父行程寫：已套用 chemistry constraint 的權重
    {job_dir}/round_{R}/r_{pidx}.npy    agent 寫（實際由 worker_eval.py 寫）：
                                        [validity, uniqueness, HBA, HBD]
    {job_dir}/round_{R}/done_{rank}.json agent 寫：本節點各粒子的狀態，供父行程診斷

放置位置：node_agent.py（專案根目錄，與 worker_eval.py 同層）
==============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def _node_rank() -> int:
    """
    取得本 agent 在這一個 srun step 中的節點序號。

    SLURM_NODEID 是「節點」序號；SLURM_PROCID 是「task」序號。
    我們固定 --ntasks-per-node=1，兩者等價，但 NODEID 語意較準確，
    故優先使用，並保留 PROCID 作為後備（某些 SLURM 版本未匯出 NODEID）。
    """
    for var in ("SLURM_NODEID", "SLURM_PROCID"):
        val = os.environ.get(var)
        if val is not None and val.strip().isdigit():
            return int(val.strip())
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="QMG multi-node dispatch agent (v11.0)")
    p.add_argument("--job_dir",        type=str, required=True,
                   help="共享檔案系統上的 job 目錄（必須所有節點皆可讀寫）")
    p.add_argument("--round",          type=int, required=True)
    p.add_argument("--n_local",        type=int, default=8,
                   help="本節點使用的 GPU 數（= 每節點可同時跑的 worker 數）")
    p.add_argument("--repo",           type=str, default=None,
                   help="專案根目錄；預設為本檔案所在目錄")
    p.add_argument("--num_heavy_atom", type=int, default=9)
    p.add_argument("--num_sample",     type=int, default=5000)
    p.add_argument("--backend",        type=str, default="cudaq_nvidia")
    p.add_argument("--timeout",        type=int, default=360,
                   help="單一 worker_eval 子行程的最大秒數")
    p.add_argument("--report_hbahbd",  action="store_true", default=False)
    args = p.parse_args()

    rank      = _node_rank()
    hostname  = os.uname().nodename
    repo      = args.repo or os.path.dirname(os.path.abspath(__file__))
    worker    = os.path.join(repo, "worker_eval.py")
    round_dir = os.path.join(args.job_dir, f"round_{args.round}")

    def log(msg: str) -> None:
        # stdout 由 srun 收集回父行程；加上 rank/host 方便追蹤是哪個節點。
        print(f"[agent r{rank} {hostname}] {msg}", flush=True)

    if not os.path.exists(worker):
        log(f"FATAL worker_eval.py 不存在：{worker}")
        sys.exit(2)

    manifest_path = os.path.join(round_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        log(f"FATAL manifest 不存在：{manifest_path}")
        sys.exit(2)

    with open(manifest_path, "r") as f:
        slots = json.load(f)["slots"]

    # ── 認領本節點的粒子區段 ──────────────────────────────────────────────
    #   slots 已依全域 slot 序排好，node rank r 取第 r 段。
    #   最後一輪可能不滿（M 不整除 G），該段可能為空 → 正常結束即可。
    lo   = rank * args.n_local
    hi   = min(lo + args.n_local, len(slots))
    mine = slots[lo:hi] if lo < len(slots) else []

    log(f"round={args.round} 認領 {len(mine)} 個粒子：{mine}")

    if not mine:
        # 沒有工作也要寫 done 檔，父行程才知道本節點確實啟動過（而非 srun 沒排到）。
        with open(os.path.join(round_dir, f"done_{rank}.json"), "w") as f:
            json.dump({"rank": rank, "host": hostname, "particles": [],
                       "status": {}}, f)
        return

    # ── 每 GPU 一個 worker 子行程（沿用 v10.x 已驗證模式）────────────────
    t0     = time.time()
    procs  = []
    for local_gpu, pidx in enumerate(mine):
        wpath = os.path.join(round_dir, f"w_{pidx}.npy")
        rpath = os.path.join(round_dir, f"r_{pidx}.npy")

        env = os.environ.copy()
        # 關鍵：在子行程 import cudaq（CUDA 初始化）之前綁定單一 GPU。
        env["CUDA_VISIBLE_DEVICES"] = str(local_gpu)
        env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
        # srun step 已用 --gres 綁定本節點 GPU，這裡的 index 是「step 內的相對 index」。

        cmd = [
            sys.executable, worker,
            "--weight_path",    wpath,
            "--result_path",    rpath,
            "--num_heavy_atom", str(args.num_heavy_atom),
            "--num_sample",     str(args.num_sample),
            "--backend",        args.backend,
        ]
        if args.report_hbahbd:
            cmd.append("--report_hbahbd")

        proc = subprocess.Popen(
            cmd, env=env, cwd=repo,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        procs.append((proc, pidx, local_gpu))

    # ── 收割 ─────────────────────────────────────────────────────────────
    status: dict = {}
    for proc, pidx, local_gpu in procs:
        try:
            _, stderr_bytes = proc.communicate(timeout=args.timeout)
            if proc.returncode == 0:
                status[str(pidx)] = "ok"
            else:
                msg = stderr_bytes.decode("utf-8", errors="replace")[-300:]
                status[str(pidx)] = f"exit={proc.returncode}"
                log(f"粒子 {pidx} (gpu {local_gpu}) 失敗 exit={proc.returncode}: {msg}")
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            status[str(pidx)] = "timeout"
            log(f"粒子 {pidx} (gpu {local_gpu}) 逾時 >{args.timeout}s")
        except Exception as e:                       # noqa: BLE001
            status[str(pidx)] = f"exception:{e}"
            log(f"粒子 {pidx} (gpu {local_gpu}) 例外：{e}")

    elapsed = time.time() - t0
    with open(os.path.join(round_dir, f"done_{rank}.json"), "w") as f:
        json.dump({"rank": rank, "host": hostname, "particles": mine,
                   "status": status, "elapsed": elapsed}, f)

    n_ok = sum(1 for v in status.values() if v == "ok")
    log(f"round={args.round} 完成 {n_ok}/{len(mine)} 成功  耗時 {elapsed:.1f}s")

    # 注意：即使有粒子失敗也回傳 0。單一粒子失敗不應讓整個 srun step 被判定失敗，
    # 父行程會依 result 檔內容（worker_eval 預設寫 [0,0,0,0]）自行判定有效性，
    # 與單節點模式的容錯語意一致。
    sys.exit(0)


if __name__ == "__main__":
    main()
