"""
==============================================================================
run_qpso_qmg_cudaq_hbahbd_multiobj.py

Opt-in HBA/HBD multi-objective runner for the CUDA-Q AE-SOQPSO pipeline.

This file intentionally leaves the normal v10.4 runner unchanged. It reuses the
same worker_eval.py, chemistry constraint, subprocess GPU isolation, Sobol
initialization, and AESOQPSOOptimizer. The only opt-in difference is the scalar
fitness:

  objective = (V * U) * ((1 - chem_weight) + chem_weight * chem_closeness)

where chem_closeness is a Gaussian closeness score around HBA=4 and HBD=3.
The optimizer still tracks true V and U from the first two evaluator fields.
==============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

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

try:
    from run_qpso_qmg_cudaq import (
        log_gpu_info,
        log_memory,
        make_sobol_positions,
        setup_logger,
    )
except ImportError as e:
    print(f"[ERROR] 無法 import run_qpso_qmg_cudaq helper: {e}")
    sys.exit(1)


@dataclass
class ObjectiveComponents:
    validity: float
    uniqueness: float
    vu: float
    hba: float
    hbd: float
    hba_error: float
    hbd_error: float
    chem_closeness: float
    objective: float


class HBAHBDMultiObjective:
    def __init__(
        self,
        hba_target: float,
        hbd_target: float,
        hba_sigma: float,
        hbd_sigma: float,
        chem_weight: float,
    ):
        if not (0.0 <= chem_weight <= 1.0):
            raise ValueError("--chem_weight must be in [0, 1]")
        if hba_sigma <= 0 or hbd_sigma <= 0:
            raise ValueError("--hba_sigma and --hbd_sigma must be > 0")
        self.hba_target = hba_target
        self.hbd_target = hbd_target
        self.hba_sigma = hba_sigma
        self.hbd_sigma = hbd_sigma
        self.chem_weight = chem_weight

    def components(self, metrics: Tuple[float, ...]) -> ObjectiveComponents:
        v = float(metrics[0]) if len(metrics) > 0 else 0.0
        u = float(metrics[1]) if len(metrics) > 1 else 0.0
        hba = float(metrics[2]) if len(metrics) > 2 else 0.0
        hbd = float(metrics[3]) if len(metrics) > 3 else 0.0
        vu = v * u
        hba_error = abs(hba - self.hba_target)
        hbd_error = abs(hbd - self.hbd_target)
        exponent = -0.5 * (
            (hba_error / self.hba_sigma) ** 2
            + (hbd_error / self.hbd_sigma) ** 2
        )
        chem_closeness = math.exp(exponent)
        objective = vu * (
            (1.0 - self.chem_weight)
            + self.chem_weight * chem_closeness
        )
        return ObjectiveComponents(
            validity=v,
            uniqueness=u,
            vu=vu,
            hba=hba,
            hbd=hbd,
            hba_error=hba_error,
            hbd_error=hbd_error,
            chem_closeness=chem_closeness,
            objective=objective,
        )

    def score(self, metrics: Tuple[float, ...]) -> float:
        return self.components(metrics).objective

    def describe(self) -> str:
        return (
            "objective=(V*U)*((1-chem_weight)+chem_weight*chem_closeness), "
            f"chem_weight={self.chem_weight:g}, "
            f"chem_closeness=exp(-0.5*((|HBA-{self.hba_target:g}|/{self.hba_sigma:g})^2"
            f"+(|HBD-{self.hbd_target:g}|/{self.hbd_sigma:g})^2))"
        )


class MultiObjectiveRecorder:
    def __init__(self, args: argparse.Namespace, scorer: HBAHBDMultiObjective, logger):
        self.args = args
        self.scorer = scorer
        self.logger = logger
        self.call_idx = 0
        self.best: Dict[str, float] | None = None
        self.csv_path = os.path.join(args.data_dir, f"{args.task_name}_multiobj.csv")
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write(
                "iter_label,phase,best_particle,"
                "objective,validity,uniqueness,product_validity_uniqueness,"
                "HBA,HBD,HBA_error,HBD_error,chem_closeness,"
                "batch_best_vu,batch_best_vu_HBA,batch_best_vu_HBD\n"
            )

    def _phase_label(self) -> Tuple[str, int]:
        idx = self.call_idx
        self.call_idx += 1
        if idx == 0:
            return "phase0", 0
        if self.args.obl and idx == 1:
            return "obl", -1
        return "iter", idx - (1 if self.args.obl else 0)

    def report_batch(self, results: List[Tuple[float, ...]]) -> None:
        phase, iter_label = self._phase_label()
        if not results:
            return

        components = [self.scorer.components(tuple(r)) for r in results]
        best_i = int(np.argmax([c.objective for c in components]))
        best = components[best_i]
        best_vu_i = int(np.argmax([c.vu for c in components]))
        best_vu = components[best_vu_i]

        row = {
            "iter_label": iter_label,
            "phase": phase,
            "best_particle": best_i,
            "objective": best.objective,
            "validity": best.validity,
            "uniqueness": best.uniqueness,
            "product_validity_uniqueness": best.vu,
            "HBA": best.hba,
            "HBD": best.hbd,
            "HBA_error": best.hba_error,
            "HBD_error": best.hbd_error,
            "chem_closeness": best.chem_closeness,
            "batch_best_vu": best_vu.vu,
            "batch_best_vu_HBA": best_vu.hba,
            "batch_best_vu_HBD": best_vu.hbd,
        }
        if self.best is None or row["objective"] > self.best["objective"]:
            self.best = row.copy()

        if phase == "obl":
            self.logger.info("[MultiObjective] OBL batch (not counted as an iteration)")
        else:
            self.logger.info(f"[MultiObjective] Iteration number: {iter_label}")
        self.logger.info(f"multi_objective_score (maximize): {best.objective:.6f}")
        self.logger.info(f"product_validity_uniqueness (maximize): {best.vu:.3f}")
        self.logger.info(f"HBA (close to {self.scorer.hba_target:g}): {best.hba:.3f}")
        self.logger.info(f"HBD (close to {self.scorer.hbd_target:g}): {best.hbd:.3f}")
        self.logger.info(
            f"  [objective terms] V={best.validity:.3f} U={best.uniqueness:.3f} "
            f"HBA_err={best.hba_error:.3f} HBD_err={best.hbd_error:.3f} "
            f"chem_closeness={best.chem_closeness:.4f} "
            f"best_particle={best_i}"
        )
        self.logger.info(
            f"  [batch V*U best] V*U={best_vu.vu:.3f} "
            f"HBA={best_vu.hba:.3f} HBD={best_vu.hbd:.3f} "
            f"particle={best_vu_i}"
        )

        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{row['iter_label']},{row['phase']},{row['best_particle']},"
                f"{row['objective']:.8f},{row['validity']:.6f},"
                f"{row['uniqueness']:.6f},{row['product_validity_uniqueness']:.8f},"
                f"{row['HBA']:.6f},{row['HBD']:.6f},"
                f"{row['HBA_error']:.6f},{row['HBD_error']:.6f},"
                f"{row['chem_closeness']:.8f},"
                f"{row['batch_best_vu']:.8f},{row['batch_best_vu_HBA']:.6f},"
                f"{row['batch_best_vu_HBD']:.6f}\n"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CUDA-Q AE-SOQPSO HBA/HBD multi-objective runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num_heavy_atom", type=int, default=9)
    p.add_argument("--num_sample", type=int, default=10000)
    p.add_argument("--particles", type=int, default=128)
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n_gpus", type=int, default=8)
    p.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument(
        "--backend",
        type=str,
        default="cudaq_nvidia",
        choices=["cudaq_qpp", "cudaq_nvidia", "cudaq_nvidia_fp64"],
    )
    p.add_argument("--subprocess_timeout", type=int, default=900)

    p.add_argument("--sobol_init", action="store_true", default=True)
    p.add_argument("--no_sobol_init", action="store_false", dest="sobol_init")

    p.add_argument("--alpha_max", type=float, default=1.2)
    p.add_argument("--alpha_min", type=float, default=0.3)
    p.add_argument("--mutation_prob", type=float, default=0.15)
    p.add_argument("--stagnation_limit", type=int, default=12)
    p.add_argument("--reinit_fraction", type=float, default=0.25)
    p.add_argument("--ae_weighting", action="store_true", default=True)
    p.add_argument("--no_ae_weighting", action="store_false", dest="ae_weighting")
    p.add_argument("--pair_interval", type=int, default=4)
    p.add_argument("--rotate_factor", type=float, default=0.015)
    p.add_argument("--obl", action="store_true", default=True)
    p.add_argument("--no_obl", action="store_false", dest="obl")
    p.add_argument("--vu_decouple", action="store_true", default=True)
    p.add_argument("--no_vu_decouple", action="store_false", dest="vu_decouple")
    p.add_argument("--w_vu", type=float, default=0.70)
    p.add_argument("--w_v", type=float, default=0.15)
    p.add_argument("--w_u", type=float, default=0.15)
    p.add_argument("--min_u_for_v_track", type=float, default=0.50)
    p.add_argument("--min_v_for_u_track", type=float, default=0.50)
    p.add_argument("--mode_collapse_u_thresh", type=float, default=0.20)

    p.add_argument("--hba_target", type=float, default=4.0)
    p.add_argument("--hbd_target", type=float, default=3.0)
    p.add_argument("--hba_sigma", type=float, default=1.0)
    p.add_argument("--hbd_sigma", type=float, default=1.0)
    p.add_argument(
        "--chem_weight",
        type=float,
        default=0.40,
        help="Weight of HBA/HBD closeness in the scalar objective.",
    )

    p.add_argument(
        "--task_name",
        type=str,
        default="chemistry_constraint_cudaq_multiobj_4HBA_3HBD_M128",
    )
    p.add_argument("--data_dir", type=str, default="results_hbahbd_multiobj")
    return p.parse_args()


def verify_workers_hbahbd(
    args: argparse.Namespace,
    cwg: ConditionalWeightsGenerator,
    logger,
    worker_script: str,
    gpu_ids: List[str],
) -> bool:
    logger.info(
        f"[multiobj] worker verification: launching {len(gpu_ids)} subprocesses "
        "with 5 shots and HBA/HBD reporting..."
    )
    pythonpath = os.environ.get("PYTHONPATH", ".")
    w_test = cwg.generate_conditional_random_weights(random_seed=99)
    procs = []
    paths = []
    t0 = time.time()

    for gpu_id in gpu_ids:
        uid = uuid.uuid4().hex[:8]
        wpath = os.path.join(tempfile.gettempdir(), f"qmg_tv_w_{uid}.npy")
        rpath = os.path.join(tempfile.gettempdir(), f"qmg_tv_r_{uid}.npy")
        np.save(wpath, w_test)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = pythonpath
        cmd = [
            sys.executable,
            worker_script,
            "--weight_path", wpath,
            "--result_path", rpath,
            "--num_heavy_atom", str(args.num_heavy_atom),
            "--num_sample", "5",
            "--backend", args.backend,
            "--report_hbahbd",
        ]
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        procs.append((proc, rpath, str(gpu_id)))
        paths.append(wpath)

    all_ok = True
    for proc, rpath, gpu_id in procs:
        try:
            _, stderr_bytes = proc.communicate(timeout=180)
            if proc.returncode == 0:
                arr = np.load(rpath)
                if len(arr) < 4:
                    logger.error(f"  GPU {gpu_id}: result has only {len(arr)} fields")
                    all_ok = False
                else:
                    logger.info(
                        f"  GPU {gpu_id}: V={arr[0]:.3f} U={arr[1]:.3f} "
                        f"HBA={arr[2]:.3f} HBD={arr[3]:.3f} OK"
                    )
            else:
                msg = stderr_bytes.decode("utf-8", errors="replace")[-300:]
                logger.error(f"  GPU {gpu_id}: worker failed\n  {msg}")
                all_ok = False
        except subprocess.TimeoutExpired:
            proc.kill()
            logger.error(f"  GPU {gpu_id}: timeout")
            all_ok = False
        except Exception as e:
            logger.error(f"  GPU {gpu_id}: {e}")
            all_ok = False
        finally:
            try:
                os.remove(rpath)
            except FileNotFoundError:
                pass

    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    logger.info(
        f"[multiobj] worker verification complete ({time.time() - t0:.1f}s): "
        f"{'all GPUs OK' if all_ok else 'failed'}"
    )
    return all_ok


def make_parallel_batch_evaluate_fn(
    args: argparse.Namespace,
    cwg: ConditionalWeightsGenerator,
    logger,
    worker_script: str,
    gpu_ids: List[str],
    recorder: MultiObjectiveRecorder,
):
    n_gpus = len(gpu_ids)
    pythonpath = os.environ.get("PYTHONPATH", ".")

    def batch_evaluate_fn(positions: np.ndarray) -> List[Tuple[float, float, float, float]]:
        M = positions.shape[0]
        results: List[Tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 0.0)] * M
        t_batch_start = time.time()
        n_rounds = (M + n_gpus - 1) // n_gpus

        for round_idx in range(n_rounds):
            round_start = round_idx * n_gpus
            round_end = min(round_start + n_gpus, M)
            round_pids = list(range(round_start, round_end))
            round_size = len(round_pids)
            t_round = time.time()
            procs = []
            weight_paths = []

            for local_i, particle_idx in enumerate(round_pids):
                gpu_id = str(gpu_ids[local_i % n_gpus])
                uid = uuid.uuid4().hex[:8]
                wpath = os.path.join(tempfile.gettempdir(), f"qmg_mo_w_{uid}.npy")
                rpath = os.path.join(tempfile.gettempdir(), f"qmg_mo_r_{uid}.npy")
                w_c = cwg.apply_chemistry_constraint(positions[particle_idx].copy())
                np.save(wpath, w_c)

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                env["PYTHONPATH"] = pythonpath
                cmd = [
                    sys.executable,
                    worker_script,
                    "--weight_path", wpath,
                    "--result_path", rpath,
                    "--num_heavy_atom", str(args.num_heavy_atom),
                    "--num_sample", str(args.num_sample),
                    "--backend", args.backend,
                    "--report_hbahbd",
                ]
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                procs.append((proc, rpath, particle_idx, gpu_id))
                weight_paths.append(wpath)

            for proc, rpath, particle_idx, gpu_id in procs:
                try:
                    _, stderr_bytes = proc.communicate(timeout=args.subprocess_timeout)
                    if proc.returncode == 0:
                        arr = np.load(rpath)
                        if len(arr) >= 4:
                            results[particle_idx] = (
                                float(arr[0]),
                                float(arr[1]),
                                float(arr[2]),
                                float(arr[3]),
                            )
                        else:
                            logger.warning(
                                f"[parallel] GPU {gpu_id} particle {particle_idx} "
                                f"returned {len(arr)} fields"
                            )
                    else:
                        msg = stderr_bytes.decode("utf-8", errors="replace")[-400:]
                        logger.warning(
                            f"[parallel] GPU {gpu_id} particle {particle_idx} "
                            f"exit={proc.returncode}\n  stderr: {msg}"
                        )
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    logger.warning(
                        f"[parallel] GPU {gpu_id} particle {particle_idx} "
                        f"timeout > {args.subprocess_timeout}s"
                    )
                except Exception as e:
                    logger.warning(
                        f"[parallel] GPU {gpu_id} particle {particle_idx}: {e}"
                    )
                finally:
                    try:
                        os.remove(rpath)
                    except FileNotFoundError:
                        pass

            for path in weight_paths:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

            valid_in_round = sum(1 for idx in round_pids if results[idx][0] > 0)
            logger.info(
                f"  [parallel round {round_idx + 1}/{n_rounds}] "
                f"particles {round_start}..{round_end - 1} "
                f"GPU={[str(gpu_ids[i % n_gpus]) for i in range(round_size)]} "
                f"valid={valid_in_round}/{round_size} "
                f"round={time.time() - t_round:.1f}s "
                f"batch_total={time.time() - t_batch_start:.1f}s"
            )

        recorder.report_batch(results)
        return results

    return batch_evaluate_fn


def main() -> None:
    args = parse_args()
    gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
    effective_n_gpus = min(args.n_gpus, len(gpu_ids))
    gpu_ids = gpu_ids[:effective_n_gpus]

    os.makedirs(args.data_dir, exist_ok=True)
    log_path = os.path.join(args.data_dir, f"{args.task_name}.log")
    logger = setup_logger(log_path)

    scorer = HBAHBDMultiObjective(
        hba_target=args.hba_target,
        hbd_target=args.hbd_target,
        hba_sigma=args.hba_sigma,
        hbd_sigma=args.hbd_sigma,
        chem_weight=args.chem_weight,
    )

    logger.info(f"Task name: {args.task_name}")
    logger.info("Task: ['product_validity_uniqueness', 'HBA', 'HBD']")
    logger.info(f"Condition: ['None', '{args.hba_target:g}', '{args.hbd_target:g}']")
    logger.info("objective: ['maximize', 'minimize', 'minimize']")
    logger.info(f"multi_objective_scalar: {scorer.describe()}")
    logger.info(f"# of heavy atoms: {args.num_heavy_atom}")
    logger.info(f"# of samples: {args.num_sample}")
    logger.info("smarts: None")
    logger.info("disable_connectivity_position: []")
    logger.info(f"CUDA-Q backend: {args.backend}")
    logger.info(f"subprocess_timeout: {args.subprocess_timeout}s")
    logger.info(
        f"initialization: {'Sobol scrambled (seed=0)' if args.sobol_init else f'pseudo-random seed={args.seed}'}"
    )
    logger.info(f"OBL Phase 0: {'on' if args.obl else 'off'}")
    logger.info(f"V-U decoupled mbest: {'on' if args.vu_decouple else 'off'}")
    logger.info(
        f"parallel subprocess pool: N_GPUS={effective_n_gpus} GPU_IDs={gpu_ids}"
    )
    log_gpu_info(logger, gpu_ids)
    log_memory(logger, "startup")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(script_dir, "worker_eval.py")
    if not os.path.exists(worker_script):
        logger.error(f"[ERROR] worker_eval.py not found: {worker_script}")
        sys.exit(1)
    logger.info(f"worker_eval.py: {worker_script}")

    cwg = ConditionalWeightsGenerator(
        args.num_heavy_atom,
        smarts=None,
        disable_connectivity_position=[],
    )
    n_flexible = int((cwg.parameters_indicator == 0.0).sum())
    logger.info(f"Number of flexible parameters: {n_flexible}")
    assert n_flexible == cwg.length_all_weight_vector

    sec_per_eval = 284 if args.num_sample >= 10000 else 142
    rounds_per_iter = (args.particles + effective_n_gpus - 1) // effective_n_gpus
    obl_batches = 1 if args.obl else 0
    total_batches = (args.iterations + 1) + obl_batches
    est_h = total_batches * rounds_per_iter * sec_per_eval / 3600
    total_evals = args.particles * (args.iterations + 1) + (
        args.particles if args.obl else 0
    )
    logger.info(
        f"[config] M={args.particles} T={args.iterations} "
        f"total_evals≈{total_evals} "
        f"rounds_per_batch={rounds_per_iter}x{effective_n_gpus}GPU "
        f"estimated={est_h:.1f}h"
    )

    if not verify_workers_hbahbd(args, cwg, logger, worker_script, gpu_ids):
        logger.error("[ERROR] worker verification failed")
        sys.exit(1)
    log_memory(logger, "after worker verification")

    recorder = MultiObjectiveRecorder(args, scorer, logger)
    batch_evaluate_fn = make_parallel_batch_evaluate_fn(
        args=args,
        cwg=cwg,
        logger=logger,
        worker_script=worker_script,
        gpu_ids=gpu_ids,
        recorder=recorder,
    )

    optimizer = AESOQPSOOptimizer(
        n_params=n_flexible,
        n_particles=args.particles,
        max_iterations=args.iterations,
        logger=logger,
        batch_evaluate_fn=batch_evaluate_fn,
        fitness_fn=scorer.score,
        objective_label="multi_objective(V*U,HBA,HBD)",
        compare_bo_baseline=False,
        seed=args.seed,
        alpha_max=args.alpha_max,
        alpha_min=args.alpha_min,
        data_dir=args.data_dir,
        task_name=args.task_name,
        stagnation_limit=args.stagnation_limit,
        reinit_fraction=args.reinit_fraction,
        mutation_prob=args.mutation_prob,
        ae_weighting=args.ae_weighting,
        pair_interval=args.pair_interval,
        rotate_factor=args.rotate_factor,
        obl=args.obl,
        vu_decouple=args.vu_decouple,
        w_vu=args.w_vu,
        w_v=args.w_v,
        w_u=args.w_u,
        min_u_for_v_track=args.min_u_for_v_track,
        min_v_for_u_track=args.min_v_for_u_track,
        mode_collapse_u_thresh=args.mode_collapse_u_thresh,
    )

    if args.sobol_init:
        sobol_pos = make_sobol_positions(args.particles, n_flexible, logger)
        if sobol_pos is not None:
            optimizer.positions = sobol_pos.copy()
            optimizer.pbest = sobol_pos.copy()
            logger.info(
                f"[Sobol] positions overwritten shape={optimizer.positions.shape} "
                f"range=[{optimizer.positions.min():.4f}, {optimizer.positions.max():.4f}]"
            )
    else:
        logger.info(f"pseudo-random initialization seed={args.seed}")

    log_memory(logger, "before optimize")
    try:
        best_params, best_fitness = optimizer.optimize()
    except Exception as e:
        logger.error(f"[ERROR] optimizer.optimize() failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        log_memory(logger, "after optimize")

    best_npy = os.path.join(args.data_dir, f"{args.task_name}_best_params.npy")
    np.save(best_npy, best_params)
    logger.info(f"best params saved: {best_npy}")
    logger.info(f"final multi-objective score: {best_fitness:.8f}")
    if recorder.best is not None:
        best_json = os.path.join(args.data_dir, f"{args.task_name}_multiobj_best.json")
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(recorder.best, f, indent=2, ensure_ascii=False)
        logger.info(f"best observed metrics saved: {best_json}")
        logger.info(
            "best observed metrics: "
            f"objective={recorder.best['objective']:.8f} "
            f"V*U={recorder.best['product_validity_uniqueness']:.6f} "
            f"V={recorder.best['validity']:.4f} "
            f"U={recorder.best['uniqueness']:.4f} "
            f"HBA={recorder.best['HBA']:.4f} "
            f"HBD={recorder.best['HBD']:.4f}"
        )
    logger.info(f"main log: {log_path}")
    logger.info(f"multi-objective CSV: {recorder.csv_path}")
    log_memory(logger, "shutdown")


if __name__ == "__main__":
    main()
