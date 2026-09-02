# Running on the NCHC DGX cluster

Operational notes accumulated while running ~400 jobs. Most of these cost hours
to learn; they are recorded so they only cost that once.

---

## Partitions and the time-limit trap

```
PARTITION   NODES          TIME LIMIT
nchc        DGX101–106     2-00:00:00   (48 h)
jhub        DGX101–111     8:00:00
large       DGX101–111     4:00:00
```

All eleven nodes carry **8 × Tesla V100-SXM2-16GB, compute capability 7.0**, so
CUDA-Q 0.7.1 (which ships only sm_70 SASS) runs everywhere. GPU type is not the
constraint — the **time limit** is.

A full-budget run (9,664 evaluations, M = 64, 1,000 shots) takes roughly
**19 hours**. That fits in `nchc` but not in `jhub` or `large`. Using only `nchc`
caps you at six nodes, one of which is usually drained.

### Reaching DGX107–111: chained jobs

`optimizers/base.py` checkpoints after every batch and `--resume` restores from
the CSV plus `{task}_state.npz`. A long run can therefore be split across several
short jobs on `jhub`, each picking up where the last stopped:

```bash
PREV=""
for SEG in 1 2 3; do
  DEP=""; [ -n "$PREV" ] && DEP="--dependency=afterany:$PREV"
  PREV=$(sbatch --parsable -N 1 -p jhub --time=07:45:00 --gres=gpu:4 \
    -J "task.p${SEG}" $DEP \
    --export=ALL,ALGO=cmaes,SEED=10,MAX_EVALS=9664,RESUME=1,TASK_NAME=my_task \
    benchmark/benchmark.slurm)
done
```

`afterany` rather than `afterok`: a segment that exits on the wall clock is not
"ok", but its checkpoint is still valid and the next segment must run.

Moving four seeds onto `jhub` in this way took a 30-job experiment from 6
concurrent runs to 17, and the projected wall time from three days to about one.

> **Data-corruption hazard.** Two jobs sharing a `TASK_NAME` will append to the
> same CSV simultaneously and destroy it. Before and after any requeue, assert
> that each task has at most one job in `RUNNING`:
>
> ```bash
> squeue -u $USER -h -t RUNNING -o "%j" | sed 's/\.p[0-9]$//' \
>   | sort | uniq -c | awk '$1>1 {print "CONFLICT: " $2}'
> ```
>
> Chained segments must be `PENDING (Dependency)`, never concurrent.

---

## Node quirks

| Node | Observed |
|---|---|
| DGX102 | Measured **~100 s/eval against ~7 s/eval** elsewhere — roughly 14× slower — while `sinfo` showed no contention (4/76 CPUs allocated). Cause never identified. Exclude it. |
| DGX106, DGX101 | Have each appeared as `drained` with `Kill task failed`, then recovered. Check state at submission time rather than trusting a previous run's exclusion list. |
| DGX107 | Frequently `mixed` with another user's allocation; accepts fewer concurrent jobs. |

```bash
sbatch --exclude=DGX102,DGX106 ...
```

Re-check `sinfo -p nchc -N -o "%.10N %.9T %.24E"` before each submission. An
exclusion list copied from an earlier experiment silently wastes capacity: one
batch here ran on three nodes for hours because it still excluded two that had
since recovered.

---

## Throughput

**Two workers per node, not eight.** Measured with 4 nodes × 8 workers, 5,000
shots, 600 s timeout: only **17 of 32** evaluations returned valid results. The
rest were silently lost — no error, just zeros folded into the fitness. Two per
node is the stable configuration.

**Cost scales with shots, roughly linearly.** About 7 s/eval at 1,000 shots on a
healthy node, so 9,664 evaluations ≈ 19 h. At 5,000 shots the same run is ~5×
longer, which is why the shot count is a first-order experimental design decision
and not a detail.

**Sequential BO is far more expensive per evaluation than any population method,**
and its GP cost grows as O(n³). `optimizers/bayesopt.py` caps GP training points
(`--max_gp_points`, default 400) and retunes hyperparameters every 25 iterations
(`--tune_every`) purely to make it finish. Both are exposed on the CLI because a
cap that changes results must be varied from the experiment layer, not frozen in
a constructor default.

---

## Submission ordering

Submit **seed-major**, not algorithm-major:

```bash
for SEED in $(seq 0 9); do
  for ALGO in qpso rr_qpso cmaes; do submit $ALGO $SEED; done
done
```

Algorithm-major ordering lets the queue consume every seed of the first algorithm
before starting the second. If the batch is cut short, algorithm-major leaves you
with complete data for some arms and none for others — useless for a paired
design. Seed-major leaves every arm equally advanced, so a partial batch is still
analysable at a reduced budget.

---

## Failure modes seen in practice

**`Job credential expired`.** An intermittent SLURM fault that aborts an entire
`srun` step; the other nodes' tasks are killed with it. Immediately retrying the
same step succeeds. A long run has hundreds of rounds, so one unretried
occurrence zeroes an entire iteration's fitness. `run_qpso_qmg_cudaq.py` retries
via `--srun_retries` (default 3), and only for particles whose result file is
missing — a worker that genuinely failed writes `[0,0,0,0]`, so the file exists
and the fault-tolerance semantics are preserved.

**Requeued jobs and `RESUME=0`.** SLURM requeues jobs routinely (node drain,
preemption), and a requeued job reuses its original `--export`, including
`RESUME=0`. That combination truncated two runs holding 4,160 and 4,992
evaluations — roughly 25 GPU-hours — within 44 seconds of restarting. Always
submit with `RESUME=1`. `optimizers/base.py` now also refuses to overwrite a
non-empty CSV and resumes instead, with a warning.

**`/tmp` is node-local.** Multi-node dispatch must place its exchange directory on
beegfs. Writing to `/tmp` appears to work — the parent writes, the agent reads its
*own* node's `/tmp` — and every particle silently returns zero.
`run_qpso_qmg_cudaq.py` rejects a `job_dir` under `/tmp`, `/var/tmp` or `/dev/shm`.

**Verification thresholds.** The multi-node self-test originally required all G
slots to return V > 0. At 100 probe shots a legitimate statistical zero is
expected, and this aborted valid jobs twice. It now requires result *files* for
every slot (an infrastructure criterion) plus a V > 0 ratio of at least 60% (a
statistical one).

**`watch -n 5 nvidia-smi` segfaults on DGX111.** Use `while true; do clear; nvidia-smi; sleep 10; done`.

---

## Keeping the cluster copy current

The cluster working directory is a git clone of this repository. **Update it with
`git pull`, not by copying selected files.**

```bash
cd ~/sqmg_project-cudaq && git pull --ff-only origin main
```

A hand-maintained list of "files to sync" will eventually omit one. It did:
`evaluator.py` was missing from the list, so a fix that was correct locally was
absent on the cluster, and an experiment ran against the old code. `git pull`
does not have that failure mode.

Copying files over also fights git directly. Anything pushed by SFTP shows up on
the cluster as a *modified tracked file*, which blocks the next `git pull` and has
to be resolved by hand each time. The SFTP tool has therefore been retired — it
now prints this guidance and exits non-zero rather than copying anything. If a
pull is blocked by modified tracked files, `SQMG/scripts/dgx_resolve_and_pull.sh`
hashes each against `origin/main`, restores the identical ones, backs up any that
differ, and then pulls.

Two things make this safe while jobs are running:

- Every `results*/` directory is git-ignored, so `git pull` and even
  `git reset --hard` cannot touch experiment data. Verify with
  `git check-ignore -q results_xxx` before any reset.
- Never use `git clean -fdx`. The `-x` flag removes ignored files, which is
  precisely the experiment data. `git clean -fd` is safe.
- Running processes have already imported their modules, so changing files
  underneath them has no effect on jobs in flight. Newly spawned workers pick up
  the new code, which is normally what you want.

If `git pull` refuses because an untracked file would be overwritten, compare it
against the incoming version before deleting it:

```bash
sha256sum path/to/file
git show origin/main:path/to/file | sha256sum
```

Three-way agreement between the local machine, the cluster and GitHub can be
checked with `SQMG/tools/sync_three_way.py`, which normalises line endings before
hashing. Include `.json` in whatever you compare — the statistics files are the
sole source of the numbers in the documents, and omitting them once let the
cluster sit a commit behind while the check reported "synchronised".

---

## Monitoring

Report evaluation counts and node throughput, not fitness values, while a
pre-registered experiment is in flight — the analysis plan for experiments 4 and
7 commits to running no hypothesis test on partial data. Interim looks are for
detecting stragglers and dead nodes.

```bash
# per-run throughput and wall-clock projection
squeue -u $USER -h -t RUNNING -o "%j|%N|%M|%L"   # name, node, elapsed, remaining
# then: rate = elapsed / rows_in_csv;  needed = (target - rows) * rate
```

A run whose projected remaining time exceeds its remaining wall clock will
produce nothing. Cancel it and resubmit with `RESUME=1` onto a healthy node
rather than letting it burn to the limit.
