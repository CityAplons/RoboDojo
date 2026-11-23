# Eval speed: findings and options

Measured on `stack_bowls` with `demo_policy`, one env per process (RTX 5070 Ti Laptop, Ryzen AI 9 HX 370, CPU PhysX).

## Where the time goes

A demo-policy episode runs 800 actions and takes about 62 s (67 s before this branch). Each action takes about 75–84 ms:

| Cost per action | Time | Notes |
| :-- | --: | :-- |
| Physics, 10 PhysX steps | ~38 ms | Fixed overhead per native step. More or fewer PhysX threads barely matters: 2.9 vs 2.7 ms per step. |
| Camera render, 3 × 640×480 RTX | ~22 ms | Already one render per capture. The `performance` preset only saves about 3%. |
| Control Python, 10 steps | ~4–6 ms | Trimmed on this branch. |
| Policy round trip (`update_obs`) | ~4 ms | Raw images over msgpack. This lives in XPolicyLab. |
| `get_obs`, reward, video | ~5 ms | The video share was trimmed on this branch. |

Fixed costs per episode and per launch:

| Fixed cost | Time |
| :-- | --: |
| Reset and settle, per episode | ~15 s |
| Full sim rebuild between episodes | ~2 s |
| Kit startup, per launch | ~25 s |
| `create_eval_env`, per launch | 17 s before this branch, 1 s now |

## Not worth it

| Idea | Why not |
| :-- | :-- |
| Reuse the sim across episodes | Saves only about 2 s per episode. The in-place reload path is also buggy on CPU physics: hidden prims accumulate and render products leak. |
| Cheaper render preset | Saves less than 1 ms per frame and changes every image. |
| Two concurrent evals on one GPU | 1.4× slower per step and memory-bound (about 13 GB RAM per sim plus server). In the measured run it was slower in total than running sequentially. |
| Changing PhysX threads, solver iterations, or warm-up step counts | Changes results, or gains nothing. |

## Changes on this branch (results-neutral)

- **Control path:** joint-id arrays are cached per articulation, and the arm and gripper position targets are set in one write. `MetaControl.get_action` is memoized within a joint snapshot.
- **Video:** frames are written by a background thread. Output is byte-identical.
- **cuRobo planners:** built lazily on the first IK call. RNG state is saved and restored around the build, so random streams are unchanged. Joint-action policies never build them.
- **Multi-task:** one Isaac Sim launch can run several tasks.

## How to use

```bash
# Several tasks in one Isaac Sim launch (split mode; start the server first)
bash scripts/robodojo.sh server --policy-dir XPolicyLab/policy/<POLICY> --task <TASK> --ckpt <CKPT> \
  --policy-env <ENV> --policy-port 18765
bash scripts/robodojo.sh client --policy-dir XPolicyLab/policy/<POLICY> --policy-host localhost \
  --policy-port 18765 --task stack_bowls,push_T,arrange_largest_number --eval-num 1

# Sweep with one launch per client worker
bash scripts/robodojo.sh client --policy-dir XPolicyLab/policy/<POLICY> --policy-host localhost \
  --policy-port 18765 --only stack_bowls,push_T --single-launch
```

In multi-task mode:

- Results go to `eval_result/.../<RUN_ID>_<task>/`, the same layout the sweep uses.
- Each finished task writes `_done_<RUN_ID>_<task>.json`, so a restart (in-process or from the bash retry loop) skips finished tasks.
- A failed task does not stop the rest, but the process exits with code 1. Set `ROBODOJO_FAIL_FAST=1` to stop at the first failure instead.

Each sweep task still gets its own PASS/FAIL row, from its `_result.json` (`eval_time >= 1`) and its done marker.

| Env var | Effect |
| :-- | :-- |
| `ROBODOJO_PROFILE=1` | Writes `profile.json` (wall-clock spans) next to `_result.json`. |
| `ROBODOJO_CPROFILE=1` | Writes `run_eval.prof` (cProfile). Read it with `python -m pstats`. |
| `ROBODOJO_EAGER_PLANNER=1` | Builds cuRobo planners at env creation (the old behavior). |

## Caveats

- **Persisted Kit flags:** never pass `--/persistent/...` Kit flags in experiments. They are saved to `.../isaacsim/kit/data/Kit/IsaacLab/3.0/user.config.json`. `numThreads=1` there deadlocks PhysX init on every later launch.
- **Policy server task:** a single policy server serves every task in the list, but it was started with one `--task`. See the next section.

## Caveat: one policy server for many tasks

**How it works**

- **What the server is told.** The server gets one task name when it starts: `robodojo.sh server --task X` passes `task_name=X` through `setup_eval_policy_server.sh` into `model_cfg`.
- **What it never learns.** Nothing tells the server when the client moves on to the next task.
- **What changes per episode.** The client sends `reset` at the start of every episode, so per-episode policy state is cleared. It also sends a per-task `trial_id`.
- **What does not change.** Everything the model built from `task_name` at load time stays fixed for the whole list.

**Is it safe for your policy?** Check how its `XPolicyLab/policy/<POLICY>/model.py` uses `model_cfg["task_name"]`. 42 of the 70 adapters read it, in three ways:

1. **Fallback prompt only: safe.**
   - Adapters: EventVLA, StarVLA, Spark_0, Meituan_Robotics_0, LDA_1B, G05, GigaWorldPolicy, and the many `default_prompt = prompt or task_name` adapters (Abot_M0, RDT_1B, SmolVLA, X_VLA, OpenVLA_OFT, GR00T_N17, …).
   - The task name is used only when the observation has no `instruction`. RoboDojo always sends the task's own `instruction`, so the server's task name is never used.
   - Before relying on this, confirm that your adapter's `default_prompt` really is only a fallback.
2. **Changes what the model loads or how it normalizes: unsafe.** Every later task would silently be evaluated with the first task's settings. Known cases:
   - `Evo_1`: per-task normalization stats (`robotwin_<task_name>`).
   - `DP`: `model_training_config['task'] = task_name`, which selects the task's training config and checkpoint.
   - `H_RDT`: loads the language embedding `<embedding_dir>/<task_name>.pt`.
   - `Mem_0`: selects the task type (Mn or M1) from the task name.
   - `AHA_WAM`: task config override `task=<task_name>`.
   - `A1`: the checkpoint run dir falls back to `task_name` when `ckpt_name` is empty.
   - Server scripts that read `task_name` beyond passing it on also need a look. These are AHA_WAM, FastWAM, Dexora_1B, H_RDT, Mem_0, EventVLA, StarVLA, Spark_0 and GigaWorldPolicy.
3. **Cosmetic: safe.** For example, `ViGAR` uses it in its log directory name.

**What to do**

- **Unsafe policies:** run one `server` and one `client --task <task>` per task, as before. Or pass `--single-launch` only for groups of tasks that share a checkpoint.
- **Any policy:** a single-task client run is the reference. If a policy's results differ between single-task and multi-task runs on the same seeds, treat it as unsafe.
- **`robodojo.sh eval`** still starts its own server, with exactly the `--task` it was given. Pass a single task there.
