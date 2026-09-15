# Babel smoke checklist (LangGraph floor) — SLURM / preempt

No interactive GPU node: submit from login. **Every job requests GPU(s).**

```bash
ssh babel
cd ~/AI-agent-talkshow
mkdir -p slurm-logs

# 1) Deps (1 GPU)
sbatch deploy/slurm-install-deps.sh

# 2) Planner tests (1 GPU)
sbatch deploy/slurm-test-show-graph.sh

# 3) Full stack — 3 GPUs; slots/envs from .env (defaults 0/1/2)
sbatch deploy/slurm-talkshow-3gpu.sh
# When running: cat logs/slurm-<jobid>/gpu-bind.txt
# Laptop: pnpm dev + token → room talkshow-dev
```

Scripts use `--partition=preempt` and `--qos=preempt_qos`.

Expected: floor behavior unchanged; only orchestration is LangGraph.
