# τ²-Bench

SpecAgent runs as a tau2 agent (`agent.py`); the Weak-only and Strong-only baselines run the
same agent loop with one model (`baseline_agent.py`). `run_eval.py` registers both with tau2's
runner. Its defaults are the paper setting: review held state-changing actions (writes and
`transfer_to_human_agents`) with chunk size K=6, send replies to the simulated user without
holding them, three trials per task, eight tasks in parallel.

## Setup

```bash
git clone https://github.com/sierra-research/tau2-bench.git
cd tau2-bench && git checkout a2c024725189473d2d7cea3a5cfdbcc67478e41f && pip install -e .
python /path/to/specagent/benchmarks/tau2/apply_orchestrator_patch.py
```

The patch lets the agent execute read-only tool calls inside its own turn (so a draft can
continue from real observations) and tells the orchestrator not to execute them again.

## Run

```bash
export SPECAGENT_WEAK_BASE_URL=http://127.0.0.1:30000/v1 SPECAGENT_WEAK_MODEL=qwen3.5-4b
export SPECAGENT_STRONG_BASE_URL=... SPECAGENT_STRONG_MODEL=qwen3.5-397b-a17b
# user simulator (the paper uses DeepSeek-V4-Flash-0731 for every method)
export SPECAGENT_JUDGE_BASE_URL=... SPECAGENT_JUDGE_MODEL=deepseek-v4-flash-0731

python run_eval.py --domain airline --save-to results/airline_specagent.json
python run_eval.py --domain retail --mode baseline_strong --save-to results/retail_strong.json
```

Thinking is on for acting turns (Weak drafts and Strong repair) and off for the review call.
Per-call token usage and timings are written next to the results (`--specagent-stats`,
`--baseline-stats`) for the resource estimator.
