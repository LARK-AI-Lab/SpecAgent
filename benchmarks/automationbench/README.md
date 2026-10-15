# AutomationBench

Long business workflows scored on the final environment state. `run.py` runs the benchmark
through the `verifiers` environment with either SpecAgent (`--protocol rp_fused`, the default)
or one model alone (`--protocol solo --role weak|strong`). The SpecAgent profile is in
`rp_fused_protocol.py`: state-changing tool calls and the final message are held for review,
K=8 reads commit without review, and a rejected tool call may be repaired by the verifier's
replacement call.

Defaults are the paper run: the six public domains (finance, hr, marketing, operations, sales,
support; 600 tasks), one trajectory per task, at most 50 turns, eight tasks in parallel.

## Setup

`run.py` imports the benchmark from a source checkout, so clone AutomationBench at its 1.0.6
release and point `AUTOMATIONBENCH_ROOT` at it (or place the checkout at
`third_party/AutomationBench` in this repository):

```bash
git clone https://github.com/zapier/AutomationBench.git
git -C AutomationBench checkout 6d210543b7a046f0f451c828cd2dadef774276eb   # 1.0.6
pip install -e ./AutomationBench "verifiers==0.2.1"
export AUTOMATIONBENCH_ROOT=$PWD/AutomationBench
```

Run the commands below from this directory.

## Run

```bash
export SPECAGENT_WEAK_BASE_URL=... SPECAGENT_WEAK_API_KEY=... SPECAGENT_WEAK_MODEL=qwen3.5-4b
export SPECAGENT_STRONG_BASE_URL=... SPECAGENT_STRONG_API_KEY=... SPECAGENT_STRONG_MODEL=...

python run.py --run-name specagent                                  # SpecAgent
python run.py --run-name weak --protocol solo --role weak           # Weak-only
python run.py --run-name strong --protocol solo --role strong       # Strong-only
```

Runs checkpoint per shard; add `--resume` to continue an interrupted run.
