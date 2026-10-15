# MCP-Atlas

Multi-step tool use against real Model Context Protocol servers, scored by an LLM judge on the
share of reference claims the final answer covers ([MCP-Atlas](https://github.com/scaleapi/mcp-atlas)).
The harness talks to any OpenAI-compatible endpoint, so SpecAgent runs as the gateway server
(`python -m specagent.server`) in front of the Weak and Strong models. Its defaults are the
paper's MCP-Atlas setting.

## Setup

1. Clone MCP-Atlas at commit `f24ba3fb0bfa484c86acb28431fad6d7282455f9` and apply our patch:

   ```bash
   git clone https://github.com/scaleapi/mcp-atlas.git && cd mcp-atlas
   git checkout f24ba3fb0bfa484c86acb28431fad6d7282455f9
   git apply /path/to/specagent/benchmarks/mcp_atlas/harness/mcp-atlas.patch
   cp /path/to/specagent/benchmarks/mcp_atlas/harness/*_mcp_wrapper.py \
      services/agent-environment/src/agent_environment/
   ```

   The patch records per-task token usage and wall time, marks runs that end without a
   terminal answer as errors, retries transient network failures, pins the MCP server
   packages, bounds arXiv result pages, asks the judge for a bare JSON object, and sets a descriptive Wikipedia User-Agent
   (override with `WIKIPEDIA_USER_AGENT`).
2. Build and start the agent environment and harness as described in the MCP-Atlas README.
3. Install this repository (`pip install -e .` at the repository root).

## Tasks

The task files are taken from the public MCP-Atlas dataset
([ScaleAI/MCP-Atlas](https://huggingface.co/datasets/ScaleAI/MCP-Atlas)) and remain under its
terms. `data/tasks.csv` holds the 107 public tasks whose ground-truth trajectories use only servers
that run without private API keys (the harness's default keyless servers plus `github`).
`data/excluded_task_ids.txt` lists 6 tasks on which the harness did not complete runs
reliably; `data/tasks_eval.csv` is the remaining 101 tasks used in the paper.

## Run

```bash
export MCP_ATLAS_DIR=/path/to/mcp-atlas
export WEAK_BASE_URL=http://127.0.0.1:30000/v1 WEAK_MODEL=qwen3.5-4b
export STRONG_BASE_URL=https://provider/v1 STRONG_MODEL=... SPECAGENT_STRONG_API_KEY=...

./run.sh weak      out/          # Weak-only
./run.sh strong    out/          # Strong-only (key from the harness .env)
./run.sh specagent out/          # SpecAgent
```

The review call runs with thinking disabled. The server's default review body is
`{"thinking":{"type":"disabled"}}`; set `VERIFIER_EXTRA_BODY` to the provider's own switch
(for example `{"enable_thinking":false}`) when it differs. Repair turns keep the provider's
default thinking unless `STRONG_EXTRA_BODY` sets it.

`tools/chat_path_shim.py` serves the harness's `/v1/chat/completions` path in front of a
provider whose chat path differs (set `UPSTREAM_CHAT_URL` and `UPSTREAM_API_KEY`).

## Score

```bash
export JUDGE_BASE_URL=https://judge-provider/v1 JUDGE_API_KEY=...
./judge.sh out/specagent/outputs.csv out/specagent/judge specagent
```

A task passes when its response covers at least 75% of the graded claims.
