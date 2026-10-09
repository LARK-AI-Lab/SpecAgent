# SpecAgent

Code for *Trajectory-Level Speculation with Semantic Verification for Tool-Using Agents*.

SpecAgent turns the strong model of an agent from the author of every step into a reviewer of
steps drafted by a weak model:

1. **Draft.** The weak model drafts up to K steps. Read-only tool calls execute immediately,
   so every drafted step is grounded in real observations. The first state-changing action
   (or the final answer, where it is the graded deliverable) is held, not executed. A chunk
   of K reads commits without review.
2. **Verify.** The strong model reviews the chunk once, with thinking off, and returns
   `{reason, accepted_steps, replacement_action}`: the longest semantically valid prefix,
   why the next step is wrong, and optionally a replacement tool call.
3. **Repair.** A valid replacement executes directly. Otherwise the strong model takes one
   repair turn, with thinking on, from the committed history and the reason. The repair is
   committed without a second review; the rejected suffix is discarded.

## Layout

| Path | Contents |
| --- | --- |
| `src/specagent/runtime/gateway.py` | The protocol as an OpenAI-compatible chat gateway (draft, review, repair, commit) |
| `src/specagent/server.py` | HTTP server for the gateway: `python -m specagent.server` |
| `src/specagent/protocol/` | Verifier prompt and parser, prefix-commit state, action classification, and the FLOPs / HBM-read estimator |
| `benchmarks/tau2/` | τ²-Bench agent, baselines, and runner |
| `benchmarks/automationbench/` | AutomationBench runner and SpecAgent profile |
| `benchmarks/mcp_atlas/` | MCP-Atlas harness patch, task set, run and scoring scripts |
| `tests/` | Unit tests and recorded model traffic for the paper settings |

Every runner and the server default to the settings reported in the paper; each benchmark
directory has a README with setup and run commands.

## Install

```bash
pip install -e ".[tau2,automationbench]"
```

Python 3.10 or newer. Benchmark-specific dependencies (tau2-bench, AutomationBench, the
MCP-Atlas harness) are installed as described in each benchmark README. Model serving is out
of scope: point the runners at any OpenAI-compatible endpoints (the paper serves the weak
model with SGLang).

## Resource estimates

`specagent.protocol.flops` and `specagent.protocol.hbm` estimate per-request FLOPs and
batch-1-equivalent HBM-read from token counts (input, output, prefix-cached) and the model
architecture, for Qwen3.5-4B, Qwen3.5-397B-A17B, DeepSeek-V4-Flash-0731, and GLM-5
(`src/specagent/protocol/architectures/`). The runners record per-call token usage so these
estimates can be computed for every method alike.

How much of each request's input is a reused prefix comes from `specagent.protocol.prefix_reuse`,
the paper's model of radix prefix caching: requests are grouped into sessions (task, trial,
model, role, replica), and drafting, verification and repair requests each reuse the running
maximum of their own backbone. Its `cached_input_tokens` output is the cached count the FLOPs
and HBM-read estimators take.

## Tests

```bash
pip install pytest && pytest
```

`tests/golden/` replays scripted sessions under the paper settings and checks that every
request sent to the models is unchanged; the tau2 part runs when tau2-bench with the
orchestrator patch is installed.

## License

MIT
