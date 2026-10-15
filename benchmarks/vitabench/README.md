# VitaBench

Multi-turn life-service tasks with a simulated user, scored by a rubric judge
([VitaBench](https://github.com/meituan-longcat/vitabench)). VitaBench's agent posts plain
OpenAI chat completions, so SpecAgent runs as the gateway server (`python -m specagent.server
--setting vitabench`) in front of the Weak and Strong models. Only the agent goes through the
gateway; the user simulator and the judge call their own endpoints.

## Setting

The paper runs the Delivery domain: all 100 tasks, in English, one trial per task, at most 100
steps, eight tasks in parallel. The user simulator and the judge are DeepSeek-V4-Flash-0731 for
every method. These are the defaults of `run_eval.py`; the paper reports the mean of two runs of
every method.

SpecAgent (`--setting vitabench`):

- Only state-changing tool calls are reviewed. Which calls change state comes from VitaBench's
  own labels: a tool decorated `@is_tool(ToolType.WRITE)` is state-changing (27 of the 66 tools),
  every READ or GENERIC tool is a read. `run_eval.py` extracts the labels from the checkout into
  `tool_classes.json` and passes it to the server with `--tool-class-file`; the name heuristics
  apply only to tools the file does not list.
- Replies to the simulated user commit without review, and so does a chunk of K=6 reads.
- Weak drafts with thinking on, `reasoning_effort` high, and an output cap of 8,192 tokens.
- A review runs with thinking off and at most 1,024 output tokens, and does not see Weak's
  reasoning. A rejection commits the reviewer's schema-valid replacement call; otherwise Strong
  takes one repair turn (thinking on, `reasoning_effort` high, at most 16,384 tokens) that
  commits without another review.
- The review payload carries the task id, input-integrity labels, and workspace anchor that the
  paper run's reviewer received (`--verifier-payload-labels legacy`, `--workspace-dir`).
  `tests/golden/` checks that the gateway's model traffic in this setting is byte-identical to
  that of the code the paper run used.

The thinking switches are SGLang `chat_template_kwargs`, as in the paper's deployments. For
another provider, pass its own switches to the server, for example
`--gateway-arg=--verifier-extra-body='{"enable_thinking":false}'`.

The Weak-only and Strong-only arms call the model directly with thinking on; Weak-only also sends
`reasoning_effort` high and the same 8,192-token cap (`BASELINE_AGENT_EXTRA` in `run_eval.py`,
replaced by `--agent-extra-json`). The paper's AgentCollab numbers come from the authors'
reimplementation, which is not part of this repository.

## Setup

```bash
git clone https://github.com/meituan-longcat/vitabench.git && cd vitabench
git checkout 742e240855bf8686a0842360749d5ea970ea3987 && pip install -e .
export VITABENCH_DIR=$PWD
pip install -e /path/to/specagent
```

Each run applies `apply_session_patch.py` to the checkout (idempotent; it stops if the files have
drifted from the pinned commit). The patch sends an `x-specagent-session-id` header per task
attempt, so concurrent tasks keep separate gateway sessions; makes the model read timeout
configurable (`VITA_LLM_TIMEOUT_S`, default 600 s); keeps the `model:` field of a shared
models.yaml entry out of the request arguments; and adds background (`VITA_ASYNC_EVAL`) and
deferred (`VITA_DEFER_EVAL`) judging.

## Run

```bash
export SPECAGENT_WEAK_BASE_URL=http://127.0.0.1:30000/v1 SPECAGENT_WEAK_MODEL=Qwen3.5-4B
export SPECAGENT_STRONG_BASE_URL=... SPECAGENT_STRONG_MODEL=Qwen3.5-397B-A17B
export SPECAGENT_STRONG_API_KEY=...
# user simulator and judge (SPECAGENT_USER_* / SPECAGENT_EVALUATOR_* override them per role)
export SPECAGENT_JUDGE_BASE_URL=... SPECAGENT_JUDGE_MODEL=deepseek-v4-flash-0731

python run_eval.py --mode specagent --output-dir out/   # SpecAgent
python run_eval.py --mode weak      --output-dir out/   # Weak-only
python run_eval.py --mode strong    --output-dir out/   # Strong-only
```

`--*-auth-scheme raw` sends a key without the `Bearer` prefix. Each run writes
`out/<mode>_<tag>/`: `summary.json` (reward per task and the mean), `simulations.json`
(VitaBench's results), `vita_run.log`, and `models.yaml` (holds the keys; mode 600). A SpecAgent
run adds `tool_classes.json`, the gateway log, and `gateway.jsonl` with one line per request:
the decision, the verdict, the repair, and every model call with its token usage.

Finished tasks are judged in a background pool while the next tasks run (`--no-async-eval` judges
inline). `--defer-eval` skips judging; `reeval.py` then fills the rewards in afterwards.

The tests in `tests/vitabench/` that read the benchmark run when `VITABENCH_DIR` points at the
checkout.
