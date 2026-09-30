#!/usr/bin/env bash
# Score an arm's outputs.csv with the MCP-Atlas claim judge (a task passes at coverage >= 0.75).
#
# usage: judge.sh <outputs.csv> <out_dir> <arm_name>
#
# Requires MCP_ATLAS_DIR, JUDGE_BASE_URL and JUDGE_API_KEY. Judge calls go through
# tools/judge_usage_proxy.py so every call's token usage lands in <out_dir>/judge_usage.jsonl.
set -euo pipefail

MODEL_FILE=${1:?outputs.csv}
OUT=${2:?out_dir}
ARM=${3:?arm_name}
A=${MCP_ATLAS_DIR:?}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=${JUDGE_PROXY_PORT:-18300}
mkdir -p "$OUT"

python "$HERE/tools/judge_usage_proxy.py" "$PORT" "$OUT/judge_usage.jsonl" &
PROXY_PID=$!
trap 'kill $PROXY_PID 2>/dev/null || true' EXIT
sleep 2

# The proxy adds the real key; the scorer only needs a non-empty placeholder.
(cd "$A" && EVAL_LLM_API_KEY=proxy python services/scoring/score_claims.py \
  --groundtruth-file="$HERE/data/tasks.csv" --model-file="$MODEL_FILE" \
  --model-name="$ARM" --evaluator-model="${JUDGE_MODEL:-gemini-3.1-pro-preview}" \
  --base-url="http://127.0.0.1:$PORT" --output-dir="$OUT" --concurrency=4) > "$OUT/score.log" 2>&1
grep -m1 'pass_rate_0.75' "$OUT/score.log"
