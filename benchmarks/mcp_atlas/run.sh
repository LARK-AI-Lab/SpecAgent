#!/usr/bin/env bash
# Run one MCP-Atlas arm: weak, strong, or specagent.
#
# usage: run.sh <weak|strong|specagent> <out_dir> [tasks.csv]
#
# Required environment:
#   MCP_ATLAS_DIR     MCP-Atlas checkout with harness/mcp-atlas.patch applied and the agent
#                     harness built (see README.md)
#   WEAK_BASE_URL, WEAK_MODEL        OpenAI-compatible Weak endpoint (weak, specagent)
#   STRONG_BASE_URL, STRONG_MODEL    OpenAI-compatible Strong endpoint (strong, specagent)
#   SPECAGENT_STRONG_API_KEY         Strong API key (specagent); the strong arm reads its key
#                                    from the harness's own .env
# Optional:
#   STRONG_EXTRA_BODY, VERIFIER_EXTRA_BODY   provider thinking controls as JSON; the defaults
#                                            are the server's (review thinking disabled)
#   GATEWAY_PORT (18081), HARNESS_PORT (3020), CONCURRENCY (4), TIMEOUT (1800)
set -euo pipefail

ARM=${1:?weak|strong|specagent}
OUT=${2:?out_dir}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS=${3:-$HERE/data/tasks_eval.csv}
A=${MCP_ATLAS_DIR:?}
GATEWAY_PORT=${GATEWAY_PORT:-18081}
HARNESS_PORT=${HARNESS_PORT:-3020}
mkdir -p "$OUT/$ARM"

case "$ARM" in
  weak) BASE_URL=${WEAK_BASE_URL:?}; MODEL=${WEAK_MODEL:?} ;;
  strong) BASE_URL=${STRONG_BASE_URL:?}; MODEL=${STRONG_MODEL:?} ;;
  specagent)
    EXTRA=()
    [[ -n "${STRONG_EXTRA_BODY:-}" ]] && EXTRA+=(--strong-extra-body "$STRONG_EXTRA_BODY")
    [[ -n "${VERIFIER_EXTRA_BODY:-}" ]] && EXTRA+=(--verifier-extra-body "$VERIFIER_EXTRA_BODY")
    python -m specagent.server \
      --weak-base-url "${WEAK_BASE_URL:?}" --weak-model "${WEAK_MODEL:?}" \
      --strong-base-url "${STRONG_BASE_URL:?}" --strong-model "${STRONG_MODEL:?}" \
      --port "$GATEWAY_PORT" "${EXTRA[@]}" \
      --telemetry "$OUT/specagent/gateway.jsonl" \
      >> "$OUT/specagent/gateway.log" 2>&1 &
    GATEWAY_PID=$!
    trap 'kill $GATEWAY_PID 2>/dev/null || true' EXIT
    for _ in $(seq 1 30); do
      curl -s -m 2 "http://127.0.0.1:$GATEWAY_PORT/v1/models" >/dev/null && break; sleep 1
    done
    BASE_URL="http://127.0.0.1:$GATEWAY_PORT"
    MODEL="specagent"
    ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

(cd "$A/services/agent-harness" && PORT=$HARNESS_PORT LLM_BASE_URL=$BASE_URL \
  node dist/index.js >> "$OUT/$ARM/harness.log" 2>&1) &
HARNESS_PID=$!
trap 'kill $HARNESS_PID ${GATEWAY_PID:-} 2>/dev/null || true' EXIT
sleep 5

(cd "$A" && HARNESS_URL="http://127.0.0.1:$HARNESS_PORT" python run_eval.py \
  --input "$TASKS" --output "$OUT/$ARM/outputs.csv" --model "$MODEL" \
  --concurrency "${CONCURRENCY:-4}" --timeout "${TIMEOUT:-1800}" --skip-health-check)
