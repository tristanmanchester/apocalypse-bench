#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-apocbench-retrieval-debug-10.generated.json}
RUN_ID=${RUN_ID:-retrieval-debug-10-gemma31b-$(date +%Y%m%d-%H%M%S)}
CONDITIONS=${CONDITIONS:-direct bm25 hybrid bm25-research}
REPEATS=${REPEATS:-10}
LOG=${LOG:-logs/${RUN_ID}.log}
RESUME_EXISTING=${RESUME_EXISTING:-0}

cd "$(dirname "$0")/.."
mkdir -p "$(dirname "$LOG")"

model_ids_for_condition() {
  local condition=$1
  local ids=()
  local idx
  for idx in $(seq 1 "$REPEATS"); do
    ids+=("gemma31b-${condition}-r$(printf '%02d' "$idx")")
  done
  local IFS=,
  printf '%s' "${ids[*]}"
}

{
  echo "RUN_ID=${RUN_ID}"
  echo "CONFIG=${CONFIG}"
  echo "CONDITIONS=${CONDITIONS}"
  echo "REPEATS=${REPEATS}"
  echo "RESUME_EXISTING=${RESUME_EXISTING}"
  date -Is

  first=1
  for condition in $CONDITIONS; do
    models=$(model_ids_for_condition "$condition")
    echo "chunk_start condition=${condition} models=${models}"
    if [[ "$first" == 1 && "$RESUME_EXISTING" != 1 ]]; then
      pnpm -s dev run -c "$CONFIG" --quiet --json --models "$models" "$RUN_ID"
      first=0
    else
      pnpm -s dev resume -c "$CONFIG" --quiet --json --models "$models" "$RUN_ID"
      first=0
    fi
    echo "chunk_done condition=${condition}"
  done

  date -Is
  echo "EXIT:0"
} 2>&1 | tee "$LOG"
