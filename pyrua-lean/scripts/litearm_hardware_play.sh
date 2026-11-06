#!/usr/bin/env bash
# Run one hardware episode with the fixed real-robot adapter and no VLA.
#
#   LITEARM_HW_CONFIG=/abs/station.json \
#   LITEARM_HW_INSTRUCTION='put the red block in the tray' \
#   LITEARM_HW_MODE=offline-fake \
#   scripts/litearm_hardware_play.sh
#
# Modes are explicit: read-only (default), offline-fake or live.  The output
# directory is unique per invocation and is never reused.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${LITEARM_HW_CONFIG:?set LITEARM_HW_CONFIG to an absolute station JSON}"
INSTRUCTION="${LITEARM_HW_INSTRUCTION:?set LITEARM_HW_INSTRUCTION}"
if [[ "${CONFIG}" != /* ]]; then
  echo "LITEARM_HW_CONFIG must be absolute: ${CONFIG}" >&2
  exit 2
fi
MODE="${LITEARM_HW_MODE:-read-only}"
case "${MODE}" in
  read-only|offline-fake|live) ;;
  *) echo "LITEARM_HW_MODE must be read-only, offline-fake or live" >&2; exit 2 ;;
esac
ENV_NAME="${LITEARM_HW_AGENT_ENV:-pyrualean-hw-agent}"
BASE_OUT="${LITEARM_HW_RESULTS:-${ROOT}/runs_hw}"
BUDGET="${LITEARM_HW_BUDGET_S:-1200}"
MAX_STEPS="${LITEARM_HW_MAX_STEPS:-300}"
TASK_ID="${LITEARM_HW_TASK:-hardware_task}"
RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
OUT="${BASE_OUT}/${RUN_ID}"
if [[ -e "${OUT}" ]]; then
  echo "refusing to reuse output directory: ${OUT}" >&2
  exit 2
fi
mkdir -p "${BASE_OUT}"
cd "${ROOT}"
args=(
  pyrualean play
  --robot litearm_real
  --arm cells
  --primitives no-vla
  --hardware-config "${CONFIG}"
  --instruction "${INSTRUCTION}"
  --execution-mode "${MODE}"
  --task "${TASK_ID}"
  --budget-s "${BUDGET}"
  --max-episode-steps "${MAX_STEPS}"
  --out "${OUT}"
)
exec conda run --no-capture-output -n "${ENV_NAME}" "${args[@]}"
