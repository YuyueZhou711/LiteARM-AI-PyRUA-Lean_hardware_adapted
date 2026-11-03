#!/usr/bin/env bash
# Start the LiteArm policy server for a split evaluation.
#
# Usage:
#   POLICY_PORT=18231 BIND_HOST=0.0.0.0 bash scripts/run_server.sh [--task NAME] [--seed N]
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
cd "${REPO_ROOT}"

task="${TASK:-stack_blocks}"
seed="${SEED:-0}"
port="${POLICY_PORT:-18231}"
bind_host="${BIND_HOST:-0.0.0.0}"
extra=()

while [ $# -gt 0 ]; do
    case "$1" in
        --task) task="${2:?--task needs a value}"; shift 2 ;;
        --seed) seed="${2:?--seed needs a value}"; shift 2 ;;
        --) shift; extra=("$@"); break ;;
        *) extra+=("$1"); shift ;;
    esac
done

if [ -f config/env.sh ]; then
    # shellcheck disable=SC1091
    source config/env.sh
fi
: "${OPENAI_BASE_URL:?set OPENAI_BASE_URL in config/env.sh}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY in config/env.sh}"
: "${PYRUALEAN_MODEL:?set PYRUALEAN_MODEL in config/env.sh}"

policy_dir="robodojo/XPolicyLab/policy/PyRUA_Lean"
[ -f "${policy_dir}/setup_eval_policy_server.sh" ] \
    || die "policy adapter missing; run scripts/fetch_sources.sh --policy-only first"
[ -s "${policy_dir}/checkpoints/sam3.pt" ] \
    || warn "SAM3 checkpoint not found; run scripts/fetch_assets.sh"

info "starting the policy server on ${bind_host}:${port} (task=${task})"
exec bash robodojo/scripts/robodojo.sh server \
    --policy-dir "${policy_dir}" \
    --task "${task}" \
    --ckpt no-checkpoint \
    --policy-env "${PYRUALEAN_POLICY_ENV:-pyrualean}" \
    --env-cfg litearm \
    --action-type joint \
    --seed "${seed}" \
    --policy-port "${port}" \
    --bind-host "${bind_host}" \
    "${extra[@]}"
