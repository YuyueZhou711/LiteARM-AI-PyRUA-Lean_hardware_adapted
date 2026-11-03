#!/usr/bin/env bash
# Run one LiteArm evaluation on a single machine (policy server and simulator
# client on localhost).
#
# Usage:
#   bash scripts/run_eval.sh [--task NAME] [--seed N] [--env-cfg litearm]
#                            [--eval-num N] [-- extra robodojo.sh options]
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
cd "${REPO_ROOT}"

task="stack_blocks"
seed="0"
env_cfg="litearm"
eval_num="1"
extra=()

while [ $# -gt 0 ]; do
    case "$1" in
        --task) task="${2:?--task needs a value}"; shift 2 ;;
        --seed) seed="${2:?--seed needs a value}"; shift 2 ;;
        --env-cfg) env_cfg="${2:?--env-cfg needs a value}"; shift 2 ;;
        --eval-num) eval_num="${2:?--eval-num needs a value}"; shift 2 ;;
        -h|--help)
            sed -n '2,8p' "$0"
            exit 0
            ;;
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

policy_env="${PYRUALEAN_POLICY_ENV:-pyrualean}"
eval_env="${ROBODOJO_EVAL_ENV:-RoboDojo}"
policy_dir="robodojo/XPolicyLab/policy/PyRUA_Lean"

[ -f "${policy_dir}/setup_eval_policy_server.sh" ] \
    || die "policy adapter missing; run scripts/fetch_sources.sh first"
[ -f robodojo/Assets/Robots/litearm/litearm.usd ] \
    || die "LiteArm assets missing; run scripts/fetch_assets.sh first"
[ -s "${policy_dir}/checkpoints/sam3.pt" ] \
    || warn "SAM3 checkpoint not found; run scripts/fetch_assets.sh"

info "task=${task} env_cfg=${env_cfg} seed=${seed} eval_num=${eval_num}"
exec bash robodojo/scripts/robodojo.sh eval \
    --policy-dir "${policy_dir}" \
    --task "${task}" \
    --ckpt no-checkpoint \
    --policy-env "${policy_env}" \
    --eval-env "${eval_env}" \
    --env-cfg "${env_cfg}" \
    --action-type joint \
    --seed "${seed}" \
    --eval-num "${eval_num}" \
    "${extra[@]}"
