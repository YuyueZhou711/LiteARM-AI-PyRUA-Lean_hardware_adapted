#!/usr/bin/env bash
# Run the Isaac Sim client against a policy server started with run_server.sh.
#
# Usage:
#   POLICY_HOST=192.168.1.10 POLICY_PORT=18231 bash scripts/run_client.sh [--task NAME] [--seed N]
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
cd "${REPO_ROOT}"

task="${TASK:-stack_blocks}"
seed="${SEED:-0}"
host="${POLICY_HOST:?set POLICY_HOST to the policy server address}"
port="${POLICY_PORT:-18231}"
eval_num="${EVAL_NUM:-1}"
extra=()

while [ $# -gt 0 ]; do
    case "$1" in
        --task) task="${2:?--task needs a value}"; shift 2 ;;
        --seed) seed="${2:?--seed needs a value}"; shift 2 ;;
        --eval-num) eval_num="${2:?--eval-num needs a value}"; shift 2 ;;
        --) shift; extra=("$@"); break ;;
        *) extra+=("$1"); shift ;;
    esac
done

[ -f robodojo/Assets/Robots/litearm/litearm.usd ] \
    || die "LiteArm assets missing; run scripts/fetch_assets.sh first"

sim_env="${ROBODOJO_EVAL_ENV:-RoboDojo}"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${sim_env}" || die "cannot activate conda environment ${sim_env}"
else
    warn "conda not found; run this script in the ${sim_env} environment"
fi

info "connecting to ${host}:${port} (task=${task}, seed=${seed})"
exec bash robodojo/scripts/robodojo.sh client \
    --policy-dir robodojo/XPolicyLab/policy/PyRUA_Lean \
    --policy-host "${host}" \
    --policy-port "${port}" \
    --task "${task}" \
    --ckpt no-checkpoint \
    --env-cfg litearm \
    --action-type joint \
    --seed "${seed}" \
    --eval-num "${eval_num}" \
    "${extra[@]}"
