#!/usr/bin/env bash
# Install the LiteArm stack on one machine, or one side of a split setup.
#
# Usage:
#   bash scripts/setup.sh                 # simulator + policy environments
#   bash scripts/setup.sh --skip-sim      # policy server host only
#   bash scripts/setup.sh --skip-policy   # Isaac Sim client host only
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
cd "${REPO_ROOT}"

usage() {
    cat <<'EOF'
Usage: bash scripts/setup.sh [--skip-sim] [--skip-policy]

  --skip-sim     Skip Isaac Sim, IsaacLab and cuRobo (policy server host).
  --skip-policy  Skip the PyRUA-Lean/SAM3 environment (simulator host).

Prerequisites: Linux, an NVIDIA GPU with a current driver, git, and enough
disk space (about 40 GB for Isaac Sim and the RoboDojo assets, plus 4 GB for
the SAM3 checkpoint).
EOF
}

skip_sim=0
skip_policy=0
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-sim) skip_sim=1 ;;
        --skip-policy) skip_policy=1 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done

[ "${skip_sim}" -eq 1 ] && [ "${skip_policy}" -eq 1 ] && die "nothing to install"

if [ "${skip_sim}" -eq 1 ]; then
    bash scripts/fetch_sources.sh --policy-only
else
    bash scripts/fetch_sources.sh
fi

if [ "${skip_sim}" -eq 0 ]; then
    require_cmd wget
    info "installing the simulator environment (Isaac Sim 5.1, IsaacLab, cuRobo)"
    bash robodojo/scripts/install.sh -i
fi

if [ "${skip_policy}" -eq 0 ]; then
    bash scripts/setup_policy_env.sh
fi

cat <<'EOF'

[setup] Environment installation finished.

Next steps:
  1. Download assets:  bash scripts/fetch_assets.sh
  2. Configure the model endpoint:
       cp config/env.example.sh config/env.sh
       edit config/env.sh
  3. Run an evaluation:  bash scripts/run_eval.sh --task stack_blocks --seed 0

For a split setup, run scripts/run_server.sh on the policy host and
scripts/run_client.sh on the simulator host.
EOF
