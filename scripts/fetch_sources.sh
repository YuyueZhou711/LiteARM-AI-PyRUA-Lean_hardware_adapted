#!/usr/bin/env bash
# Clone the third-party checkouts RoboDojo needs and apply the LiteArm overlay.
#
# Usage:
#   bash scripts/fetch_sources.sh              # XPolicyLab, IsaacLab, cuRobo
#   bash scripts/fetch_sources.sh --policy-only  # XPolicyLab only (policy server host)
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
cd "${REPO_ROOT}"

usage() {
    cat <<'EOF'
Usage: bash scripts/fetch_sources.sh [--policy-only]

  --policy-only  Clone only XPolicyLab, which the policy server needs.
                 Use this on a machine that runs no Isaac Sim client.
EOF
}

policy_only=0
while [ $# -gt 0 ]; do
    case "$1" in
        --policy-only) policy_only=1 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done

require_cmd git

if [ "${policy_only}" -eq 0 ]; then
    clone_pinned https://github.com/yuechen0614/IsaacLab.git \
        robodojo/third_party/IsaacLab "${ISAACLAB_COMMIT}"
    clone_pinned https://github.com/yuechen0614/curobo.git \
        robodojo/third_party/curobo "${CUROBO_COMMIT}"
fi
clone_pinned https://github.com/XPolicyLab/XPolicyLab.git \
    robodojo/XPolicyLab "${XPOLICYLAB_COMMIT}"

info "applying the LiteArm XPolicyLab overlay"
mkdir -p robodojo/XPolicyLab/utils/robot robodojo/XPolicyLab/policy
cp -a integrations/xpolicylab/utils/robot/_robot_info.json robodojo/XPolicyLab/utils/robot/
cp -a integrations/xpolicylab/policy/. robodojo/XPolicyLab/policy/

if [ ! -e robodojo/XPolicyLab/policy/PyRUA_Lean ]; then
    ln -s ../../../pyrua-lean/integrations/xpolicylab/PyRUA_Lean \
        robodojo/XPolicyLab/policy/PyRUA_Lean
fi

[ -f robodojo/XPolicyLab/client_server/ws/model_client.py ] \
    || die "XPolicyLab checkout is incomplete"
[ -f robodojo/XPolicyLab/policy/PyRUA_Lean/setup_eval_policy_server.sh ] \
    || die "PyRUA-Lean adapter link is missing"

info "third-party sources ready"
