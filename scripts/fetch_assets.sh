#!/usr/bin/env bash
# Download the RoboDojo asset dataset, install the LiteArm robot asset, and
# optionally fetch the SAM3 checkpoint.
#
# Usage: bash scripts/fetch_assets.sh [--skip-sam3]
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
cd "${REPO_ROOT}"

skip_sam3=0
policy_only=0
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-sam3) skip_sam3=1 ;;
        --policy-only) policy_only=1 ;;
        -h|--help)
            echo "Usage: bash scripts/fetch_assets.sh [--skip-sam3] [--policy-only]"
            exit 0
            ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done

require_cmd python3
adapter="${REPO_ROOT}/robodojo/XPolicyLab/policy/PyRUA_Lean"

if [ "${policy_only}" -eq 0 ]; then
    assets_ready=1
    for subdir in Robots Object Material Eval_Layout; do
        [ -d "robodojo/Assets/${subdir}" ] || assets_ready=0
    done
    if [ "${assets_ready}" -eq 0 ]; then
        require_cmd git
        git lfs version >/dev/null 2>&1 || die "git-lfs is required for the RoboDojo asset dataset"
        info "downloading the RoboDojo asset dataset (tens of GB)"
        bash robodojo/scripts/init_assets.sh
    fi

    info "installing the LiteArm robot asset"
    mkdir -p robodojo/Assets/Robots/litearm
    cp -a assets/litearm/. robodojo/Assets/Robots/litearm/

    # RoboDojo resolves ${ASSETS_PATH} in the *_tmp.yml templates at install time.
    info "generating the cuRobo config for this checkout"
    (cd robodojo && python3 utils/update_embodiment_config_path.py)

    [ -f robodojo/Assets/Robots/litearm/litearm.usd ] \
        || die "LiteArm USD asset is missing after installation"
    [ -f robodojo/Assets/Robots/litearm/curobo.yml ] \
        || die "cuRobo config was not generated"
fi

if [ "${skip_sam3}" -eq 0 ] && [ ! -s "${adapter}/checkpoints/sam3.pt" ]; then
    info "downloading the SAM3 checkpoint (3.4 GB, gated on Hugging Face)"
    if ! bash "${adapter}/download_assets.sh"; then
        die "SAM3 download failed. Accept the facebook/sam3 licence, run 'hf auth login', then retry."
    fi
fi

info "assets ready"
