#!/usr/bin/env bash
# Create the policy-server conda environment (PyRUA-Lean + XPolicyLab + SAM3).
#
# Usage: bash scripts/setup_policy_env.sh [ENV_NAME]
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
cd "${REPO_ROOT}"

env_name="${1:-${PYRUALEAN_POLICY_ENV:-pyrualean}}"
torch_version="2.9.1"
torchvision_version="0.24.1"
adapter="${REPO_ROOT}/robodojo/XPolicyLab/policy/PyRUA_Lean"

if [ ! -f "${adapter}/install.sh" ]; then
    info "adapter link missing; cloning XPolicyLab first"
    bash "${REPO_ROOT}/scripts/fetch_sources.sh" --policy-only
fi

if ! command -v conda >/dev/null 2>&1; then
    die "conda not found. Install Miniconda, or run robodojo/scripts/install.sh -i on a simulator host."
fi
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    info "creating conda environment ${env_name} (Python 3.11)"
    conda create -n "${env_name}" python=3.11 -y
fi
conda activate "${env_name}"

info "installing the CUDA torch stack"
python -m pip install --upgrade pip
python -m pip install "torch==${torch_version}" "torchvision==${torchvision_version}" \
    --index-url https://download.pytorch.org/whl/cu128
python -m pip install "numpy==1.26.4"

if ! python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    die "torch cannot see a CUDA device in ${env_name}; check the NVIDIA driver"
fi

info "installing XPolicyLab, PyRUA-Lean and SAM3 into ${env_name}"
bash "${adapter}/install.sh" "${REPO_ROOT}/pyrua-lean"

python - <<'PY'
import torch
import pyrualean
from sam3.model_builder import build_sam3_image_model

print(f"[setup] torch {torch.__version__}, pyrualean {pyrualean.__version__}, SAM3 imports")
PY

cat <<EOF
[setup] policy environment ${env_name} is ready.
[setup] Next, download the SAM3 checkpoint:
[setup]   bash ${adapter}/download_assets.sh
[setup] facebook/sam3 is gated: accept its licence and run 'hf auth login' first.
EOF
