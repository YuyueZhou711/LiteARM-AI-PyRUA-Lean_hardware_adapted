#!/usr/bin/env bash
# Shared paths, pinned revisions, and logging helpers for the repository scripts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Third-party revisions recorded by the RoboDojo base commit (f57e163).
ISAACLAB_COMMIT="afca7b09d60d8beb9c1cb28b43066499940b969b"
CUROBO_COMMIT="d17b54ce32cba095c0b000c4c58777075d11de0e"
XPOLICYLAB_COMMIT="5d0be1499634d3176258c8f2e31f5fe816497b0a"

info() { printf '[setup] %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*" >&2; }
die() { printf '[error] %s\n' "$*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }

clone_pinned() {
    local url=$1 dest=$2 commit=$3
    if [ -d "${dest}/.git" ]; then
        info "${dest} already cloned"
        return 0
    fi
    if [ -e "${dest}" ]; then
        die "${dest} exists but is not a git checkout; move it away and retry"
    fi
    mkdir -p "$(dirname "${dest}")"
    info "cloning ${dest} at ${commit:0:12}"
    git clone --filter=blob:none "${url}" "${dest}"
    git -C "${dest}" checkout "${commit}"
}
