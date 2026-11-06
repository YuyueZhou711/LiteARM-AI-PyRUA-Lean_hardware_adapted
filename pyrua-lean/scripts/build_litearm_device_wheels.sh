#!/usr/bin/env bash
# Build fixed, non-editable wheels for the two vendor SDKs.
#
# Inputs:
#   LITEARM_SDK_SRC  checkout of litearm-python at version 2.1.0
#   LITEGRIP_SDK_SRC checkout of litegrip-python at version 2.2.0 (source commit 6b5fdce)
#
# The LiteGrip repository uses semantic-release placeholders in pyproject.toml;
# this script copies the source to a temporary build tree and writes the
# distribution version that the verified install reports (2.2.0).  It does not
# modify the input checkout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/wheelhouse"
LITEARM_SRC="${LITEARM_SDK_SRC:-/home/hzh/Code/litearm-python}"
LITEGRIP_SRC="${LITEGRIP_SDK_SRC:-/home/hzh/Code/NEXTFORM/litegrip-python}"

if [[ ! -d "${LITEARM_SRC}/src/litearm" ]]; then
  echo "litearm-python source not found: ${LITEARM_SRC}" >&2
  exit 2
fi
if [[ ! -d "${LITEGRIP_SRC}/src/litegrip" ]]; then
  echo "litegrip-python source not found: ${LITEGRIP_SRC}" >&2
  echo "check it out from https://github.com/nexform-tech/litegrip-python at the verified commit" >&2
  exit 2
fi

mkdir -p "${OUT}"
build_one() {
  local src="$1" version="$2" name="$3"
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp}"' RETURN
  cp -a "${src}/." "${tmp}/"
  python3 - "${tmp}/pyproject.toml" "${version}" <<'PY'
import re, sys
path, version = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
text, count = re.subn(r'(?m)^version\s*=\s*"[^"]+"', f'version = "{version}"', text, count=1)
if count != 1:
    raise SystemExit(f"no static project.version in {path}")
open(path, "w", encoding="utf-8").write(text)
PY
  python3 -m pip wheel --no-deps --no-build-isolation -w "${OUT}" "${tmp}"
  echo "built ${name} ${version} into ${OUT}"
}

build_one "${LITEARM_SRC}" "2.1.0" "litearm-python"
build_one "${LITEGRIP_SRC}" "2.2.0" "litegrip-python"
ls -l "${OUT}"
