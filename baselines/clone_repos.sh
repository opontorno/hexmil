#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Clone the official baseline repositories at the exact commits used in HexMIL.
#
# The baseline models are imported *verbatim* from their authors' code (never
# re-implemented) to guarantee a fair comparison. Their repositories are not
# vendored in this project (see .gitignore: `baselines/git_repo/*`); this script
# fetches them and checks out the pinned commit for each.
#
# Usage:
#   bash baselines/clone_repos.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/git_repo"
mkdir -p "$DEST"
cd "$DEST"

# name|url|commit
REPOS=(
  "D3|https://github.com/BigAandSmallq/D3.git|14f21ad"
  "MVSS-Net|https://github.com/dong03/MVSS-Net.git|cc2aed7"
  "TruFor|https://github.com/grip-unina/TruFor.git|ae54475"
  "ManTraNet|https://github.com/ISICV/ManTraNet.git|59436db"
  "Deep_inpainting_localization|https://github.com/lihaod/Deep_inpainting_localization.git|d33468d"
)

for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url commit <<< "$entry"
  if [ -d "$name/.git" ]; then
    echo "[skip] $name already present"
    continue
  fi
  echo "[clone] $name ($commit)"
  git clone --quiet "$url" "$name"
  git -C "$name" checkout --quiet "$commit" || \
    echo "  [warn] could not checkout $commit for $name (upstream history may have changed)"
done

echo "Done. Baseline repositories are in: $DEST"
