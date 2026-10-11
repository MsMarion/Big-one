#!/usr/bin/env bash
# Fetch the official SO-101 robot description (URDF + STL meshes) from TheRobotStudio/SO-ARM100 (Apache-2.0),
# pinned to one commit so the model never changes under us. ~19 MB, not kept in git.
#   bash robots/so101/fetch_description.sh
set -euo pipefail
COMMIT=5f6d2b876a53a4872e405b991dd925556c9e38a4
REPO=TheRobotStudio/SO-ARM100
SRC=Simulation/SO101
DEST="$(cd "$(dirname "$0")" && pwd)/description"
mkdir -p "$DEST/assets"
raw() { curl -fsSL "https://raw.githubusercontent.com/$REPO/$COMMIT/$SRC/$1" -o "$DEST/$1"; }
for f in so101_new_calib.urdf README.md joints_properties.xml; do raw "$f"; done
for f in $(curl -fsSL "https://api.github.com/repos/$REPO/contents/$SRC/assets?ref=$COMMIT" | grep -o '"name": *"[^"]*\.stl"' | sed 's/.*"\([^"]*\)"$/\1/'); do
  raw "assets/$f"
done
curl -fsSL "https://raw.githubusercontent.com/$REPO/$COMMIT/LICENSE" -o "$DEST/LICENSE"
echo "SO-101 description @ $COMMIT -> $DEST ($(ls "$DEST/assets" | wc -l) meshes)"
