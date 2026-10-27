#!/bin/bash
# Download Kaggle "asl-signs" (Google ISLR) landmark files for chosen signs.
#
# Run this YOURSELF in a terminal, not through an agent: it reads the Kaggle
# token from the macOS Keychain (via OrbStack's `mac` command) into this
# process only, so the token never enters an agent session, a file or history.
#
# One-time setup, in a Mac terminal (prompts for the token, hidden):
#     security add-generic-password -s kaggle-api -a "$USER" -w
#
# Usage (in the Linux VM):
#     pipeline/fetch_islr.sh                 # train.csv + sign map only
#     pipeline/fetch_islr.sh hello yes no    # plus every sequence of these signs
set -euo pipefail

OUT="$(cd "$(dirname "$0")" && pwd)/.cache/islr"
KAGGLE="${KAGGLE:-$HOME/.local/bin/kaggle}"
mkdir -p "$OUT"

KAGGLE_API_TOKEN="$(mac security find-generic-password -s kaggle-api -w)"
export KAGGLE_API_TOKEN  # this process and its children only

get() {  # get <file in competition>; skips files already downloaded
  local dest="$OUT/$1"
  [ -f "$dest" ] && return 0
  mkdir -p "$(dirname "$dest")"
  "$KAGGLE" competitions download asl-signs -f "$1" -p "$(dirname "$dest")" -q
  # Kaggle may deliver a single file zipped.
  if [ -f "$dest.zip" ]; then unzip -o -q "$dest.zip" -d "$(dirname "$dest")" && rm "$dest.zip"; fi
}

get train.csv
get sign_to_prediction_index_map.json

for sign in "$@"; do
  files=$(awk -F, -v s="$sign" 'NR > 1 && $4 == s { print $1 }' "$OUT/train.csv")
  [ -n "$files" ] || { echo "no sequences for sign '$sign'" >&2; exit 1; }
  echo "$sign: $(echo "$files" | wc -l) sequences"
  for f in $files; do get "$f"; done
done
echo "done: $OUT"
