#!/usr/bin/env bash
# Prepare a fresh git worktree: link or copy the gitignored files it needs from
# the main checkout, then build the venv. Safe to re-run.
set -euo pipefail

WORKTREE="$(git rev-parse --show-toplevel)"
MAIN="$(cd "$(git rev-parse --git-common-dir)/.." && pwd -P)"
cd "$WORKTREE"

if [ "$(pwd -P)" = "$MAIN" ]; then
  echo "Running in the main checkout; nothing to set up."
  exit 0
fi

# Large or read-only inputs: shared via symlink.
link() {
  local path="$1"
  if [ -e "$path" ] || [ -L "$path" ]; then
    echo "skip   $path (already present)"
  elif [ -e "$MAIN/$path" ]; then
    mkdir -p "$(dirname "$path")"
    ln -s "$MAIN/$path" "$path"
    echo "link   $path -> $MAIN/$path"
  else
    echo "warn   $path missing in main checkout"
  fi
}

# Per-worktree config: copied so branches can diverge.
copy() {
  local path="$1"
  if [ -e "$path" ]; then
    echo "skip   $path (already present)"
  elif [ -e "$MAIN/$path" ]; then
    cp "$MAIN/$path" "$path"
    echo "copy   $path"
  fi
}

link data
link docs/assignment
copy .env
copy .envrc

# Own venv per worktree; venvs hold absolute paths, so never share or copy them.
uv sync --locked

echo "Worktree ready: $WORKTREE"
