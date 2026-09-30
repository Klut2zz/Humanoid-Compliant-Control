#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"

ROBOT_HOST="${ROBOT_HOST:-unitree@192.168.123.164}"
TARGET_DIR="${TARGET_DIR:-/home/unitree/softmimic_release}"

echo "Syncing project from $PROJECT_ROOT to $ROBOT_HOST:$TARGET_DIR"
rsync -az --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'wandb' \
  --exclude 'logs' \
  --exclude 'outputs' \
  "$PROJECT_ROOT/" "$ROBOT_HOST:$TARGET_DIR"
