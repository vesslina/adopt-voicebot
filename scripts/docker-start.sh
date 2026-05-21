#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
docker compose up -d --build
docker compose ps
echo
echo "If this is the first run, pull the Ollama model:"
echo "  ./scripts/ollama-pull.sh qwen3:8b"

