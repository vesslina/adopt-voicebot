#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-qwen3:8b}"

cd "$(dirname "$0")/.."
docker compose exec ollama ollama pull "$MODEL"
docker compose exec ollama ollama list

