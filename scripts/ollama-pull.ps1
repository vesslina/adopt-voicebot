param(
    [string]$Model = "qwen3:8b"
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not $env:DOCKER_CONFIG) {
    $env:DOCKER_CONFIG = Join-Path (Get-Location) ".docker-cli"
}
New-Item -ItemType Directory -Force -Path $env:DOCKER_CONFIG | Out-Null

Write-Host "Pulling Ollama model: $Model"
docker compose exec ollama ollama pull $Model

Write-Host ""
Write-Host "Available models:"
docker compose exec ollama ollama list
