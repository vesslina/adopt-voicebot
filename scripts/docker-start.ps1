$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not $env:DOCKER_CONFIG) {
    $env:DOCKER_CONFIG = Join-Path (Get-Location) ".docker-cli"
}
New-Item -ItemType Directory -Force -Path $env:DOCKER_CONFIG | Out-Null

Write-Host "Building and starting voicebot stack..."
docker compose up -d --build

Write-Host ""
Write-Host "Stack started. Current containers:"
docker compose ps

Write-Host ""
Write-Host "If this is the first run, pull the Ollama model:"
Write-Host "  .\scripts\ollama-pull.ps1 qwen3:8b"
