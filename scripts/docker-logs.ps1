$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not $env:DOCKER_CONFIG) {
    $env:DOCKER_CONFIG = Join-Path (Get-Location) ".docker-cli"
}
New-Item -ItemType Directory -Force -Path $env:DOCKER_CONFIG | Out-Null
docker compose logs -f voicebot
