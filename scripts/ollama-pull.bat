@echo off
if "%~1"=="" (
  powershell.exe -ExecutionPolicy Bypass -File "%~dp0ollama-pull.ps1" qwen3:8b
) else (
  powershell.exe -ExecutionPolicy Bypass -File "%~dp0ollama-pull.ps1" %*
)

