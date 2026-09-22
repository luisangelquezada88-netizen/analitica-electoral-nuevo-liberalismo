# Dashboard Electoral — script de arranque (PowerShell)
# Uso: clic derecho > Ejecutar con PowerShell, o en terminal:
#   powershell -ExecutionPolicy Bypass -File .\run_dashboard.ps1
Set-Location -LiteralPath $PSScriptRoot
if (Test-Path -LiteralPath ".\.venv\Scripts\streamlit.exe") {
  & .\.venv\Scripts\streamlit.exe run src\dashboard_electoral.py --server.port 8501
} else {
  Write-Output "No se encontró .\.venv\Scripts\streamlit.exe — crea el venv e instala requirements.txt primero."
  exit 1
}
