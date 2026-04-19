# SENTINEL - Start Script
# Activates the Python virtual environment and launches the Flask API

Write-Host ""
Write-Host "  ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗     " -ForegroundColor Cyan
Write-Host "  ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║     " -ForegroundColor Cyan
Write-Host "  ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║     " -ForegroundColor Cyan
Write-Host "  ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║     " -ForegroundColor Cyan
Write-Host "  ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗ " -ForegroundColor Cyan
Write-Host "  ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  NLP-based Crime & Threat Detection System" -ForegroundColor White
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Activate virtual environment
$venvActivate = Join-Path $scriptDir "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "[*] Activating virtual environment..." -ForegroundColor Yellow
    & $venvActivate
} else {
    Write-Host "[!] Virtual environment not found at $venvActivate" -ForegroundColor Red
    Write-Host "[!] Run: python -m venv venv && venv\Scripts\Activate.ps1 && pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}

# Install/upgrade dependencies
Write-Host "[*] Checking dependencies..." -ForegroundColor Yellow
pip install -r (Join-Path $scriptDir "requirements.txt") -q

Write-Host ""
Write-Host "[*] Starting SENTINEL API server..." -ForegroundColor Green
Write-Host "[*] Model will load from cache (trained_model.pkl) — ready in a few seconds" -ForegroundColor Yellow
Write-Host ""
Write-Host "  App URL:  http://localhost:5000        (frontend + API unified)" -ForegroundColor Cyan
Write-Host "  API only: http://localhost:5000/api/status" -ForegroundColor DarkCyan
Write-Host ""
Write-Host "  Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Auto-open browser after a short delay (let Flask start first)
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 3
    Start-Process "http://localhost:5000"
} | Out-Null

# Launch Flask
python (Join-Path $scriptDir "app.py")
