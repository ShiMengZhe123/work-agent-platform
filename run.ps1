$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$env:PYTHONUTF8 = "1"
$env:PIP_CACHE_DIR = Join-Path $root ".cache\pip"
$env:TMP = Join-Path $root ".tmp"
$env:TEMP = Join-Path $root ".tmp"

if (-not (Test-Path (Join-Path $root ".venv\Scripts\python.exe"))) {
    throw "未找到 .venv，请先按 README 配置环境。"
}

& (Join-Path $root ".venv\Scripts\python.exe") -m streamlit run app.py --server.port 8501
