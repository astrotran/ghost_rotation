# Local Windows development bootstrap.
# Run from the repository root in PowerShell.

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"

Write-Host "Environment ready."
Write-Host "Run tests with: pytest"
Write-Host "Run API with: uvicorn api.main:app --reload"
