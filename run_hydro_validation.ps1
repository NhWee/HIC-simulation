$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project environment not found. Run .\setup.ps1 first."
}

Push-Location $ProjectRoot
try {
    & $Python -m hic.hydro_validation @args
    $PythonExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $PythonExitCode
