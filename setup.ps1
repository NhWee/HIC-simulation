param(
    [string]$PythonExe = "",
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Invoke-Checked {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Executable $($Arguments -join ' ')"
    }
}

function Resolve-PythonLauncher {
    if ($PythonExe) {
        $Requested = Get-Command $PythonExe -ErrorAction SilentlyContinue
        if ($Requested) {
            return @($Requested.Source)
        }
        if (Test-Path -LiteralPath $PythonExe) {
            return @((Resolve-Path -LiteralPath $PythonExe).Path)
        }
        throw "Python executable was not found: $PythonExe"
    }

    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        return @($PyLauncher.Source, "-3")
    }

    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) {
        return @($PythonCommand.Source)
    }

    throw "Python 3 was not found. Install Python 3.11 or newer, or pass -PythonExe."
}

$Launcher = @(Resolve-PythonLauncher)
$LauncherExe = $Launcher[0]
$LauncherPrefix = @($Launcher | Select-Object -Skip 1)

Push-Location $ProjectRoot
try {
    Invoke-Checked $LauncherExe ($LauncherPrefix + @(
        "-c",
        "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ is required'; print(sys.version)"
    ))

    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Invoke-Checked $LauncherExe ($LauncherPrefix + @("-m", "venv", ".venv"))
    }

    Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-Checked $VenvPython @("-m", "pip", "install", "-r", "requirements.txt")
    $TorchRequirements = if ($Cpu) { "requirements-cpu.txt" } else { "requirements-cuda.txt" }
    Invoke-Checked $VenvPython @("-m", "pip", "install", "-r", $TorchRequirements)
    New-Item -ItemType Directory -Force -Path "outputs" | Out-Null

    Invoke-Checked $VenvPython @(
        "-c",
        "import numpy, plotly, pyqtgraph, torch; import PySide6; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('numpy', numpy.__version__, 'plotly', plotly.__version__, 'PySide6', PySide6.__version__, 'pyqtgraph', pyqtgraph.__version__)"
    )
}
finally {
    Pop-Location
}
