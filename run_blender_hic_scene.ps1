param(
    [string]$BlenderPath = "",
    [string]$EventCache = "outputs\blender_physics_event.npz",
    [double]$ImpactParameter = 5.7,
    [int]$Seed = 17,
    [int]$GridSize = 160,
    [int]$Particles = 900,
    [switch]$ReuseCache,
    [switch]$Cpu,
    [switch]$Open,
    [switch]$RenderAnimation,
    [switch]$NoPreviews
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SceneOutput = "outputs\hic_blender_physics_scene.blend"

function Resolve-Blender {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        if (Test-Path -LiteralPath $RequestedPath) {
            return (Resolve-Path -LiteralPath $RequestedPath).Path
        }
        throw "BlenderPath was provided but does not exist: $RequestedPath"
    }

    if ($env:BLENDER_EXE -and (Test-Path -LiteralPath $env:BLENDER_EXE)) {
        return (Resolve-Path -LiteralPath $env:BLENDER_EXE).Path
    }

    $Command = Get-Command blender -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }

    $Candidates = @(
        "$env:ProgramFiles\Blender Foundation\Blender*\blender.exe",
        "${env:ProgramFiles(x86)}\Blender Foundation\Blender*\blender.exe",
        "$env:LOCALAPPDATA\Programs\Blender Foundation\Blender*\blender.exe",
        "$env:LOCALAPPDATA\Microsoft\WindowsApps\blender.exe"
    )

    foreach ($Pattern in $Candidates) {
        if (-not $Pattern) {
            continue
        }
        $Matches = Get-ChildItem -Path $Pattern -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending
        if ($Matches) {
            return $Matches[0].FullName
        }
    }

    return $null
}

$BlenderExe = Resolve-Blender -RequestedPath $BlenderPath

if (-not $BlenderExe) {
    throw @"
Blender was not found.

Install it with:
  winget install --id BlenderFoundation.Blender -e

Then run:
  .\run_blender_hic_scene.ps1

If Blender is already installed somewhere else, run:
  .\run_blender_hic_scene.ps1 -BlenderPath "C:\Path\To\blender.exe"

Or set:
  `$env:BLENDER_EXE = "C:\Path\To\blender.exe"
"@
}

Push-Location $ProjectRoot
try {
    if (-not $ReuseCache) {
        if (-not (Test-Path -LiteralPath $Python)) {
            throw "Project environment not found. Run .\setup.ps1 first."
        }
        $SimulationArguments = @(
            "-m", "hic.blender_event_pipeline",
            "--output", $EventCache,
            "--impact-parameter", $ImpactParameter,
            "--seed", $Seed,
            "--grid-size", $GridSize,
            "--particles", $Particles
        )
        if ($Cpu) {
            $SimulationArguments += "--cpu"
        }
        & $Python @SimulationArguments
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    elseif (-not (Test-Path -LiteralPath $EventCache)) {
        throw "Event cache was not found: $EventCache"
    }

    $BlenderArguments = @("--background", "--python-exit-code", "1", "--python", "tools\blender_hic_scene.py")
    $SceneArguments = @("--cache", $EventCache, "--output", $SceneOutput)
    if ($RenderAnimation) {
        $SceneArguments += "--render-animation"
    }
    if ($NoPreviews) {
        $SceneArguments += "--no-previews"
    }
    $BlenderArguments += "--"
    $BlenderArguments += $SceneArguments

    & $BlenderExe @BlenderArguments
    $BlenderExitCode = $LASTEXITCODE
    if ($BlenderExitCode -ne 0) {
        exit $BlenderExitCode
    }
    if ($Open) {
        & $BlenderExe $SceneOutput
    }
    else {
        Write-Host ""
        Write-Host "Created $SceneOutput"
        Write-Host "Physics cache: $EventCache"
        Write-Host "Open it with:"
        Write-Host "  .\run_blender_hic_scene.ps1 -Open"
        Write-Host "or:"
        Write-Host "  & `"$BlenderExe`" $SceneOutput"
        Write-Host ""
        Write-Host "Render the full 13-second MP4 without recalculating the event:"
        Write-Host "  .\run_blender_hic_scene.ps1 -ReuseCache -RenderAnimation"
    }
}
finally {
    Pop-Location
}

exit $BlenderExitCode
