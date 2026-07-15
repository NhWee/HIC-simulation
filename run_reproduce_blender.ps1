param(
    [string]$BlenderPath = "",
    [switch]$Cpu,
    [switch]$SkipAnimation
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BlenderRunner = Join-Path $ProjectRoot "run_blender_hic_scene.ps1"
$ValidationRunner = Join-Path $ProjectRoot "run_hydro_validation.ps1"
$AuditPath = Join-Path $ProjectRoot "outputs\blender_physics_event.audit.json"

Push-Location $ProjectRoot
try {
    $ValidationArguments = @("--output", "outputs\hydro_validation.csv")
    if ($Cpu) {
        $ValidationArguments += "--cpu"
    }
    & $ValidationRunner @ValidationArguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $SceneArguments = @{
        ImpactParameter = 5.7
        Seed = 17
        GridSize = 160
        Particles = 900
    }
    if ($BlenderPath) {
        $SceneArguments["BlenderPath"] = $BlenderPath
    }
    if ($Cpu) {
        $SceneArguments["Cpu"] = $true
    }
    if (-not $SkipAnimation) {
        $SceneArguments["RenderAnimation"] = $true
    }

    & $BlenderRunner @SceneArguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    if (-not (Test-Path -LiteralPath $AuditPath)) {
        throw "Physics audit was not generated: $AuditPath"
    }
    $Audit = Get-Content -Raw -LiteralPath $AuditPath | ConvertFrom-Json
    if (-not $Audit.passed) {
        throw "Physics audit failed. Inspect $AuditPath"
    }

    Write-Host ""
    Write-Host "Reproduction completed and physics audit passed."
    Write-Host "Scene: outputs\hic_blender_physics_scene.blend"
    if (-not $SkipAnimation) {
        Write-Host "Video: outputs\hic_blender_physics_simulation.mp4"
    }
}
finally {
    Pop-Location
}
