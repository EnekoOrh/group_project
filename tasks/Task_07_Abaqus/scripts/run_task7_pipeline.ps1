param(
    [switch]$BuildOnly,
    [switch]$SkipAbaqus,
    [switch]$SkipCae,
    [Alias("SkipViewer")]
    [switch]$SkipPlots,
    [switch]$SkipReport,
    [switch]$SkipLatex
)

$ErrorActionPreference = "Stop"

function Assert-LastExitCode {
    param([string]$StepName)
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

function Assert-NoAbaqusProcesses {
    $running = Get-Process | Where-Object {
        $_.ProcessName -match '^(ABQ|abq)' -or $_.ProcessName -in @('standard', 'explicit', 'pre', 'package')
    }
    if ($running) {
        $names = ($running | Select-Object -ExpandProperty ProcessName -Unique) -join ", "
        throw "Abaqus processes are still running ($names). Close all Abaqus sessions before continuing."
    }
}

function Quarantine-CaeArtifacts {
    param(
        [string]$CaeDir,
        [string]$TimestampTag
    )

    $patterns = @(
        "Task7_Montresor_WindStudy_staged.cae",
        "Task7_Montresor_WindStudy_candidate.cae",
        "ABQcae*.exception"
    )

    $items = foreach ($pattern in $patterns) {
        Get-ChildItem -Path $CaeDir -Filter $pattern -ErrorAction SilentlyContinue
    }
    if (-not $items) {
        return $null
    }

    $quarantineDir = Join-Path $CaeDir ("quarantine\\" + $TimestampTag)
    New-Item -ItemType Directory -Force $quarantineDir | Out-Null
    foreach ($item in $items) {
        Move-Item -LiteralPath $item.FullName -Destination (Join-Path $quarantineDir $item.Name) -Force
    }
    return $quarantineDir
}

function Invoke-AbaqusNoGui {
    param(
        [string]$ScriptPath,
        [string[]]$PythonArgs
    )

    $escapedArgs = ($PythonArgs | ForEach-Object {
        $value = $_ -replace '\\', '\\\\'
        $value = $value -replace "'", "\\'"
        "'" + $value + "'"
    }) -join ", "
    $wrapperPath = Join-Path $env:TEMP ("task7_wrapper_" + [guid]::NewGuid().ToString("N") + ".py")
@"
import runpy, sys
sys.argv = [$escapedArgs]
runpy.run_path(r'$ScriptPath', run_name='__main__')
"@ | Set-Content -Path $wrapperPath -Encoding Ascii

    abaqus cae "noGUI=$wrapperPath"
    Assert-LastExitCode "Abaqus noGUI run for $ScriptPath"
}

function Assert-CaeAuditPass {
    param([string]$AuditJsonPath)
    if (-not (Test-Path $AuditJsonPath)) {
        throw "Missing CAE integrity audit report: $AuditJsonPath"
    }
    $audit = Get-Content -Raw $AuditJsonPath | ConvertFrom-Json
    if ($audit.status -ne "pass") {
        throw "CAE integrity audit failed. See $AuditJsonPath for details."
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..\\..")).Path
$TaskDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ResultsDir = Join-Path $TaskDir "results"
$JobsDir = Join-Path $ResultsDir "jobs"
$CaeDir = Join-Path $ResultsDir "cae"
$TimestampTag = Get-Date -Format "yyyyMMdd-HHmmss"

$CaeCanonicalBase = Join-Path $CaeDir "Task7_Montresor_WindStudy"
$CaeCanonicalPath = "$CaeCanonicalBase.cae"
$CaeCandidateBase = Join-Path $CaeDir "Task7_Montresor_WindStudy_candidate"
$CaeCandidatePath = "$CaeCandidateBase.cae"
$CaeLastGoodPath = Join-Path $CaeDir "Task7_Montresor_WindStudy_last_good.cae"
$CaeLastGoodStampedPath = Join-Path $CaeDir ("Task7_Montresor_WindStudy_last_good_" + $TimestampTag + ".cae")

$ConfigPath = Join-Path $TaskDir "config\\study_config.json"
$CasesPath = Join-Path $TaskDir "candidates\\selected_cases.json"
$ManifestPath = Join-Path $ResultsDir "data\\job_manifest.json"
$InputAuditCsv = Join-Path $ResultsDir "data\\input_contract_audit.csv"
$UnitAuditCsv = Join-Path $ResultsDir "data\\unit_load_contract_audit.csv"
$CaeAuditCsv = Join-Path $ResultsDir "data\\cae_integrity_audit.csv"
$CaeAuditJson = Join-Path $ResultsDir "data\\cae_integrity_audit.json"
$SummaryCsv = Join-Path $ResultsDir "data\\abaqus_summary.csv"
$MeshCsv = Join-Path $ResultsDir "data\\mesh_sensitivity.csv"
$FieldsDir = Join-Path $ResultsDir "data\\fields"
$LatexDir = Join-Path $TaskDir "report_latex"

$CaeBuilder = (Resolve-Path (Join-Path $PSScriptRoot "build_cae_study.py")).Path
$CaeValidator = (Resolve-Path (Join-Path $PSScriptRoot "validate_cae_integrity.py")).Path
$AsciiPdfPath = Join-Path $TaskDir "Task7_Montresor_CoolingTowerWind.pdf"
$FinalPdfPath = Join-Path $TaskDir "Task7_MontrÃ©sor_CoolingTowerWind.pdf"

Write-Host "Exporting Task 6 candidate geometries..."
python (Join-Path $PSScriptRoot "export_task6_candidates.py")
Assert-LastExitCode "Task 6 candidate export"

Write-Host "Building Abaqus input decks..."
python (Join-Path $PSScriptRoot "build_abaqus_inputs.py")
Assert-LastExitCode "Abaqus input build"

Write-Host "Validating generated input deck contracts..."
python (Join-Path $PSScriptRoot "validate_input_contracts.py") --manifest $ManifestPath --config $ConfigPath --output-csv $InputAuditCsv --unit-output-csv $UnitAuditCsv
Assert-LastExitCode "Abaqus input contract validation"

if (-not $SkipCae) {
    Assert-NoAbaqusProcesses
    New-Item -ItemType Directory -Force $CaeDir | Out-Null

    if (Test-Path $CaeCanonicalPath) {
        Copy-Item -LiteralPath $CaeCanonicalPath -Destination $CaeLastGoodPath -Force
        Copy-Item -LiteralPath $CaeCanonicalPath -Destination $CaeLastGoodStampedPath -Force
        Write-Host "Preserved canonical CAE snapshots:"
        Write-Host " - $CaeLastGoodPath"
        Write-Host " - $CaeLastGoodStampedPath"
    }

    $quarantineDir = Quarantine-CaeArtifacts -CaeDir $CaeDir -TimestampTag $TimestampTag
    if ($quarantineDir) {
        Write-Host "Quarantined prior staged/corrupt artifacts to $quarantineDir"
    }

    if (Test-Path $CaeCandidatePath) {
        Remove-Item -LiteralPath $CaeCandidatePath -Force
    }

    Write-Host "Building CAE candidate..."
    Invoke-AbaqusNoGui -ScriptPath $CaeBuilder -PythonArgs @(
        "build_cae_study.py",
        "--config", (Resolve-Path $ConfigPath).Path,
        "--cases", (Resolve-Path $CasesPath).Path,
        "--output-cae", $CaeCandidateBase
    )

    Write-Host "Validating CAE candidate integrity..."
    Invoke-AbaqusNoGui -ScriptPath $CaeValidator -PythonArgs @(
        "validate_cae_integrity.py",
        "--cae-path", $CaeCandidatePath,
        "--cases", (Resolve-Path $CasesPath).Path,
        "--output-csv", $CaeAuditCsv,
        "--output-json", $CaeAuditJson
    )
    Assert-CaeAuditPass -AuditJsonPath $CaeAuditJson

    $newExceptions = Get-ChildItem -Path $CaeDir -Filter "ABQcae*.exception" -ErrorAction SilentlyContinue
    if ($newExceptions) {
        throw "New ABQcae*.exception files were generated during candidate validation. Recovery aborted."
    }

    Write-Host "Promoting validated candidate to canonical CAE..."
    if (Test-Path $CaeCanonicalPath) {
        Remove-Item -LiteralPath $CaeCanonicalPath -Force
    }
    Move-Item -LiteralPath $CaeCandidatePath -Destination $CaeCanonicalPath -Force

    Write-Host "Re-validating canonical CAE after promotion..."
    Invoke-AbaqusNoGui -ScriptPath $CaeValidator -PythonArgs @(
        "validate_cae_integrity.py",
        "--cae-path", $CaeCanonicalPath,
        "--cases", (Resolve-Path $CasesPath).Path,
        "--output-csv", $CaeAuditCsv,
        "--output-json", $CaeAuditJson
    )
    Assert-CaeAuditPass -AuditJsonPath $CaeAuditJson
}

if (-not $BuildOnly -and -not $SkipAbaqus) {
    $manifest = Get-Content $ManifestPath | ConvertFrom-Json
    New-Item -ItemType Directory -Force $JobsDir | Out-Null

    foreach ($case in $manifest.cases) {
        $jobName = $case.job_name
        $inputPath = "..\\inputs\\$jobName.inp"
        Write-Host "Running Abaqus job $jobName ..."
        Push-Location $JobsDir
        try {
            abaqus job=$jobName input=$inputPath interactive
            Assert-LastExitCode "Abaqus job $jobName"
        }
        finally {
            Pop-Location
        }
    }

    $refinedCases = $manifest.cases | Where-Object { $_.mesh_level -eq "refined" }
    foreach ($case in $refinedCases) {
        $refinedOdb = Join-Path $JobsDir ($case.job_name + ".odb")
        if (-not (Test-Path $refinedOdb)) {
            throw "Refined sweep incomplete. Missing ODB: $refinedOdb"
        }
    }
    Write-Host "Validated full refined sweep: $($refinedCases.Count) refined ODB files present."

    Write-Host "Postprocessing ODB results..."
    abaqus python (Join-Path $PSScriptRoot "postprocess_odb.py") --manifest $ManifestPath --output-csv $SummaryCsv --mesh-csv $MeshCsv
    Assert-LastExitCode "ODB postprocessing"
}

if (-not $BuildOnly) {
    New-Item -ItemType Directory -Force $FieldsDir | Out-Null
    Write-Host "Exporting structured ODB field data..."
    abaqus python (Join-Path $PSScriptRoot "export_odb_fields.py") --manifest $ManifestPath --output-dir $FieldsDir
    Assert-LastExitCode "ODB field export"

    if (-not $SkipPlots) {
        Write-Host "Rendering custom figures from exported Abaqus fields..."
        python (Join-Path $PSScriptRoot "plot_abaqus_fields.py")
        Assert-LastExitCode "Field plotting"
    }
}

if (-not $BuildOnly -and -not $SkipReport) {
    Write-Host "Generating Task 7 report..."
    python (Join-Path $PSScriptRoot "generate_report.py")
    Assert-LastExitCode "Markdown report generation"

    if (-not $SkipLatex) {
        Write-Host "Syncing markdown report into LaTeX..."
        python (Join-Path $LatexDir "scripts\\sync_report_md_to_tex.py")
        Assert-LastExitCode "Markdown to LaTeX sync"

        Write-Host "Compiling Task 7 PDF report..."
        Push-Location $LatexDir
        try {
            latexmk -pdf -interaction=nonstopmode main.tex
            Assert-LastExitCode "Task 7 PDF compilation"
        }
        finally {
            Pop-Location
        }

        if (Test-Path $AsciiPdfPath) {
            @"
from pathlib import Path
src = Path(r"$AsciiPdfPath")
dst = Path(r"$TaskDir") / "Task7_Montr\u00e9sor_CoolingTowerWind.pdf"
if src.exists():
    for candidate in Path(r"$TaskDir").glob("Task7_Montr*CoolingTowerWind.pdf"):
        if candidate != src:
            candidate.unlink(missing_ok=True)
    if dst.exists():
        dst.unlink()
    src.rename(dst)
"@ | python -
        }
    }
}

Write-Host "Task 7 pipeline complete."
