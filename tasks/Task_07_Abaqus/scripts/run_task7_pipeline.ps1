param(
    [switch]$BuildOnly,
    [switch]$SkipAbaqus,
    [Alias("SkipViewer")]
    [switch]$SkipPlots,
    [switch]$SkipReport
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..\\..")).Path
$TaskDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ResultsDir = Join-Path $TaskDir "results"
$JobsDir = Join-Path $ResultsDir "jobs"
$ManifestPath = Join-Path $ResultsDir "data\\job_manifest.json"
$SummaryCsv = Join-Path $ResultsDir "data\\abaqus_summary.csv"
$MeshCsv = Join-Path $ResultsDir "data\\mesh_sensitivity.csv"
$FieldsDir = Join-Path $ResultsDir "data\\fields"

Write-Host "Exporting Task 6 candidate geometries..."
python (Join-Path $PSScriptRoot "export_task6_candidates.py")

Write-Host "Building Abaqus input decks..."
python (Join-Path $PSScriptRoot "build_abaqus_inputs.py")

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
        }
        finally {
            Pop-Location
        }
    }

    Write-Host "Postprocessing ODB results..."
    abaqus python (Join-Path $PSScriptRoot "postprocess_odb.py") --manifest $ManifestPath --output-csv $SummaryCsv --mesh-csv $MeshCsv

}

if (-not $BuildOnly) {
    New-Item -ItemType Directory -Force $FieldsDir | Out-Null
    Write-Host "Exporting structured ODB field data..."
    abaqus python (Join-Path $PSScriptRoot "export_odb_fields.py") --manifest $ManifestPath --output-dir $FieldsDir

    if (-not $SkipPlots) {
        Write-Host "Rendering custom figures from exported Abaqus fields..."
        python (Join-Path $PSScriptRoot "plot_abaqus_fields.py")
    }
}

if (-not $BuildOnly -and -not $SkipReport) {
    Write-Host "Generating Task 7 report..."
    python (Join-Path $PSScriptRoot "generate_report.py")
}

Write-Host "Task 7 pipeline complete."
