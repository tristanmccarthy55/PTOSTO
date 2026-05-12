# Full PTO/STO production pipeline -- unattended.
#
# Run from the PTOSTO directory:
#     powershell -ExecutionPolicy Bypass -File run_production.ps1
#     powershell -ExecutionPolicy Bypass -File run_production.ps1 -Room   # room-temp DWFs (W3 ablation)
# or just:
#     .\run_production.ps1
# Cryo (phonon σ ×0.65) is the DEFAULT; -Room reverts to room-temperature σ.
#
# Steps (W2 grid = 5x5 = 25 tiles, 5 nm overfocus; with the current
# TARGET_OVERLAP=0.90 that's ≈0.5 A step ≈ 64 pos/tile ≈ 1600 positions, total
# ~2 h on the user's GPU -- sim ~1.5 h, recon ~0.5 h. Raise TARGET_OVERLAP to
# 0.95 in params.py for the ~4× larger final run [≈6400 positions, ~7-9 h]):
#   1. simulate_4dstem.py --all --overwrite   (n_tiles_tiled^2 tiles, 8 phonon
#      configs, 1 A slices).  --overwrite is REQUIRED: any tile zarrs on disk are
#      from earlier (smaller-grid / mini) runs and MUST be regenerated with the
#      current params or the reconstruction would stitch incompatible data.
#   2. reconstruct_ptycho.py --stage both     (74 slices, batch 64, crop 64x64)
#   3. validate.py                            (production thresholds)
#
# Everything is logged to production_run_<timestamp>.log via Start-Transcript.
# If a step exits non-zero the run aborts and the log says where.

param(
    [switch]$Room
)

$py = "C:\Users\Trist\HyperSpy-bundle\python.exe"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "production_run_$stamp.log"
$dwfArgs = if ($Room) { @("--room") } else { @() }   # cryo σ is the default

Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
Start-Transcript -Path $log | Out-Null

Write-Host "================================================================"
Write-Host "PRODUCTION RUN START: $(Get-Date)"
Write-Host "Log file: $log"
Write-Host "================================================================"

Write-Host ""
Write-Host "--- [1/3] Simulating tile grid  (simulate_4dstem.py --all --overwrite $dwfArgs) ---"
$t0 = Get-Date
& $py simulate_4dstem.py --all --overwrite @dwfArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "!!! SIMULATION FAILED (exit code $LASTEXITCODE) after $((Get-Date) - $t0). Aborting."
    Stop-Transcript | Out-Null
    exit 1
}
Write-Host "--- Simulation done in $((Get-Date) - $t0) ---"

Write-Host ""
Write-Host "--- [2/3] Reconstruction  (reconstruct_ptycho.py --stage both) ---"
$t0 = Get-Date
& $py reconstruct_ptycho.py --stage both @dwfArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "!!! RECONSTRUCTION FAILED (exit code $LASTEXITCODE) after $((Get-Date) - $t0). Aborting."
    Write-Host "    (tile zarrs are intact; rerun reconstruct_ptycho.py --stage both once fixed)"
    Stop-Transcript | Out-Null
    exit 1
}
Write-Host "--- Reconstruction done in $((Get-Date) - $t0) ---"

Write-Host ""
Write-Host "--- [3/3] Validation  (validate.py) ---"
& $py validate.py

Write-Host ""
Write-Host "================================================================"
Write-Host "PRODUCTION RUN COMPLETE: $(Get-Date)"
Write-Host "Outputs:"
Write-Host "  ptycho_recon.zarr            -- object_complex, object_phase, probe"
Write-Host "  ptycho_recon_metadata.json   -- run parameters"
Write-Host "  validation_plots/            -- per-slice std, kz-FRC, XZ slice"
Write-Host "  $log                         -- this run's full log"
Write-Host "================================================================"
Stop-Transcript | Out-Null
