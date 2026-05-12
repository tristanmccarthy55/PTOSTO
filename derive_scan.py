# Run with: C:\Users\Trist\HyperSpy-bundle\python.exe derive_scan.py
"""Analytical scan-design pass for the W2+W3 changes (no simulation).

Sweeps OVERFOCUS_NM x TARGET_OVERLAP for the planned 6x6 tiled grid and prints,
for each combination: effective probe FWHM, scan step, positions per tile, total
positions over the grid, raw on-disk CBED size, a rough recon-VRAM estimate, and
estimated sim / recon wall-clock. Also echoes the fixed depth-resolution budget
(dz ~ lambda / alpha^2) and the paper's "features must be separated by >= 2x dz"
rule against the ~7 nm polarisation tornado.

Use this to pick OVERFOCUS_NM and TARGET_OVERLAP before committing any
multi-hour run, then write the chosen values into params.py.
"""
from __future__ import annotations

import sys

import numpy as np

import params as P

# Placeholder box dims so we don't need abTEM / the POSCAR on disk. These match
# params.py's __main__ fallback (47.88 x 70.01 x 73.93 A; beam = +Z).
BOX_DIMS_A = (47.88, 70.01, 73.93)

# What we're planning to switch to.
N_TILES_TILED = 6          # 6x6 tiles of TILE_SIZE_A -> ~24x24 A scan window
TILE_SIZE_A = P.TILE_SIZE_A  # 4.0

OVERFOCUS_NM_SWEEP = [1.0, 2.0, 5.0, 10.0]
OVERLAP_SWEEP = [0.75, 0.85, 0.90, 0.95]

# Reference numbers for the current production run (overfocus 1 nm, overlap 0.75,
# 3x3 tiles) and the paper (~26x26 A, 64x64 = 4096 positions).
PAPER_POSITIONS = 64 * 64
PAPER_SCAN_A = 26.0

# Rough per-position sim cost (GPU), calibrated from the handover: ~15 min/tile
# at ~256 positions/tile (overfocus 1 nm, step ~0.25 A) -> ~3.5 s/position.
SEC_PER_POSITION_SIM = 15.0 * 60.0 / 256.0
# Recon Stage A+B together (~120 iter) scale ~linearly with position count;
# handover: production recon (~2300 pos, 74 slices, batch 64) ~= 30-40 min.
SEC_PER_POSITION_RECON = 35.0 * 60.0 / 2304.0

# CBED on disk: the binned (still non-square) pattern, ~65 x ~95 px float32,
# before zstd compression (compresses ~2-4x in practice; we report the raw size).
CBED_PX_Y, CBED_PX_X = 65, 95
BYTES_CBED = CBED_PX_Y * CBED_PX_X * 4


def positions_per_tile_axis(scan_step_a: float) -> int:
    """abTEM GridScan with `sampling=scan_step` on a tile of length TILE_SIZE_A
    yields ~round(extent / sampling) points along each axis (>= 1)."""
    return max(1, int(round(TILE_SIZE_A / scan_step_a)))


def recon_pixel_size_a(wavelength_a: float) -> float:
    """py4DSTEM real-space pixel ~= 1 / (2 k_max) = lambda / (2 * theta_max_rad)."""
    theta_max_rad = P.DETECTOR_MAX_ANGLE_MRAD * 1e-3
    return wavelength_a / (2.0 * theta_max_rad)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    wavelength_a = P.relativistic_wavelength_a(P.ENERGY_EV)
    alpha_rad = P.CONVERGENCE_MRAD * 1e-3
    probe_fwhm_focused_a = 0.61 * wavelength_a / alpha_rad
    dz_optical_a = wavelength_a / (alpha_rad ** 2)
    tornado_a = BOX_DIMS_A[2]  # ~74 A column along the beam

    recon_num_slices = int(round(BOX_DIMS_A[2] / 1.0))
    px_a = recon_pixel_size_a(wavelength_a)
    scan_extent_a = N_TILES_TILED * TILE_SIZE_A
    # Object grid ~= (scan extent + a couple of probe widths) / pixel size.
    pad_a = 8.0

    print("=" * 78)
    print("Scan-design analytical pass  (W2+W3 -- 6x6 tiles of "
          f"{TILE_SIZE_A:.1f} A = {scan_extent_a:.0f} x {scan_extent_a:.0f} A window)")
    print("=" * 78)
    print(f"  lambda            = {wavelength_a:.5f} A   (300 keV)")
    print(f"  alpha             = {P.CONVERGENCE_MRAD:.0f} mrad")
    print(f"  probe FWHM (foc.) = {probe_fwhm_focused_a:.3f} A")
    print(f"  dz_optical = l/a^2 = {dz_optical_a:.2f} A   "
          f"(features resolvable in depth if separated by >= {2*dz_optical_a:.1f} A)")
    print(f"  tornado length    = {tornado_a:.1f} A  =>  "
          f"~{tornado_a/(2*dz_optical_a):.0f} independent depth samples at best")
    print(f"  recon grid (est.) = {recon_num_slices} slices x "
          f"~{int((scan_extent_a+pad_a)/px_a)}^2 px  (px ~= {px_a:.3f} A)")
    print(f"  paper reference   = {PAPER_SCAN_A:.0f} x {PAPER_SCAN_A:.0f} A, "
          f"{PAPER_POSITIONS} positions")
    print()

    hdr = (f"{'ovf':>5} {'ovl':>5} | {'FWHMeff':>8} {'step':>7} | "
           f"{'pos/tile':>9} {'pos tot':>8} {'/paper':>7} | "
           f"{'CBED disk':>10} | {'obj VRAM':>9} | {'sim h':>7} {'recon h':>8}")
    print(hdr)
    print("-" * len(hdr))

    for ovf_nm in OVERFOCUS_NM_SWEEP:
        overfocus_a = ovf_nm * 10.0
        geom_spread_a = alpha_rad * overfocus_a
        fwhm_eff_a = float(np.hypot(probe_fwhm_focused_a, geom_spread_a))
        for ovl in OVERLAP_SWEEP:
            step_a = (1.0 - ovl) * fwhm_eff_a
            n_axis = positions_per_tile_axis(step_a)
            pos_tile = n_axis * n_axis
            pos_tot = pos_tile * N_TILES_TILED * N_TILES_TILED
            cbed_disk_gb = pos_tot * BYTES_CBED / 1e9
            obj_vram_gb = recon_num_slices * ((scan_extent_a + pad_a) / px_a) ** 2 * 8 / 1e9
            sim_h = pos_tot * SEC_PER_POSITION_SIM / 3600.0
            recon_h = pos_tot * SEC_PER_POSITION_RECON / 3600.0
            print(f"{ovf_nm:>5.1f} {ovl:>5.2f} | {fwhm_eff_a:>8.3f} {step_a:>7.3f} | "
                  f"{pos_tile:>9d} {pos_tot:>8d} {pos_tot/PAPER_POSITIONS:>6.2f}x | "
                  f"{cbed_disk_gb:>9.2f}G | {obj_vram_gb:>8.2f}G | "
                  f"{sim_h:>7.2f} {recon_h:>8.2f}")
        print("-" * len(hdr))

    print()
    print("Guidance: pick the (overfocus, overlap) with paper-like position count "
          "(~1x), CBED disk well under ~5 GB, object VRAM well under 8 GB, and "
          "sim+recon wall-clock you can live with. Bigger overfocus spreads the "
          "probe (more overlap diversity) but coarsens the step at fixed overlap; "
          "raise overlap to keep the position count up. Then set OVERFOCUS_NM and "
          "TARGET_OVERLAP in params.py to the chosen values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
