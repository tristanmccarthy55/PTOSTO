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

# Base (untiled) cell dimensions — 47.88 x 70.01 x 73.93 A, beam = +Z.
# Kept as a constant so this script stays lightweight (no abTEM/POSCAR read).
# Final box = (STO_X_PAD widening) -> TILE_X -> TILE_Y. TILE_Z is forbidden.
_BASE_BOX_DIMS_A = (47.88, 70.01, 73.93)
_A_PTO = _BASE_BOX_DIMS_A[0] / 12.0   # POSCAR is 12 perovskite UCs along X
_PADDED_BX = _BASE_BOX_DIMS_A[0] + 2.0 * P.STO_X_PAD_UC * _A_PTO
BOX_DIMS_A = (
    _PADDED_BX * P.TILE_X,
    _BASE_BOX_DIMS_A[1] * P.TILE_Y,
    _BASE_BOX_DIMS_A[2],
)

# Mirrors params.py (kept here so the sweep below is self-contained).
N_TILES_TILED = P.N_TILES_TILED   # 5 -> 5x5 tiles of TILE_SIZE_A -> ~20x20 A window
TILE_SIZE_A = P.TILE_SIZE_A       # 4.0
TILE_X = P.TILE_X
TILE_Y = P.TILE_Y

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
    print(f"Scan-design analytical pass  (W2+W3 -- {N_TILES_TILED}x{N_TILES_TILED} "
          f"tiles of {TILE_SIZE_A:.1f} A = {scan_extent_a:.0f} x {scan_extent_a:.0f} A window)")
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

    # 'ovl_tgt' = the TARGET_OVERLAP knob (fraction of effective FWHM used to set
    # the step). 'ovl_lin' = the *realized* linear overlap of adjacent probe
    # footprints, footprint Ø = 2*alpha*defocus (the geometric shadow disk).
    # The realized overlap is what matters for ptychographic reconstructability
    # (rule of thumb: keep it >= ~60-70%; the reference paper ran ~99%).
    hdr = (f"{'ovf':>5} {'ovl_tgt':>7} | {'FWHMeff':>8} {'ftprtØ':>7} {'step':>7} "
           f"{'ovl_lin':>8} | {'pos/tile':>9} {'pos tot':>8} {'/paper':>7} | "
           f"{'CBED disk':>10} | {'obj VRAM':>9} | {'sim h':>7} {'recon h':>8}")
    print(hdr)
    print("-" * len(hdr))

    for ovf_nm in OVERFOCUS_NM_SWEEP:
        overfocus_a = ovf_nm * 10.0
        geom_spread_a = alpha_rad * overfocus_a            # disk radius
        footprint_a = max(probe_fwhm_focused_a, 2.0 * geom_spread_a)  # disk diameter
        fwhm_eff_a = float(np.hypot(probe_fwhm_focused_a, geom_spread_a))
        for ovl in OVERLAP_SWEEP:
            step_a = (1.0 - ovl) * fwhm_eff_a
            ovl_lin = max(0.0, 1.0 - step_a / footprint_a)
            n_axis = positions_per_tile_axis(step_a)
            pos_tile = n_axis * n_axis
            pos_tot = pos_tile * N_TILES_TILED * N_TILES_TILED
            cbed_disk_gb = pos_tot * BYTES_CBED / 1e9
            obj_vram_gb = recon_num_slices * ((scan_extent_a + pad_a) / px_a) ** 2 * 8 / 1e9
            sim_h = pos_tot * SEC_PER_POSITION_SIM / 3600.0
            recon_h = pos_tot * SEC_PER_POSITION_RECON / 3600.0
            print(f"{ovf_nm:>5.1f} {ovl:>7.2f} | {fwhm_eff_a:>8.3f} {footprint_a:>7.2f} "
                  f"{step_a:>7.3f} {ovl_lin*100:>7.1f}% | "
                  f"{pos_tile:>9d} {pos_tot:>8d} {pos_tot/PAPER_POSITIONS:>6.2f}x | "
                  f"{cbed_disk_gb:>9.2f}G | {obj_vram_gb:>8.2f}G | "
                  f"{sim_h:>7.2f} {recon_h:>8.2f}")
        print("-" * len(hdr))

    # ------------------------------------------------------------------
    # Probe-vs-cell clearance check.
    #
    # An OVERFOCUSED probe has its crossover Δf ABOVE the entrance surface, so
    # inside the sample it only ever diverges:  radius(z) = α·(Δf + z), with
    # z = 0 at entrance and z = t at exit.  abTEM uses a periodic cell, so if
    # the probe (at the worst-case scan corner, i.e. half the scan extent off
    # the centre) has appreciable amplitude at a cell edge it WRAPS AROUND and
    # corrupts the diffraction pattern.  The cell short axis (x ≈ 48 Å) is the
    # binding constraint.  This is also why the reference paper could use ~20 nm
    # overfocus and we can't: their α was ~5× smaller, so 2·α·Δf was ~5× smaller.
    cx, cy = P.CENTER_X_A, P.CENTER_Y_A
    bx, by, t = BOX_DIMS_A
    half_scan = scan_extent_a / 2.0
    print()
    print("Probe-vs-cell clearance  (overfocused: probe widest at the EXIT surface)")
    print(f"  TILE_X={TILE_X}, TILE_Y={TILE_Y}  =>  "
          f"cell (x×y) = {bx:.1f} × {by:.1f} Å ; thickness t = {t:.1f} Å ; "
          f"scan centred at ({cx:.1f}, {cy:.1f}) Å ; scan box = "
          f"[{cx-half_scan:.1f},{cx+half_scan:.1f}] × [{cy-half_scan:.1f},{cy+half_scan:.1f}] Å")
    if (TILE_X != 1 or TILE_Y != 1) and (cx > bx or cy > by):
        print(f"  [note] scan centre ({cx:.1f},{cy:.1f}) is OUTSIDE the tiled cell "
              f"({bx:.1f}×{by:.1f}) — update CENTER_X_A / CENTER_Y_A to a tiled mid-cell")
    ghdr = (f"  {'ovf nm':>6} | {'probeØ in':>10} {'probeØ out':>11} | "
            f"{'reach@out':>10} | {'overhang(ax)':>13} | {'%of probeØ':>11} | verdict")
    print(ghdr)
    print("  " + "-" * (len(ghdr) - 2))
    for ovf_nm in OVERFOCUS_NM_SWEEP + [15.0, 20.0]:
        df_a = ovf_nm * 10.0
        r_in = alpha_rad * df_a                    # entrance probe radius
        r_out = alpha_rad * (df_a + t)             # exit probe radius (worst case)
        reach = half_scan + r_out                  # worst-case lateral reach from centre
        x_lo, x_hi = cx - reach, cx + reach
        y_lo, y_hi = cy - reach, cy + reach
        x_over = max(0.0, 0.0 - x_lo, x_hi - bx)
        y_over = max(0.0, 0.0 - y_lo, y_hi - by)
        overhang = max(x_over, y_over)
        bound_axis = 'y' if y_over >= x_over else 'x'
        frac = overhang / max(2 * r_out, 1e-9)      # as a fraction of the probe diameter
        verdict = ("OK — fits" if overhang <= 0.0 else
                   "OK — soft tail clipped" if frac < 0.10 else
                   "MARGINAL — aliasing likely" if frac < 0.25 else
                   "UNSAFE — probe wraps")
        print(f"  {ovf_nm:>6.1f} | {2*r_in:>9.1f}Å {2*r_out:>10.1f}Å | "
              f"{reach:>9.1f}Å | {overhang:>9.1f}Å ({bound_axis}) | {frac*100:>10.0f}% | {verdict}")
    print("  Reading it: 'reach' = worst-case (scan-corner) lateral distance from "
          "the scan centre to the probe edge at the EXIT surface; 'overhang' = "
          "how far that pokes past the periodic cell along whichever axis (x or "
          "y) is binding — the tag in parens names the offender. A few % of the "
          "probe diameter is a low-amplitude tail (tolerable); >~25% means real "
          "aliasing/wraparound. The EXIT probe radius α·(Δf+t) has a fixed α·t "
          "≈ 7.4 Å piece from the 74 Å thickness ALONE — that plus half the scan "
          "extent is the floor; overfocus only adds to it. At α=100 mrad the "
          "untiled 48×70 Å cell wraps even at 5 nm overfocus on the Y axis; "
          "Phase C tiles to (2,2) (96×140 Å) to clear 20 nm comfortably.")

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
