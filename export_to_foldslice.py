# Run with: C:\Users\Trist\HyperSpy-bundle\python.exe export_to_foldslice.py
"""Bridge the stitched 4D-STEM datacube to fold_slice's HDF5 input.

Calls `reconstruct_ptycho.load_and_stitch()` (now native, non-square CBED)
and writes a single .h5 file with:

  /dp                  (Npos, Dy, Dx) float32 — diffraction intensities
  /scan_positions_a    (Npos, 2)      float64 — (y, x) in angstroms

plus all the engine-relevant scalars in root attrs. Per-axis Q-pixel
calibration is preserved so fold_slice can consume the native 65×95 CBED
without an anisotropic resample (the step that was biasing py4DSTEM's
auto-rotation).

Sign convention bridge: abTEM `defocus = -overfocus_a` ⇔ PtychoShelves
`C10 = +overfocus_a`. We write `+p.overfocus_a` into the attrs.

Exact HDF5 dataset names defer to Yu Lei's reply — the maximalist schema
here can be renamed in one place if his prepare-script wants different keys.

Usage::

    python export_to_foldslice.py --out dataset.h5
    python export_to_foldslice.py --tile-dir . --out dataset.h5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

import params as P
from reconstruct_ptycho import load_and_stitch

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


def _all_tile_pairs(p: P.Params) -> list[tuple[int, int]]:
    return [(i, j) for i in range(p.n_tiles_tiled) for j in range(p.n_tiles_tiled)]


def _scan_positions_a(Sy: int, Sx: int, p: P.Params) -> np.ndarray:
    """Build the (Npos, 2) scan-position list in angstroms.

    Grid is Sy × Sx, step = p.scan_step_a, centred on (center_y_a, center_x_a).
    Row-major flatten matches `data_4d.reshape(Sy*Sx, Dy, Dx)`.
    """
    ys = (np.arange(Sy) - (Sy - 1) / 2) * p.scan_step_a + p.center_y_a
    xs = (np.arange(Sx) - (Sx - 1) / 2) * p.scan_step_a + p.center_x_a
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([YY.ravel(), XX.ravel()], axis=1).astype("float64")


def _assert_contracts(data_4d: np.ndarray, scan_pos: np.ndarray, p: P.Params) -> None:
    """Section B.4 sanity invariants. Stop hard on any failure."""
    Sy, Sx, Dy, Dx = data_4d.shape
    expected_pos = (p.n_tiles_tiled * round(p.tile_size_a / p.scan_step_a)) ** 2
    assert Sy * Sx == expected_pos, (
        f"scan-count mismatch: got {Sy * Sx}, expected {expected_pos} "
        f"(= ({p.n_tiles_tiled} tiles × {round(p.tile_size_a / p.scan_step_a)} per axis)²)"
    )
    assert abs(p.scan_step_a - 0.25) < 0.1, (
        f"scan step {p.scan_step_a:.4f} Å is far from the expected 0.25 Å — "
        "either params drift or a TARGET_OVERLAP change; re-derive before exporting"
    )
    assert scan_pos.shape == (Sy * Sx, 2), (
        f"scan_positions shape {scan_pos.shape} != ({Sy * Sx}, 2)"
    )
    assert data_4d.dtype == np.float32, f"dp dtype {data_4d.dtype} != float32"

    # dq ratio invariant: dq_y / dq_x = pix_mrad_y / pix_mrad_x = box_x / box_y.
    dq_ratio = p.bf_eff_pixel_mrad_y / p.bf_eff_pixel_mrad
    box_ratio = p.box_x_a / p.box_y_a
    assert abs(dq_ratio - box_ratio) < 0.01, (
        f"dq ratio {dq_ratio:.4f} != box ratio {box_ratio:.4f} — "
        "per-axis Q calibration is inconsistent with the cell aspect"
    )


def export(out_path: Path, tile_dir: Path | None = None,
           *, pipeline: bool = False,
           counts_per_frame: float = 1e4) -> None:
    """
    pipeline=False (default): canonical export with native non-square CBED,
        per-axis Q calibration, and a /scan_positions_a list. For when Yu
        confirms fold_slice can ingest non-square + per-axis dq directly.

    pipeline=True: PSO_science-style ready-for-engine output. Square-resamples
        CBED to 65x65 at the coarser mrad/px (same as py4DSTEM has been seeing),
        writes /dp in MATLAB-friendly (Dy, Dx, Npos) column-major order, and
        omits the scan_positions_a list (fold_slice generates positions from
        raster nx/ny/step). Drop-in for `prepare_data_PSO_science.m` style
        loading. Use this for the first test run with guess params.
    """
    p = P.derive()
    tile_dir = tile_dir or P.PROJECT_ROOT
    tile_pairs = _all_tile_pairs(p)

    print("=" * 60)
    print(f"Exporting stitched 4D-STEM datacube to fold_slice HDF5  "
          f"[{'pipeline-square' if pipeline else 'native-non-square'}]")
    print("=" * 60)

    data_4d, sim_meta = load_and_stitch(tile_dir, p, tile_pairs)
    # data_4d shape: (Sy, Sx, Dy, Dx) float32  — native, non-square CBED

    # abTEM emits per-frame intensity that integrates to ~1 (probability units).
    # fold_slice's LSQ-ML / L1 error metric expects per-frame counts ~1e4-1e6;
    # at probability scale, gradient updates underflow to NaN at iter 1. Scale
    # each frame's mean sum to `counts_per_frame` (~10^4 e per frame at moderate
    # ptycho dose).
    mean_sum = float(data_4d.sum(axis=(2, 3)).mean())
    if mean_sum > 0 and counts_per_frame > 0:
        scale = counts_per_frame / mean_sum
        data_4d = data_4d * np.float32(scale)
        print(f"Scaled intensity: mean per-frame sum {mean_sum:.3e} -> "
              f"{counts_per_frame:.1e} (factor {scale:.3e})")
    Sy, Sx, Dy, Dx = data_4d.shape

    scan_pos_a = _scan_positions_a(Sy, Sx, p)
    _assert_contracts(data_4d, scan_pos_a, p)

    if pipeline:
        # Re-use py4DSTEM's square-resample to get a 65x65 CBED at the coarser
        # mrad/px. Matches what py4DSTEM has been seeing, so the test isolates
        # the engine change (LSQ-ML+momentum vs py4DSTEM's gradient descent).
        from reconstruct_ptycho import _resample_cbed_to_square
        data_4d = _resample_cbed_to_square(data_4d)
        Sy, Sx, Dy, Dx = data_4d.shape
        print(f"Pipeline mode: square-resampled CBED")
    # Flatten scan dims: write Python (Npos, Dy, Dx). MATLAB h5read REVERSES
    # dim order (HDF5 row-major -> MATLAB column-major), so MATLAB sees
    # [Dx, Dy, Npos] = [65, 65, 6400] — exactly what fold_slice's prep wants
    # (positions on axis 3). Per-frame Dy/Dx are transposed vs Python labels;
    # immaterial for square CBED with no scan rotation.
    dp = data_4d.reshape(Sy * Sx, Dy, Dx)
    print(f"Flattened: dp shape = {dp.shape}, scan_positions shape = {scan_pos_a.shape}")

    lam_a = p.wavelength_a
    # eff mrad/px → Å⁻¹/px: dk = (pix_mrad · 1e-3) / λ_a
    # Empirical axis order (verified from tile zarrs): array shape is
    # (Sy, Sx, Dy=65, Dx=95). Dy = coarse axis (box_x reciprocal); Dx = fine
    # axis (box_y reciprocal). Earlier plan had these flipped — see W1 smoke.
    dq_y_inv_a = float(p.bf_eff_pixel_mrad   * 1e-3 / lam_a)  # along Dy (65 px, coarse)
    dq_x_inv_a = float(p.bf_eff_pixel_mrad_y * 1e-3 / lam_a)  # along Dx (95 px, fine)

    if pipeline:
        print(f"Q calibration (pipeline / square): both axes at "
              f"{p.bf_eff_pixel_mrad:.4f} mrad/px (= coarse value, "
              f"finer axis downsampled to match)")
    else:
        print(f"Per-axis Q calibration:")
        print(f"  Dy (coarse, {Dy} px): {p.bf_eff_pixel_mrad:.4f} mrad/px  "
              f"= {dq_y_inv_a:.6f} Å⁻¹/px  (1/box_x where box_x={p.box_x_a:.2f} Å)")
        print(f"  Dx (fine,   {Dx} px): {p.bf_eff_pixel_mrad_y:.4f} mrad/px  "
              f"= {dq_x_inv_a:.6f} Å⁻¹/px  (1/box_y where box_y={p.box_y_a:.2f} Å)")
    print(f"C10 sign convention: abTEM defocus = {-p.overfocus_a:+.2f} Å  "
          f"⇒ PtychoShelves C10 = {+p.overfocus_a:+.2f} Å")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # BF disk radius in pixels — fold_slice computes dk = alpha/1e3/rbf/lambda
    # from this, so it needs to match the (square) pixelation that was written.
    # In pipeline mode, the CBED is at the coarser mrad/px = bf_eff_pixel_mrad.
    pixel_mrad_for_dk = float(p.bf_eff_pixel_mrad) if pipeline else float(p.bf_eff_pixel_mrad)
    rbf_px = p.convergence_mrad / pixel_mrad_for_dk   # ≈ 16.2 px at 100/6.17

    chunks = (1, Dy, Dx)
    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "dp",
            data=dp,
            compression="gzip",
            compression_opts=4,
            chunks=chunks,
        )
        if not pipeline:
            # Native mode: positions list goes in the HDF5 so fold_slice can
            # use the exact (non-grid) scan layout. Pipeline mode generates
            # positions from raster nx/ny/step on the MATLAB side.
            f.create_dataset("scan_positions_a", data=scan_pos_a)

        # Engine-relevant scalars. fold_slice's prepare-script reads these.
        f.attrs["energy_ev"]         = float(p.energy_ev)
        f.attrs["voltage_kev"]       = float(p.energy_ev / 1e3)        # fold_slice convention
        f.attrs["convergence_mrad"]  = float(p.convergence_mrad)
        f.attrs["alpha_max_mrad"]    = float(p.convergence_mrad)        # alias for recon.m
        # C10_a uses py4DSTEM sign (+overfocus). defocus_a uses fold_slice /
        # abTEM sign (-overfocus). Both are written so consumers don't have
        # to guess the convention.
        f.attrs["C10_a"]             = float(+p.overfocus_a)
        f.attrs["defocus_a"]         = float(-p.overfocus_a)            # fold_slice probe_df
        f.attrs["dq_y_inv_a"]        = dq_y_inv_a
        f.attrs["dq_x_inv_a"]        = dq_x_inv_a
        f.attrs["dy_pixel_mrad"]     = float(p.bf_eff_pixel_mrad)    # Dy coarse, 65 px
        f.attrs["dx_pixel_mrad"]     = float(p.bf_eff_pixel_mrad_y)  # Dx fine,   95 px
        f.attrs["pixel_mrad"]        = pixel_mrad_for_dk            # the one fold_slice uses
        f.attrs["rbf_px"]            = float(rbf_px)
        f.attrs["scan_step_a"]       = float(p.scan_step_a)
        f.attrs["scan_nx"]           = int(Sx)                       # fold_slice raster gen
        f.attrs["scan_ny"]           = int(Sy)
        f.attrs["wavelength_a"]      = float(lam_a)
        f.attrs["object_thickness_a"]= float(p.real_space_thickness_a)
        f.attrs["num_slices"]        = int(p.recon_num_slices)
        f.attrs["slice_thickness_a"] = float(p.recon_slice_thickness_a)
        f.attrs["beam_axis"]         = "z"
        f.attrs["mode"]              = "pipeline_square_65" if pipeline else "native_non_square"
        f.attrs["dp_axis_order"]     = (
            "(Dy, Dx, Npos) MATLAB-column-major; CBED is 65x65 square at "
            f"{pixel_mrad_for_dk:.4f} mrad/px (coarse axis pixel size)"
        ) if pipeline else (
            "Npos, Dy, Dx; Dy is the COARSE axis (≈6.17 mrad/px, ~65 px, "
            "reciprocal of box_x), Dx is the FINE axis (≈4.22 mrad/px, "
            "~95 px, reciprocal of box_y)"
        )
        f.attrs["source"]            = sim_meta.get("loaded_from", "")
        f.attrs["phonon_dwf_scale"]  = float(p.phonon_dwf_scale)

    out_gb = out_path.stat().st_size / 1e9
    print(f"\nWrote {out_path}  ({out_gb:.2f} GB on disk)")
    print(f"  /dp                 {dp.shape}  {dp.dtype}")
    if not pipeline:
        print(f"  /scan_positions_a   {scan_pos_a.shape}  {scan_pos_a.dtype}")
    print(f"  rbf_px ≈ {rbf_px:.2f}  (fold_slice: dk = alpha/1e3/rbf/lambda)")
    print(f"  defocus_a = {-p.overfocus_a:+.2f} Å  (fold_slice probe_df, "
          f"= -overfocus = -{p.overfocus_a:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=P.PROJECT_ROOT / "dataset.h5",
                    help="Output HDF5 path (default: dataset.h5 in project root)")
    ap.add_argument("--tile-dir", type=Path, default=None,
                    help="Where to look for tile zarrs (default: project root)")
    ap.add_argument("--pipeline", action="store_true",
                    help="Pipeline mode: square-resampled 65x65 CBED in MATLAB "
                         "(Dy, Dx, Npos) order, no scan_positions list. "
                         "Drop-in for fold_slice's PSO_science recon script.")
    ap.add_argument("--counts-per-frame", type=float, default=1e4,
                    help="Rescale data so each frame integrates to this many "
                         "electron counts (default 1e4). abTEM emits "
                         "probability-unit data; fold_slice needs ~counts.")
    args = ap.parse_args()
    export(args.out, tile_dir=args.tile_dir, pipeline=args.pipeline,
           counts_per_frame=args.counts_per_frame)


if __name__ == "__main__":
    main()
