# Run with: C:\Users\Trist\HyperSpy-bundle\python.exe reconstruct_ptycho.py
"""py4DSTEM multislice ptychography reconstruction with VRAM-tight settings.

Differences from the original notebook (Cell 3):

1. ``polar_parameters['C10'] = -OVERFOCUS_A``  (was ``+`` in the broken copy;
   the ``WORKS`` notebook used ``-``). py4DSTEM uses ``C10 = -defocus`` so a
   probe focused 10 Å above the sample (overfocus, abTEM defocus = +10 Å)
   has C10 = -10 Å.
2. ``crop_patterns=True`` with ``diffraction_intensities_shape`` set so the
   working FFT fits in 8 GB VRAM.
3. Two-stage reconstruction:
   - Stage A: ``fix_probe=True``, ``identical_slices=True``, hard kz reg.
     Force the solver onto the depth-uniform manifold first.
   - Stage B: release everything, fit aberrations, lighter kz reg, TV denoise.
4. ``kz_regularization_filter=True`` (was ``False`` — depth prior was off).
5. ``num_slices`` matches the optical depth resolution (~1 Å/slice) instead of
   half of the simulation slice count.
6. Calibration ``Q_pixel_size`` recomputed from the correct beam axis and the
   chosen detector bin, not from ``BOX_Y / bin``.

Usage::

    python reconstruct_ptycho.py --toy
    python reconstruct_ptycho.py --stage both
    python reconstruct_ptycho.py --stage A          # warm-up only
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import params as P


def _setup_cuda_path() -> None:
    cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
    bin_path = os.path.join(cuda_root, "bin")
    bin_x64_path = os.path.join(bin_path, "x64")
    os.environ["PATH"] = bin_path + os.pathsep + bin_x64_path + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(bin_path)
        os.add_dll_directory(bin_x64_path)
    except (AttributeError, FileNotFoundError):
        pass


def _diagnose_object(ptycho, label: str) -> None:
    """Print NaN / amplitude / phase stats so we can localise divergence."""
    obj = np.asarray(ptycho.object)
    nan_frac = float(np.isnan(obj).sum()) / obj.size
    amp = np.abs(obj)
    finite_mask = np.isfinite(amp)
    amp_finite = amp[finite_mask] if finite_mask.any() else np.array([np.nan])
    phase_finite = np.angle(obj)[finite_mask] if finite_mask.any() else np.array([np.nan])
    print(f"  [diag {label}] shape={obj.shape} dtype={obj.dtype}  "
          f"NaN={nan_frac*100:.1f}%  "
          f"|obj| min/med/max = "
          f"{amp_finite.min():.3g}/{np.median(amp_finite):.3g}/{amp_finite.max():.3g}  "
          f"phase min/max = {phase_finite.min():.3g}/{phase_finite.max():.3g}")


def load_and_stitch(tile_dir: Path, p: P.Params,
                    tile_pairs: list[tuple[int, int]]) -> tuple[np.ndarray, dict]:
    """Stitch tile BF zarrs into a single (Sy, Sx, Dy, Dx) float32 array.

    Falls back to the parent dir if a tile is missing locally — the
    pre-existing 9-tile data lives in ``MY HYPERSPY CODE``.
    """
    import zarr

    tile_files: list[tuple[int, int, Path]] = []
    for (i, j) in tile_pairs:
        cand = [tile_dir / f"tile{i}{j}_bf.zarr",
                P.PARENT_ROOT / f"tile{i}{j}_bf.zarr"]
        for c in cand:
            if c.exists():
                tile_files.append((i, j, c))
                break
        else:
            print(f"[warn] tile {(i,j)} missing; will be left as zeros")

    if not tile_files:
        raise FileNotFoundError("No tile BF zarrs found in tile_dir or parent.")

    first = zarr.open(str(tile_files[0][2]), mode="r")
    tile_shape = first.shape
    Sy, Sx = tile_shape[:2]
    Dy, Dx = tile_shape[2:]

    # Drop tiles whose CBED shape doesn't match the first tile — this prevents
    # old notebook zarrs (different binning/detector params) from silently
    # mixing with freshly simulated data.
    compatible = []
    for entry in tile_files:
        z = zarr.open(str(entry[2]), mode="r")
        if z.shape[2:] != (Dy, Dx):
            print(f"[skip] tile ({entry[0]},{entry[1]}) {entry[2].name}: "
                  f"CBED shape {z.shape[2:]} != expected ({Dy},{Dx}); "
                  f"likely from old notebook run — ignoring")
        else:
            compatible.append(entry)
    if not compatible:
        raise RuntimeError(
            f"No tiles with matching CBED shape ({Dy},{Dx}) found. "
            "Run simulate_4dstem.py first to generate compatible data."
        )
    tile_files = compatible

    rows = max(i for i, _, _ in tile_files) + 1
    cols = max(j for _, j, _ in tile_files) + 1
    out = np.zeros((rows * Sy, cols * Sx, Dy, Dx), dtype="float32")

    metadata = {}
    for i, j, fp in tile_files:
        z = zarr.open(str(fp), mode="r")
        out[i * Sy:(i + 1) * Sy, j * Sx:(j + 1) * Sx, :, :] = np.asarray(z[:], dtype="float32")
        if not metadata:
            metadata = dict(z.attrs)
            metadata["loaded_from"] = str(fp)
        print(f"  loaded tile ({i},{j}) {fp.name}: {z.shape}")

    print(f"Stitched: {out.shape} ({out.nbytes/1e9:.2f} GB in RAM)")

    # py4DSTEM requires equal resampling factors on both diffraction axes.
    # The CBED is non-square because box_x ≠ box_y: after K-binning the two
    # axes have different mrad/px (6.17 vs 4.22 here). Zoom the finer axis
    # down to match the coarser one — this preserves total angular coverage
    # and makes both axes equal to p.bf_eff_pixel_mrad (the coarser value).
    Sdy, Sdx = out.shape[2], out.shape[3]
    if Sdy != Sdx:
        from scipy.ndimage import zoom as _zoom
        sq = min(Sdy, Sdx)
        out = _zoom(out, [1.0, 1.0, sq / Sdy, sq / Sdx],
                    order=1).astype("float32")
        print(f"  CBED resampled to square {sq}×{sq} (was {Sdy}×{Sdx}; "
              f"both axes now ~{metadata.get('bf_eff_pixel_mrad', '?'):.2f} mrad/px)")

    return out, metadata


def _calibrate(datacube, r_pixel_size: float, q_pixel_size: float) -> None:
    """Apply real- and reciprocal-space pixel calibrations + origin."""
    datacube.calibration.set_R_pixel_size(r_pixel_size)
    datacube.calibration.set_R_pixel_units("A")
    datacube.calibration.set_Q_pixel_size(q_pixel_size)
    datacube.calibration.set_Q_pixel_units("A^-1")
    Dy, Dx = datacube.data.shape[2:]
    datacube.calibration.set_origin((Dx // 2, Dy // 2))


def reconstruct(p: P.Params, *, stage: str = "both",
                tile_dir: Path | None = None,
                out_path: Path | None = None,
                tile_pairs: list[tuple[int, int]] | None = None) -> Path:
    """Run multislice ptychography. Returns the output zarr path."""
    import py4DSTEM

    tile_dir = tile_dir or P.PROJECT_ROOT
    out_path = out_path or P.PROJECT_ROOT / "ptycho_recon.zarr"
    tile_pairs = tile_pairs or [(i, j) for i in range(p.n_tiles_tiled)
                                for j in range(p.n_tiles_tiled)]

    print("=" * 60)
    print(f"Multislice ptychography | stage={stage}")
    print("=" * 60)

    data_4d, sim_meta = load_and_stitch(tile_dir, p, tile_pairs)

    # Reciprocal pixel size in Å⁻¹ (NOT in mrad). dk = pix_mrad / (1000 · λ).
    # Prefer values written by the sim into the tile zarr over a fresh
    # params.derive() — guards against a params/data mismatch silently
    # mis-calibrating the real- or reciprocal-space pixel size.
    eff_mrad = float(sim_meta.get("bf_eff_pixel_mrad", p.bf_eff_pixel_mrad))
    q_pixel_size = eff_mrad / (1000.0 * p.wavelength_a)
    r_pixel_size = float(sim_meta.get("scan_step_a", p.scan_step_a))
    if abs(r_pixel_size - p.scan_step_a) > 1e-6:
        print(f"[warn] sim scan_step_a={r_pixel_size:.4f} Å differs from "
              f"params {p.scan_step_a:.4f} Å — using the sim value")
    print(f"Calibration: R={r_pixel_size:.4f} Å/px, "
          f"Q={q_pixel_size:.4f} Å⁻¹/px (from {eff_mrad:.2f} mrad/px)")

    datacube = py4DSTEM.DataCube(data=data_4d)
    _calibrate(datacube, r_pixel_size, q_pixel_size)
    del data_4d
    gc.collect()

    # The sim now uses abTEM `defocus = -overfocus_a` for true OVERFOCUS
    # (crossover above the sample — verified by test_probe_focus.py). Since
    # abTEM `defocus = -C10` and py4DSTEM's `C10` is the same chi coefficient,
    # abTEM `defocus = -overfocus_a`  <=>  C10 = +overfocus_a. So py4DSTEM C10
    # must be +overfocus_a to model the same probe. (This matches the original
    # working notebook; the plan doc's `-overfocus` was backwards.)
    overfocus_a = float(sim_meta.get("overfocus_a", p.overfocus_a))
    polar_params = {"C10": +overfocus_a}
    print(f"Probe init: py4DSTEM C10 = {polar_params['C10']:+.2f} Å  "
          f"(overfocus; matches sim abTEM defocus = {-overfocus_a:+.2f})")

    # Pass a scalar float — py4DSTEM broadcasts it to all slices.
    # A list of length (num_slices-1) also works but the float is cleaner.

    print(f"Solver: {p.recon_num_slices} slices × "
          f"{p.recon_slice_thickness_a:.2f} Å, "
          f"crop={p.recon_diff_intensities_shape}, "
          f"batch={p.recon_max_batch_size}")

    ptycho = py4DSTEM.process.phase.MultislicePtychography(
        datacube=datacube,
        energy=p.energy_ev,
        num_slices=p.recon_num_slices,
        slice_thicknesses=p.recon_slice_thickness_a,   # scalar — broadcast to all slices
        semiangle_cutoff=p.convergence_mrad,
        polar_parameters=polar_params,
        device="gpu",
        verbose=True,
        object_type="complex",
    )

    ptycho.preprocess(
        plot_center_of_mass=False,       # str/bool accepted; False = skip
        plot_rotation=False,             # avoid GUI pop-up in headless run
        plot_probe_overlaps=False,       # idem
        crop_patterns=True,
        diffraction_intensities_shape=p.recon_diff_intensities_shape,
        store_initial_arrays=True,
    )

    stage_a_ran = False
    if stage in ("A", "both"):
        print("\n--- Stage A: probe-fixed warm-up ---")
        t0 = time.time()
        # Match the working notebook's parameters as closely as possible
        # for the diagnostic. Once we confirm finite output, the plan's
        # depth-sectioning improvements (kz_reg, identical_slices) can be
        # re-introduced one at a time.
        ptycho.reconstruct(
            num_iter=24,
            max_batch_size=p.recon_max_batch_size,
            step_size=0.025,                       # notebook value
            fix_probe=True,
            fit_probe_aberrations=False,
            kz_regularization_filter=False,        # notebook didn't use
            gaussian_filter=True,
            gaussian_filter_sigma=0.1,             # notebook value (was 0.3)
            object_positivity=False,
            fix_potential_baseline=True,
            store_iterations=False,
        )
        stage_a_ran = True
        print(f"Stage A: {(time.time()-t0)/60:.1f} min")
        _diagnose_object(ptycho, "after Stage A")

    if stage in ("B", "both"):
        if not stage_a_ran:
            raise RuntimeError(
                "--stage B cannot run alone: Stage A must precede it in the same "
                "process to provide the probe-fixed warm-up state. "
                "Re-run with --stage both."
            )
        print("\n--- Stage B: release slices + fit probe ---")
        # Flush the CUDA memory pool before the new reconstruct call.
        # Without this, py4DSTEM can deadlock at 0 iterations waiting for a
        # fragmented allocation left over from Stage A.
        gc.collect()
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            print("  CUDA memory pool flushed")
        except Exception:
            pass

        t0 = time.time()
        ptycho.reconstruct(
            num_iter=96,
            max_batch_size=p.recon_max_batch_size,
            step_size=0.05,            # 0.1 let the object amplitude run wild on small data
            reset=False,   # continues from Stage A state; never reaches here without it
            fix_probe=False,
            fit_probe_aberrations=True,
            fit_probe_aberrations_remove_initial=False,
            fit_probe_aberrations_using_scikit_image=False,  # scikit-image unwrapper has a documented kernel-hang bug
            constrain_probe_amplitude=True,   # stabilises probe on release from fix_probe
            pure_phase_object=True,    # PTO/STO is ~pure phase; pinning |obj|=1 kills the
                                       # amplitude/phase degeneracy that ate the depth info
            kz_regularization_filter=True,
            kz_regularization_gamma=0.2,   # ablated up from 0.02 — needed to break the
                                           # flat-projection minimum (std ratio 1.07 -> 1.81)
            tv_denoise=True,               # direct piecewise-depth prior (layered/vortex sample)
            tv_denoise_weights=[1e-3, 1e-4],  # [kz, kxy] — strong z smoothing toward
                                              # piecewise-constant depth, gentle in-plane
            identical_slices=False,
            gaussian_filter=True,
            gaussian_filter_sigma=0.1,
            object_positivity=False,
            fix_potential_baseline=True,
            store_iterations=False,
        )
        print(f"Stage B: {(time.time()-t0)/60:.1f} min")
        _diagnose_object(ptycho, "after Stage B")

    _save_zarr(ptycho, out_path, p, sim_meta)
    return out_path


def _save_zarr(ptycho, out_path: Path, p: P.Params, sim_meta: dict) -> None:
    import zarr
    import numcodecs

    obj = np.asarray(ptycho.object)
    probe = np.asarray(ptycho.probe)
    compressor = numcodecs.Blosc(cname="zstd", clevel=3,
                                 shuffle=numcodecs.Blosc.BITSHUFFLE)

    root = zarr.open(str(out_path), mode="w")
    root.create_dataset("object_complex", data=obj.astype("complex64"),
                        chunks=(1, *obj.shape[1:]), compressor=compressor)
    root.create_dataset("object_phase",
                        data=np.angle(obj).astype("float32"),
                        chunks=(1, *obj.shape[1:]), compressor=compressor)
    root.create_dataset("probe", data=probe.astype("complex64"),
                        compressor=compressor)
    pixel_size = float(np.atleast_1d(np.asarray(ptycho.sampling))[0])

    metadata = {
        "num_slices": p.recon_num_slices,
        "slice_thickness_a": p.recon_slice_thickness_a,
        "real_space_thickness_a": p.real_space_thickness_a,
        "scan_step_a": p.scan_step_a,
        "reconstruction_pixel_size_a": pixel_size,
        "bf_bin": p.bf_bin,
        "bf_eff_pixel_mrad": p.bf_eff_pixel_mrad,
        "overfocus_a": p.overfocus_a,
        "polar_C10": +p.overfocus_a,
        "energy_ev": p.energy_ev,
        "convergence_mrad": p.convergence_mrad,
        "diff_intensities_shape": list(p.recon_diff_intensities_shape),
        "phonon_num_configs": sim_meta.get("phonon_num_configs"),
        "sim_slice_thickness_a": sim_meta.get("sim_slice_thickness_a"),
        "loaded_from_first_tile": sim_meta.get("loaded_from"),
    }
    root.attrs.update(metadata)

    side = out_path.parent / (out_path.stem + "_metadata.json")
    side.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path} (object {obj.shape}, dtype {obj.dtype})")
    print(f"Sidecar metadata: {side}")


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true",
                    help="Single-tile (1,1) gating run with toy params.")
    ap.add_argument("--mini", action="store_true",
                    help="2x2 tile gating run (8x8 Å scan area) with toy params. "
                         "Required for a meaningful quality check before production.")
    ap.add_argument("--stage", choices=("A", "B", "both"), default="both")
    ap.add_argument("--cryo", action="store_true",
                    help="Match a --cryo sim (phonon σ ×0.65); affects recorded "
                         "metadata only — calibration is taken from the tile zarrs.")
    ap.add_argument("--tile-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.toy and args.mini:
        ap.error("Pass --toy OR --mini, not both.")

    _setup_cuda_path()
    dwf = 0.65 if args.cryo else P.PHONON_DWF_SCALE_DEFAULT
    if args.toy or args.mini:
        p = P.toy_params(dwf_scale=dwf)
    else:
        p = P.derive(dwf_scale=dwf)

    if args.mini:
        tile_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        if args.out is None:
            args.out = P.PROJECT_ROOT / "ptycho_recon_mini.zarr"
    elif args.toy:
        tile_pairs = [(1, 1)]
        if args.out is None:
            args.out = P.PROJECT_ROOT / "ptycho_recon_toy.zarr"
    else:
        tile_pairs = None  # full n_tiles_tiled × n_tiles_tiled grid

    print(P.summary(p))
    reconstruct(p, stage=args.stage, tile_dir=args.tile_dir, out_path=args.out,
                tile_pairs=tile_pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
