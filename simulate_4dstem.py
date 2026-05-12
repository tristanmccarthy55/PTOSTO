# Run with: C:\Users\Trist\HyperSpy-bundle\python.exe simulate_4dstem.py
"""abTEM 4D-STEM simulation for the PTO/STO labyrinth.

Differences from the original notebook (Cell 2):
- ``FrozenPhonons`` enabled (per :mod:`params`).
- Beam is **+Z** (abTEM convention); thickness uses ``cell.lengths()[2]``.
- Probe defocus uses the abTEM convention ``defocus = +OVERFOCUS_A`` for
  "probe focused above the sample" (overfocus).
- Detector is sized via :func:`params.derive` so the saved 4D-STEM is small
  enough to keep storage and reconstruction VRAM tractable on 8 GB.
- BF data is binned with :func:`dask.array.coarsen` *before* hitting disk.

Usage::

    python simulate_4dstem.py --tile 1 1            # one tile, production
    python simulate_4dstem.py --tile 1 1 --toy      # < 1 h smoke test
    python simulate_4dstem.py --all                 # full n_tiles_tiled² grid
    python simulate_4dstem.py --all --cryo          # ... with cryo (×0.65) DWFs
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

import params as P


def _setup_cuda_path() -> None:
    """Add the user's CUDA 13.1 bin dirs to PATH so cupy can find DLLs."""
    cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
    bin_path = os.path.join(cuda_root, "bin")
    bin_x64_path = os.path.join(bin_path, "x64")
    os.environ["PATH"] = bin_path + os.pathsep + bin_x64_path + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(bin_path)
        os.add_dll_directory(bin_x64_path)
    except (AttributeError, FileNotFoundError):
        pass


def _tile_bounds(tile_i: int, tile_j: int, p: P.Params) -> Tuple[float, float, float, float]:
    """Return (tx_start, ty_start, tx_end, ty_end) in angstroms for tile (i,j).

    The scan window is ``p.scan_window_a`` (= ``p.n_tiles_tiled · p.tile_size_a``)
    centred on (``p.center_x_a``, ``p.center_y_a``), so the simulated
    ``n_tiles_tiled × n_tiles_tiled`` grid is symmetric about that centre.
    """
    scan_start_x = p.center_x_a - p.scan_window_a / 2
    scan_start_y = p.center_y_a - p.scan_window_a / 2
    scan_end_x = p.center_x_a + p.scan_window_a / 2
    scan_end_y = p.center_y_a + p.scan_window_a / 2

    edges_x = np.arange(scan_start_x, scan_end_x, p.tile_size_a)
    edges_y = np.arange(scan_start_y, scan_end_y, p.tile_size_a)
    if edges_x[-1] < scan_end_x:
        edges_x = np.append(edges_x, scan_end_x)
    if edges_y[-1] < scan_end_y:
        edges_y = np.append(edges_y, scan_end_y)

    if tile_i + 1 >= len(edges_x) or tile_j + 1 >= len(edges_y):
        raise ValueError(f"Tile ({tile_i},{tile_j}) outside scan grid")

    return (
        float(edges_x[tile_i]),
        float(edges_y[tile_j]),
        float(edges_x[tile_i + 1]),
        float(edges_y[tile_j + 1]),
    )


def _build_potential(atoms, p: P.Params, num_configs: int):
    """Construct the FrozenPhonons-aware Potential. Lazy — do not .compute()."""
    import abtem
    from abtem import FrozenPhonons

    fp = FrozenPhonons(atoms, num_configs=num_configs,
                       sigmas=p.phonon_sigmas_a, seed=0)
    # Use the same potential sampling rule the notebook used: λ/(2 sin θ_max)
    # but bound below by 0.05 Å so it doesn't blow up the grid for marginal
    # angular gain.
    sampling = max(0.05, p.wavelength_a / (2.0 * np.sin(p.detector_max_angle_mrad * 1e-3)))
    return abtem.Potential(
        fp,
        sampling=sampling,
        slice_thickness=p.sim_slice_thickness_a,
        device="cpu",
        parametrization="kirkland",
    )


def _build_probe(p: P.Params, potential):
    """Probe with the matched grid and the **correct** abTEM defocus sign.

    Verified empirically (test_probe_focus.py) and from abtem/transfer.py:
    abTEM ``defocus = -C10``; a *negative* defocus puts the crossover *above*
    the entrance plane (in the source-side vacuum) — i.e. the probe is focused
    BEFORE it enters the sample = OVERFOCUS. ``defocus = +OVERFOCUS_A`` would
    instead focus the probe ~OVERFOCUS_A deep *inside* the crystal (underfocus).
    So for overfocus we pass ``-overfocus_a``. (This matches the original
    notebook; the rebuild's earlier ``+overfocus_a`` was the wrong sign.)
    """
    import abtem
    probe = abtem.Probe(
        energy=p.energy_ev,
        semiangle_cutoff=p.convergence_mrad,
        device="gpu",
    )
    probe.aberrations.defocus = -p.overfocus_a   # negative => overfocus (focus before sample)
    probe.grid.match(potential)
    return probe


def run_single_tile(tile_i: int, tile_j: int, p: P.Params, *,
                    num_configs: int | None = None,
                    out_dir: Path | None = None,
                    overwrite: bool = False) -> Path:
    """Simulate one tile end-to-end. Returns the BF zarr path."""
    import abtem
    import dask
    import dask.array as da
    from dask.diagnostics import ProgressBar
    import zarr
    import numcodecs
    import ase.io

    out_dir = out_dir or P.PROJECT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_name = f"tile{tile_i}{tile_j}"
    bf_path = out_dir / f"{tile_name}_bf.zarr"
    haadf_path = out_dir / f"{tile_name}_haadf.zarr"

    if bf_path.exists() and not overwrite:
        print(f"[skip] {bf_path} already exists; pass --overwrite to redo.")
        return bf_path

    n_cfg = num_configs if num_configs is not None else p.phonon_num_configs

    print("=" * 60)
    print(f"4D-STEM tile [{tile_i},{tile_j}] | n_configs={n_cfg} | "
          f"slice={p.sim_slice_thickness_a} Å")
    print("=" * 60)

    atoms = P.load_atoms()
    print(f"Cell after orthogonalize: "
          f"{atoms.cell.lengths()[0]:.2f} × {atoms.cell.lengths()[1]:.2f} × "
          f"{atoms.cell.lengths()[2]:.2f} Å (beam=Z); {len(atoms)} atoms")

    potential = _build_potential(atoms, p, num_configs=n_cfg)
    probe = _build_probe(p, potential)
    print(f"Probe: abTEM defocus = {-p.overfocus_a:+.2f} Å "
          f"(overfocus: crossover {p.overfocus_a:.1f} Å above the sample); "
          f"semiangle = {p.convergence_mrad:.0f} mrad")

    tx_start, ty_start, tx_end, ty_end = _tile_bounds(tile_i, tile_j, p)
    scan = abtem.GridScan(start=(tx_start, ty_start),
                          end=(tx_end, ty_end),
                          sampling=p.scan_step_a)
    print(f"Tile scan: {scan.shape[0]} × {scan.shape[1]} positions, "
          f"({tx_start:.1f},{ty_start:.1f}) -> ({tx_end:.1f},{ty_end:.1f}) Å")

    # Compute the actual simulated angular range from the probe grid so that
    # the HAADF outer limit never exceeds what abTEM can integrate.
    gpts = probe.grid.gpts      # (ny, nx) grid points
    extent = probe.grid.extent  # (Ly, Lx) Å
    k_max = min((gpts[0] // 2) / extent[0],
                (gpts[1] // 2) / extent[1])  # Å^-1
    sim_max_mrad = k_max * p.wavelength_a * 1000.0
    haadf_outer = min(p.haadf_outer_mrad, sim_max_mrad * 0.97)
    if haadf_outer < p.haadf_outer_mrad:
        print(f"  HAADF outer clamped: {p.haadf_outer_mrad:.0f} → "
              f"{haadf_outer:.1f} mrad  (sim max ≈ {sim_max_mrad:.1f} mrad)")
    det_haadf = abtem.AnnularDetector(inner=p.haadf_inner_mrad,
                                      outer=haadf_outer)
    det_bf = abtem.PixelatedDetector(max_angle=p.detector_max_angle_mrad)

    measurements = probe.scan(potential, scan=scan, detectors=[det_haadf, det_bf])
    haadf_dask = measurements[0].array
    bf_dask = measurements[1].array

    # Bin the BF before storage. trim_excess=True so the binned shape is
    # (..., n_native // K, n_native // K).
    bf_binned = da.coarsen(np.sum, bf_dask,
                           {2: p.bf_bin, 3: p.bf_bin},
                           trim_excess=True)

    print(f"BF native shape: {bf_dask.shape}  ->  binned ({p.bf_bin}×): {bf_binned.shape}")

    compressor = numcodecs.Blosc(cname="zstd", clevel=3,
                                 shuffle=numcodecs.Blosc.BITSHUFFLE)
    bf_target = zarr.open(str(bf_path), mode="w", shape=bf_binned.shape,
                          chunks=(1, 1, *bf_binned.shape[2:]),
                          dtype="float32", compressor=compressor)
    haadf_target = zarr.open(str(haadf_path), mode="w", shape=haadf_dask.shape,
                             chunks=haadf_dask.chunksize,
                             dtype=haadf_dask.dtype, compressor=compressor)

    # Persist sim metadata next to the BF zarr.
    bf_target.attrs.update({
        "tile_i": tile_i, "tile_j": tile_j,
        "scan_step_a": p.scan_step_a,
        "tx_start_a": tx_start, "ty_start_a": ty_start,
        "tx_end_a": tx_end, "ty_end_a": ty_end,
        "energy_ev": p.energy_ev,
        "convergence_mrad": p.convergence_mrad,
        "overfocus_a": p.overfocus_a,
        "detector_max_angle_mrad": p.detector_max_angle_mrad,
        "bf_bin": p.bf_bin,
        "bf_eff_pixel_mrad": p.bf_eff_pixel_mrad,
        "sim_slice_thickness_a": p.sim_slice_thickness_a,
        "num_slices_sim": p.num_slices_sim,
        "phonon_num_configs": n_cfg,
        "phonon_sigmas_a": p.phonon_sigmas_a,
        "phonon_dwf_scale": p.phonon_dwf_scale,
        "n_tiles_tiled": p.n_tiles_tiled,
        "scan_window_a": p.scan_window_a,
        "real_space_thickness_a": p.real_space_thickness_a,
        "wavelength_a": p.wavelength_a,
    })

    print("Streaming to disk...")
    t0 = time.time()
    with ProgressBar():
        da.store([haadf_dask, bf_binned], [haadf_target, bf_target], compute=True)
    dt = time.time() - t0
    print(f"Done tile [{tile_i},{tile_j}] in {dt/60:.1f} min "
          f"({len(scan)/dt:.1f} pos/s)  ->  {bf_path.name}")
    return bf_path


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", nargs=2, type=int, metavar=("I", "J"))
    ap.add_argument("--all", action="store_true",
                    help="Run the whole n_tiles_tiled × n_tiles_tiled grid "
                         "(default 6×6 = 36 tiles) in series.")
    ap.add_argument("--toy", action="store_true",
                    help="Toy config (n_configs=2, slice=2 Å, 2×2 grid, smaller batch).")
    ap.add_argument("--mini", action="store_true",
                    help="Mini gating run: 2x2 tile block (0,0)(0,1)(1,0)(1,1) "
                         "with toy params, at the production overfocus. Gives a "
                         "real ptychography quality check before the full run.")
    ap.add_argument("--cryo", action="store_true",
                    help="Scale all phonon σ by 0.65 (≈LN2/cryo DWFs, W3).")
    ap.add_argument("--num-configs", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    _setup_cuda_path()

    dwf = 0.65 if args.cryo else P.PHONON_DWF_SCALE_DEFAULT
    if args.toy or args.mini:
        p = P.toy_params(dwf_scale=dwf)
    else:
        p = P.derive(dwf_scale=dwf)
    print(P.summary(p))

    if args.mini:
        tiles = [(0, 0), (0, 1), (1, 0), (1, 1)]
    elif args.all:
        tiles = [(i, j) for i in range(p.n_tiles_tiled) for j in range(p.n_tiles_tiled)]
    elif args.tile is not None:
        tiles = [tuple(args.tile)]
    elif args.toy:
        tiles = [(1, 1)]   # default toy tile = centre
    else:
        ap.error("Pass --tile I J, --all, --toy, or --mini.")

    t0 = time.time()
    for (i, j) in tiles:
        run_single_tile(i, j, p,
                        num_configs=args.num_configs,
                        out_dir=args.out_dir,
                        overwrite=args.overwrite)
    print(f"\nTotal wall-clock: {(time.time()-t0)/3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
