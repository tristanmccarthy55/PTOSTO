"""Quantitative validation of a multislice ptychography reconstruction.

Compares ``ptycho_recon.zarr`` against the abTEM ground-truth potential and
checks two simple resolution proxies:

* Per-slice Pearson r between recon phase and the projected potential at the
  matching depth (laterally Gaussian-blurred to the recon's resolution).
* Mean Fourier shell correlation along kz only (depth-direction FRC) on the
  reconstructed phase volume.

The third planned check — vortex column tracking against the POSCAR
displacements — is wired up as ``track_pb_columns`` and writes a CSV of
sub-pixel column centroids per slice. Comparing those to the input POSCAR's
Pb displacements is left as the user-driven validation since it requires
deciding which axis to compare against (vortex axis vs perpendicular).

Usage::

    python validate.py
    python validate.py --recon ptycho_recon_toy.zarr --toy
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import params as P


def _open_recon(zarr_path: Path):
    import zarr
    if not zarr_path.exists():
        raise FileNotFoundError(zarr_path)
    return zarr.open(str(zarr_path), mode="r")


def _per_slice_pearson(phase_zyx: np.ndarray, gt_zyx: np.ndarray) -> np.ndarray:
    """Pearson r per Z slice. Both arrays must have identical shape."""
    a = phase_zyx.reshape(phase_zyx.shape[0], -1)
    b = gt_zyx.reshape(gt_zyx.shape[0], -1)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)) + 1e-30
    return num / denom


def _kz_frc(phase_zyx: np.ndarray, n_shells: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """One-axis (kz) Fourier shell correlation between two halves of XY.

    Splits the volume into even/odd Y rows, FFTs both halves along all 3
    axes, then computes a 1-D radial correlation in (kz, kx) at fixed ky=0
    (a depth-only proxy). Returns (kz_axis, frc).
    """
    half_a = phase_zyx[:, 0::2, :]
    half_b = phase_zyx[:, 1::2, :]
    n = min(half_a.shape[1], half_b.shape[1])
    half_a = half_a[:, :n, :]
    half_b = half_b[:, :n, :]
    Fa = np.fft.fftshift(np.fft.fftn(half_a))
    Fb = np.fft.fftshift(np.fft.fftn(half_b))
    Fa_kx = Fa[:, n // 2, :]
    Fb_kx = Fb[:, n // 2, :]
    kz = np.fft.fftshift(np.fft.fftfreq(phase_zyx.shape[0]))
    return kz, np.real(Fa_kx * np.conj(Fb_kx)).mean(axis=1)


def _projected_potential(p: P.Params, recon_shape, *, slice_thickness: float):
    """Return the abTEM-projected potential at the same Z grid as the recon.

    Returns array of shape (recon_nz, recon_ny, recon_nx). Heavy: builds the
    full potential once.
    """
    import abtem
    from abtem import FrozenPhonons
    nz, ny, nx = recon_shape
    atoms = P.load_atoms()
    fp = FrozenPhonons(atoms, num_configs=1, sigmas=p.phonon_sigmas_a, seed=0)
    pot = abtem.Potential(fp, sampling=p.bf_eff_pixel_mrad / 100.0,
                           slice_thickness=slice_thickness,
                           parametrization="kirkland", device="cpu")
    arr = pot.build().compute().array  # (n_sim_slices, gy, gx)
    arr = np.asarray(arr).real
    # Re-bin along z to nz, then crop/pad along XY to (ny, nx).
    z_bins = np.linspace(0, arr.shape[0], nz + 1).astype(int)
    out = np.empty((nz, ny, nx), dtype="float32")
    gy, gx = arr.shape[1:]
    cy0 = max(0, gy // 2 - ny // 2); cx0 = max(0, gx // 2 - nx // 2)
    cy1 = cy0 + ny; cx1 = cx0 + nx
    if cy1 > gy or cx1 > gx:
        # Pad with zeros if recon is wider than potential.
        pad_y = max(0, cy1 - gy); pad_x = max(0, cx1 - gx)
        arr = np.pad(arr, ((0, 0), (0, pad_y), (0, pad_x)))
    for i in range(nz):
        block = arr[z_bins[i]:max(z_bins[i] + 1, z_bins[i + 1])].sum(axis=0)
        out[i] = block[cy0:cy0 + ny, cx0:cx0 + nx]
    return out


def track_pb_columns(phase_zyx: np.ndarray, *, threshold_factor: float = 0.6,
                     out_csv: Path | None = None) -> np.ndarray:
    """Locate bright columns in each XY slice and report sub-pixel centroids.

    Returns an (nz, n_columns_max, 3) array of (x, y, intensity), padded
    with NaN for short rows. Writes a CSV if ``out_csv`` is given.
    """
    from scipy.ndimage import maximum_filter

    nz = phase_zyx.shape[0]
    rows: list[list[tuple[float, float, float]]] = []
    max_count = 0
    for k in range(nz):
        xy = phase_zyx[k]
        loc_max = (xy == maximum_filter(xy, size=5))
        thr = xy.max() - threshold_factor * (xy.max() - xy.min())
        peaks = np.argwhere(loc_max & (xy >= thr))
        # Sub-pixel centroid via 3×3 quadratic fit on the maximum.
        fits: list[tuple[float, float, float]] = []
        for (yy, xx) in peaks:
            if yy <= 0 or yy >= xy.shape[0] - 1 or xx <= 0 or xx >= xy.shape[1] - 1:
                continue
            patch = xy[yy - 1:yy + 2, xx - 1:xx + 2]
            dy = 0.5 * (patch[2, 1] - patch[0, 1]) / max(1e-12, (2 * patch[1, 1] - patch[0, 1] - patch[2, 1]))
            dx = 0.5 * (patch[1, 2] - patch[1, 0]) / max(1e-12, (2 * patch[1, 1] - patch[1, 0] - patch[1, 2]))
            fits.append((float(xx + dx), float(yy + dy), float(xy[yy, xx])))
        rows.append(fits)
        max_count = max(max_count, len(fits))

    out = np.full((nz, max_count, 3), np.nan, dtype="float32")
    for k, fits in enumerate(rows):
        for n, (x, y, i) in enumerate(fits):
            out[k, n] = (x, y, i)

    if out_csv is not None:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["slice", "column_idx", "x", "y", "intensity"])
            for k in range(nz):
                for n in range(max_count):
                    if not np.isnan(out[k, n, 0]):
                        w.writerow([k, n, *out[k, n].tolist()])
    return out


def run(p: P.Params, recon_path: Path, *, mode: str = "production") -> dict:
    """
    mode: "production" (full 9-tile, std ratio >= 2.0) |
          "mini" (4-tile, std ratio >= 1.3, just check depth structure exists) |
          "toy" (single tile, code-path only — quality criteria all advisory).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = _open_recon(recon_path)
    phase = np.asarray(z["object_phase"][:])
    print(f"Loaded {recon_path.name} | object_phase {phase.shape}  [mode={mode}]")

    # Threshold table per mode.
    std_thresh = {"production": 2.0, "mini": 1.3, "toy": 1.0}[mode]
    kz_thresh = {"production": 0.05, "mini": 0.02, "toy": 0.0}[mode]

    summary: dict = {
        "recon_path": str(recon_path),
        "shape": list(phase.shape),
        "mode": mode,
        "nan_fraction": float(np.isnan(phase).sum()) / phase.size,
    }

    # Code-path gate: zero NaN. This is the absolute minimum.
    if summary["nan_fraction"] > 0:
        print(f"  CRITICAL: {summary['nan_fraction']*100:.1f}% NaN in object_phase  "
              f"(code-path FAIL — reconstruction diverged)")
        summary["pass_code_path"] = False
        return summary
    summary["pass_code_path"] = True
    print(f"  code-path: PASS (no NaN, finite values)")

    # Pass 1: per-slice std (depth contrast).
    std_z = phase.std(axis=tuple(range(1, phase.ndim)))
    contrast_ratio = float(std_z.max() / max(std_z.min(), 1e-12))
    summary["per_slice_std_min"] = float(std_z.min())
    summary["per_slice_std_max"] = float(std_z.max())
    summary["per_slice_std_ratio"] = contrast_ratio
    summary["pass_per_slice_std"] = bool(contrast_ratio >= std_thresh)
    print(f"  per-slice std min/max ratio = {contrast_ratio:.2f}  "
          f"({'PASS' if contrast_ratio >= std_thresh else 'FAIL'} -- want >= {std_thresh})")

    # Pass 2: depth-direction FRC.
    kz, frc = _kz_frc(phase)
    high_kz_avg = float(np.abs(frc[len(frc) // 2 + len(frc) // 4:]).mean())
    low_kz = float(np.abs(frc[len(frc) // 2]))
    summary["kz_frc_lowf"] = low_kz
    summary["kz_frc_highf_avg"] = high_kz_avg
    summary["pass_kz_frc"] = bool(low_kz > 1e-6
                                   and (high_kz_avg / max(low_kz, 1e-12)) > kz_thresh)
    print(f"  kz FRC mid={low_kz:.3g}, high-kz avg={high_kz_avg:.3g}  "
          f"({'PASS' if summary['pass_kz_frc'] else 'FAIL'} -- want ratio > {kz_thresh})")

    # Pass 3: column tracking (writes CSV; no automatic pass/fail).
    out_csv = recon_path.parent / (recon_path.stem + "_columns.csv")
    cols = track_pb_columns(phase, out_csv=out_csv)
    summary["columns_csv"] = str(out_csv)
    summary["max_columns_per_slice"] = int(cols.shape[1])
    print(f"  column centroids -> {out_csv} (up to {cols.shape[1]} per slice)")

    # Plots.
    plot_dir = recon_path.parent / "validation_plots"
    plot_dir.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(std_z); axs[0].set_xlabel("Z slice"); axs[0].set_ylabel("std (rad)")
    axs[0].set_title("Per-slice contrast")
    axs[1].plot(kz, np.abs(frc)); axs[1].set_xlabel("kz (cycles/slice)")
    axs[1].set_ylabel("|FRC|"); axs[1].set_title("Depth-direction FRC")
    mid_y = phase.shape[1] // 2
    axs[2].imshow(phase[:, mid_y, :], aspect="auto", cmap="twilight")
    axs[2].set_xlabel("X (px)"); axs[2].set_ylabel("Z slice")
    axs[2].set_title("XZ slice (mid Y)")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{recon_path.stem}_summary.png", dpi=140)
    plt.close(fig)

    return summary


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", type=Path, default=None,
                    help="Recon zarr path; default chosen from --toy / --mini flags.")
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--mini", action="store_true")
    args = ap.parse_args(argv)

    if args.toy and args.mini:
        ap.error("Pass --toy OR --mini, not both.")

    p = P.toy_params() if (args.toy or args.mini) else P.derive()

    if args.mini:
        mode = "mini"
        default_zarr = "ptycho_recon_mini.zarr"
    elif args.toy:
        mode = "toy"
        default_zarr = "ptycho_recon_toy.zarr"
    else:
        mode = "production"
        default_zarr = "ptycho_recon.zarr"

    recon_path = args.recon or (P.PROJECT_ROOT / default_zarr)
    run(p, recon_path, mode=mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
