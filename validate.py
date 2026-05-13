"""Quantitative validation of a multislice ptychography reconstruction.

Checks on the reconstructed object_phase volume (``ptycho_recon*.zarr``):

* NaN gate (any NaN ⇒ reconstruction diverged ⇒ hard fail).
* Per-slice std min/max ratio (= depth-contrast envelope; weak test — passes
  on a projection that has a smooth reconstruction-confidence envelope).
* Mean Fourier shell correlation along kz (depth-direction FRC) — the
  meaningful test for "is there real depth structure?".
* ``track_pb_columns`` — sub-pixel column centroids per slice, written to
  CSV for downstream comparison against the POSCAR displacements.

Plotting: a strip-averaged XZ/YZ summary (avg over ±1/3 perovskite UC in the
orthogonal in-plane axis) + an XY-slice montage at evenly-spaced z, with a
"phase − z-mean" row that's flat under a pure projection and shows any real
depth modulation as a column-aligned pattern.

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
    _save_plots(phase, std_z, kz, frc, z.attrs.asdict() if hasattr(z.attrs, "asdict")
                else dict(z.attrs), recon_path, plot_dir)

    return summary


# PTO/STO perovskite a ≈ 3.905 Å — we average the depth-section view over ±1/3
# UC in the orthogonal in-plane direction (= 2/3 UC total, ~2.6 Å). That window
# spans one full row of B-site columns even with mild rotation/strain, so any
# real depth structure shows up as a column-aligned feature instead of being
# washed out by an off-column slice or a 1-pixel cut.
_UC_A = 3.905
_STRIP_FRACTION_OF_UC = 2.0 / 3.0


def _recon_pixel_size_a(zarr_attrs: dict, phase_shape, p: P.Params) -> float:
    """Read the recon real-space pixel size from zarr attrs; fall back to
    scan_extent / object_grid as a sanity estimate."""
    px = zarr_attrs.get("reconstruction_pixel_size_a")
    if px and px > 0:
        return float(px)
    # Fallback: object grid spans (scan + a few Å of probe extent) on each side.
    return float((p.scan_window_a + 8.0) / phase_shape[1])


def _strip_avg(vol_zyx: np.ndarray, axis: str, strip_px: int) -> np.ndarray:
    """Strip-average along the orthogonal in-plane axis.

    ``axis='xz'`` averages over Y in a ±strip_px window around mid-Y, returns (z, x).
    ``axis='yz'`` averages over X similarly, returns (z, y).
    """
    nz, ny, nx = vol_zyx.shape
    if axis == "xz":
        y0 = ny // 2
        lo, hi = max(0, y0 - strip_px), min(ny, y0 + strip_px + 1)
        return vol_zyx[:, lo:hi, :].mean(axis=1)
    elif axis == "yz":
        x0 = nx // 2
        lo, hi = max(0, x0 - strip_px), min(nx, x0 + strip_px + 1)
        return vol_zyx[:, :, lo:hi].mean(axis=2)
    raise ValueError(axis)


def _save_plots(phase: np.ndarray, std_z: np.ndarray, kz: np.ndarray,
                frc: np.ndarray, zarr_attrs: dict, recon_path: Path,
                plot_dir: Path) -> None:
    """Two figures: (1) summary with per-slice std + kz-FRC + strip-averaged
    XZ/YZ depth views; (2) XY-slice montage at evenly-spaced z, with the same
    slices minus the z-mean (= deviation from projection) shown below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Need params for the fallback pixel-size estimate.
    p = P.derive(box_dims_a=(47.88, 70.01, 73.93)) if False else None  # not needed
    px_a = float(zarr_attrs.get("reconstruction_pixel_size_a") or 0.05)
    strip_px = max(1, int(round(_UC_A * _STRIP_FRACTION_OF_UC / 2.0 / px_a)))
    nz, ny, nx = phase.shape
    z_mean = phase.mean(axis=0)        # the "projection"
    dev = phase - z_mean[None, :, :]   # depth-varying part

    # ---- Figure 1: summary (per-slice std / kz-FRC / strip-averaged XZ & YZ) ----
    xz_strip = _strip_avg(phase, "xz", strip_px)
    yz_strip = _strip_avg(phase, "yz", strip_px)
    xz_dev = _strip_avg(dev,   "xz", strip_px)
    yz_dev = _strip_avg(dev,   "yz", strip_px)

    z_a = np.arange(nz)               # slice index → ~Å (recon slice thickness ≈ 1 Å)
    extent_xz = [0, nx * px_a, nz, 0]
    extent_yz = [0, ny * px_a, nz, 0]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))

    axs[0, 0].plot(z_a, std_z)
    axs[0, 0].set_xlabel("Z slice (≈ Å)"); axs[0, 0].set_ylabel("std (rad)")
    axs[0, 0].set_title("Per-slice contrast (= envelope, NOT depth signal)")

    axs[0, 1].semilogy(kz, np.abs(frc) + 1e-12)
    axs[0, 1].set_xlabel("kz (cycles/slice)"); axs[0, 1].set_ylabel("|FRC| (log)")
    axs[0, 1].set_title("Depth-direction FRC (want broad, not a delta)")

    # Depth-contrast map: per-pixel std across z. Flat = projection.
    depth_contrast = phase.std(axis=0)
    im = axs[0, 2].imshow(depth_contrast,
                          extent=[0, nx * px_a, ny * px_a, 0],
                          cmap="magma")
    axs[0, 2].set_xlabel("X (Å)"); axs[0, 2].set_ylabel("Y (Å)")
    axs[0, 2].set_title("Per-pixel std along Z\n(would show columns if depth-varying)")
    fig.colorbar(im, ax=axs[0, 2], fraction=0.046)

    vmax = max(abs(xz_strip).max(), abs(yz_strip).max())
    im = axs[1, 0].imshow(xz_strip, extent=extent_xz, aspect="auto",
                          cmap="twilight", vmin=-vmax, vmax=vmax)
    axs[1, 0].set_xlabel("X (Å)"); axs[1, 0].set_ylabel("Z slice")
    axs[1, 0].set_title(f"XZ — phase, avg over ±{strip_px*px_a:.1f} Å in Y "
                        f"(={_STRIP_FRACTION_OF_UC:.2f} UC)")
    fig.colorbar(im, ax=axs[1, 0], fraction=0.046)

    im = axs[1, 1].imshow(yz_strip, extent=extent_yz, aspect="auto",
                          cmap="twilight", vmin=-vmax, vmax=vmax)
    axs[1, 1].set_xlabel("Y (Å)"); axs[1, 1].set_ylabel("Z slice")
    axs[1, 1].set_title(f"YZ — phase, avg over ±{strip_px*px_a:.1f} Å in X")
    fig.colorbar(im, ax=axs[1, 1], fraction=0.046)

    # XZ of (phase − z_mean) — pure projection ⇒ zero. Diverging cmap.
    vmax_d = max(abs(xz_dev).max(), abs(yz_dev).max(), 1e-12)
    im = axs[1, 2].imshow(xz_dev, extent=extent_xz, aspect="auto",
                          cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d)
    axs[1, 2].set_xlabel("X (Å)"); axs[1, 2].set_ylabel("Z slice")
    axs[1, 2].set_title(f"XZ — (phase − z-mean), same strip\n"
                        f"flat ⇒ projection;  pattern ⇒ real depth structure")
    fig.colorbar(im, ax=axs[1, 2], fraction=0.046)

    fig.suptitle(f"{recon_path.name}   |   object {phase.shape}   |   "
                 f"recon px ≈ {px_a:.3f} Å   |   strip = 2/3 UC")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{recon_path.stem}_summary.png", dpi=140)
    plt.close(fig)

    # ---- Figure 2: XY slice montage (raw phase top row, dev-from-projection bottom) ----
    n_slices = 6
    # Skip the very edge slices (the envelope kills contrast there).
    z_idx = np.linspace(int(0.05 * nz), int(0.95 * nz) - 1, n_slices).astype(int)
    fig, axs = plt.subplots(2, n_slices, figsize=(3.0 * n_slices, 6.5),
                            sharex=True, sharey=True)
    vmax_p = float(np.percentile(np.abs(phase), 99))
    vmax_d_xy = float(np.percentile(np.abs(dev), 99)) or 1e-12
    for col, k in enumerate(z_idx):
        axs[0, col].imshow(phase[k], cmap="twilight", vmin=-vmax_p, vmax=vmax_p,
                           extent=[0, nx * px_a, ny * px_a, 0])
        axs[0, col].set_title(f"z = {k} (≈{k:.0f} Å)")
        axs[1, col].imshow(dev[k], cmap="RdBu_r", vmin=-vmax_d_xy, vmax=vmax_d_xy,
                           extent=[0, nx * px_a, ny * px_a, 0])
    axs[0, 0].set_ylabel("phase  (rad)\nY (Å)")
    axs[1, 0].set_ylabel("phase − z-mean\nY (Å)")
    for ax in axs[1, :]:
        ax.set_xlabel("X (Å)")
    fig.suptitle(f"{recon_path.name}  —  XY slices at evenly-spaced z\n"
                 f"top: raw phase;  bottom: deviation from z-mean (= real depth signal)")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{recon_path.stem}_slices.png", dpi=140)
    plt.close(fig)

    print(f"  plots -> {plot_dir / (recon_path.stem + '_summary.png')}")
    print(f"           {plot_dir / (recon_path.stem + '_slices.png')}")


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
