# Run with: C:\Users\Trist\HyperSpy-bundle\python.exe supervisor_plot.py
"""Three-way comparison for the supervisor:
    py4DSTEM (failed engine)  vs  fold_slice smoke (3+3 iter)  vs  fold_slice latest (40+80 iter)

Each column shows:
    * XZ-deviation panel (= the "real depth signal" panel from validate.py)
    * One XY slice at z ≈ middle of sample (atomic columns)
    * kz-FRC ratio + std-ratio annotations

Plus a bottom strip overlaying all three kz-FRC spectra.

The story this tells:
  - py4DSTEM produces a smooth anti-symmetric envelope (= projection only, no real
    depth signal). XY columns ARE sharp but never change across slices.
  - fold_slice smoke shows the engine + data CAN extract atomic columns cleanly
    (XY panel is the crispest), but at 3+3 iter the depth signal hasn't developed.
  - fold_slice latest (40+80 iter) DOES build depth structure (XZ-deviation has
    column-localised vertical streaks, kz-FRC ratio gains 9000x over py4DSTEM)
    but extra iter amplifies noise on the under-determined recon.
  → next step (Phase C): cell extension + 20 nm overfocus to get BOTH cleanliness
    AND depth signal in the same run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RUNS = [
    # (label, zarr_path, color, short)
    ("py4DSTEM\n(gradient descent)",                "ptycho_recon.zarr",                  "#1f77b4", "py4DSTEM"),
    ("fold_slice smoke\n(3+3 iter — engine warm-up)", "ptycho_recon_foldslice_smoke_nopos.zarr", "#ff7f0e", "smoke"),
    ("fold_slice production\n(40+80 iter — latest)",   "ptycho_recon_foldslice_sweep3.zarr",     "#d62728", "sweep3"),
]


def _load_phase(zarr_path: Path) -> np.ndarray:
    z = zarr.open(str(zarr_path), mode="r")
    return np.asarray(z["object_phase"][:])     # (Nz, Ny, Nx) float32


def _xz_deviation_strip(phase: np.ndarray, strip_a: float = 2.6,
                        pixel_a: float = 0.05) -> np.ndarray:
    Ny = phase.shape[1]
    cy = Ny // 2
    half_px = int(round(strip_a / 2 / pixel_a))
    strip = phase[:, cy - half_px:cy + half_px, :].mean(axis=1)
    return strip - strip.mean(axis=0, keepdims=True)


def _kz_frc(phase_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """validate.py-matched kz-FRC. Returns (kz, |frc|, ratio_high_over_low, std_ratio)."""
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
    frc = np.real(Fa_kx * np.conj(Fb_kx)).mean(axis=1)
    high_kz_avg = float(np.abs(frc[len(frc) // 2 + len(frc) // 4:]).mean())
    low_kz = float(np.abs(frc[len(frc) // 2]))
    sz = phase_zyx.std(axis=(1, 2))
    return kz, np.abs(frc), high_kz_avg / max(low_kz, 1e-12), float(sz.max() / max(sz.min(), 1e-12))


def main():
    pixel_a = 0.05
    z_target_a = 14.0   # middle-ish XY slice

    data = []
    for label, path, color, short in RUNS:
        phase = _load_phase(Path(path))
        xz = _xz_deviation_strip(phase, pixel_a=pixel_a)
        kz, mag, kz_ratio, std_ratio = _kz_frc(phase)
        Nz = phase.shape[0]
        # Map z=14 Å → slice index given each run's slice thickness
        dz_a = 74.0 / Nz
        zi = int(round(z_target_a / dz_a))
        zi = max(0, min(zi, Nz - 1))
        xy = phase[zi]
        data.append({
            "label": label, "color": color, "short": short,
            "phase": phase, "xz": xz, "xy": xy, "kz": kz, "mag": mag,
            "kz_ratio": kz_ratio, "std_ratio": std_ratio,
            "z_used_a": zi * dz_a, "Nz": Nz, "dz_a": dz_a,
        })
        print(f"{short:>10}  kz-FRC={kz_ratio:.3e}  std-ratio={std_ratio:.2f}  "
              f"shape={phase.shape}  xy@z={zi*dz_a:.1f}A")

    # Layout: 3 columns × (XZ row + XY row), kz spectra at bottom spanning all columns.
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.4, 2.4, 1.4],
                          hspace=0.45, wspace=0.30)

    for i, d in enumerate(data):
        # --- XZ-deviation panel ---
        ax = fig.add_subplot(gs[0, i])
        vmax = float(np.percentile(np.abs(d["xz"]), 99))
        thickness = d["Nz"] * d["dz_a"]
        extent = [0, d["xz"].shape[1] * 0.05, thickness, 0]
        im = ax.imshow(d["xz"], aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, extent=extent)
        ax.set_title(d["label"], fontsize=11)
        ax.set_xlabel("X (Å)"); ax.set_ylabel("z (Å, beam ↓)")
        plt.colorbar(im, ax=ax, fraction=0.04, label="Δphase (rad)")
        ax.text(0.02, 0.98,
                f"kz-FRC = {d['kz_ratio']:.2e}\nstd-ratio = {d['std_ratio']:.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85))

        # --- XY slice (raw phase) at z ≈ 14 Å ---
        ax2 = fig.add_subplot(gs[1, i])
        xy = d["xy"]
        vmax2 = float(np.percentile(np.abs(xy - xy.mean()), 99))
        ny, nx = xy.shape
        extent2 = [0, nx * 0.05, ny * 0.05, 0]
        im2 = ax2.imshow(xy - xy.mean(), cmap="RdBu_r",
                         vmin=-vmax2, vmax=vmax2, extent=extent2)
        ax2.set_title(f"XY slice at z ≈ {d['z_used_a']:.1f} Å "
                      f"(atomic columns)", fontsize=10)
        ax2.set_xlabel("X (Å)"); ax2.set_ylabel("Y (Å)")
        plt.colorbar(im2, ax=ax2, fraction=0.04, label="phase − mean (rad)")

    # --- Bottom: kz-FRC spectra overlay ---
    ax_kz = fig.add_subplot(gs[2, :])
    for d in data:
        ax_kz.semilogy(d["kz"], d["mag"] / d["mag"].max(),
                       label=f"{d['short']}  (kz-FRC ratio = {d['kz_ratio']:.2e})",
                       color=d["color"], lw=1.8)
    ax_kz.set_xlabel("kz (cycles per slice)")
    ax_kz.set_ylabel("|FRC| / max  (log)")
    ax_kz.set_title("Depth-direction FRC — sharp central peak = projection only; "
                    "broad distribution = real depth signal", fontsize=10)
    ax_kz.grid(alpha=0.3, which="both")
    ax_kz.legend(loc="lower center", fontsize=9, ncol=3)

    # Headline
    py_ratio = data[0]["kz_ratio"]
    fs_latest = data[-1]["kz_ratio"]
    fig.suptitle(
        f"Engine port: same data, py4DSTEM → fold_slice  —  "
        f"kz-FRC ratio improves {fs_latest/py_ratio:.0f}× "
        f"({py_ratio:.1e} → {fs_latest:.3f}, validate.py gate ≥ 0.05)",
        fontsize=13, weight="bold", y=0.995,
    )

    out = Path("validation_plots") / "supervisor_engine_comparison.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
