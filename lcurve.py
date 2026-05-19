# Run with: C:\Users\Trist\HyperSpy-bundle\python.exe lcurve.py
"""L-curve analysis of the fold_slice Stage B sweep.

After a production run with `eng.save_results_every = 10` and Stage B = 80 iter,
fold_slice has dumped TIFF stacks at iter 10, 20, 30, ..., 80 in:

    1/roi0_Ndp65/MLs_L1_p4_g32_vp1_Ns37_dz<...>/obj_phase_roi_stack/obj_phase_roi_NiterN.tiff

Each TIFF is (37, 403, 403) uint8 — the per-z-slice phase normalised to 0..255
per checkpoint. The point is to find the iteration where:
  * per-slice std (envelope) starts inflating without new structure appearing
  * depth-direction FRC (real depth signal) is highest before noise drowns it
i.e. the knee of the L-curve. We pick that iter and re-run prod with it.

Usage:
    python lcurve.py
    python lcurve.py --stack-dir 1/roi0_Ndp65/<recon_dir>/obj_phase_roi_stack
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _find_default_stack_dir() -> Path:
    """Locate the most recent obj_phase_roi_stack/ under 1/roi0_Ndp65/."""
    base = Path("1") / "roi0_Ndp65"
    if not base.exists():
        raise FileNotFoundError(f"{base} not found — run recon.m first")
    candidates = sorted(base.glob("*/obj_phase_roi_stack"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No obj_phase_roi_stack/ under {base}")
    return candidates[0]


def _iter_n(tiff_path: Path) -> int:
    m = re.search(r"Niter(\d+)\.tiff?$", tiff_path.name)
    if not m:
        raise ValueError(f"can't parse iter number from {tiff_path.name}")
    return int(m.group(1))


def _kz_frc(phase_zyx: np.ndarray) -> tuple[float, float]:
    """Per-pixel kz spectrum: peak (low-kz) and high-kz floor.

    Mirrors validate.py's gate. Phase already mean-subtracted along z.
    """
    p = phase_zyx - phase_zyx.mean(axis=0, keepdims=True)
    # FFT along z, |F| averaged over xy
    F = np.fft.fft(p, axis=0)
    mag = np.abs(F).mean(axis=(1, 2))             # (Nz,)
    mag = np.fft.fftshift(mag)
    nz = mag.size
    centre = nz // 2
    # peak = average of central 5 bins; floor = average of outer 1/3 on each side
    peak = float(mag[centre - 2:centre + 3].mean())
    edge_w = nz // 6
    floor = float(np.concatenate([mag[:edge_w], mag[-edge_w:]]).mean())
    return peak, floor


def _slice_std(phase_zyx: np.ndarray) -> np.ndarray:
    return phase_zyx.std(axis=(1, 2))


def analyse(stack_dir: Path) -> None:
    tiffs = sorted(stack_dir.glob("obj_phase_roi_Niter*.tiff"), key=_iter_n)
    if not tiffs:
        raise FileNotFoundError(f"No obj_phase_roi_Niter*.tiff in {stack_dir}")

    iters: list[int] = []
    std_min: list[float] = []
    std_max: list[float] = []
    std_ratio: list[float] = []
    frc_ratio: list[float] = []

    print(f"L-curve over {len(tiffs)} checkpoints in {stack_dir}\n")
    print(f"{'Niter':>6} | {'std min':>8} {'std max':>8} {'ratio':>6} | "
          f"{'kz peak':>9} {'kz floor':>9} {'ratio':>6}")
    print("-" * 70)

    for tp in tiffs:
        n = _iter_n(tp)
        arr = tifffile.imread(tp).astype("float32")     # (z, y, x), uint8 -> float
        # Map uint8 0..255 to [-0.5, 0.5] — relative units, doesn't matter for ratios
        arr = (arr - 127.5) / 255.0

        sz = _slice_std(arr)
        std_min.append(float(sz.min()))
        std_max.append(float(sz.max()))
        std_ratio.append(std_max[-1] / max(std_min[-1], 1e-12))

        peak, floor = _kz_frc(arr)
        frc_ratio.append(floor / max(peak, 1e-12))

        iters.append(n)
        print(f"{n:>6d} | {std_min[-1]:>8.4f} {std_max[-1]:>8.4f} "
              f"{std_ratio[-1]:>6.2f} | {peak:>9.4f} {floor:>9.4f} "
              f"{frc_ratio[-1]:>6.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(iters, std_min, "o-", label="min")
    axes[0].plot(iters, std_max, "o-", label="max")
    axes[0].set_xlabel("Stage B iteration")
    axes[0].set_ylabel("per-slice std (uint8-relative)")
    axes[0].set_title("Envelope growth")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(iters, std_ratio, "o-")
    axes[1].axhline(2.0, color="grey", linestyle=":", label="validate gate")
    axes[1].set_xlabel("Stage B iteration")
    axes[1].set_ylabel("std max / std min")
    axes[1].set_title("Per-slice contrast ratio")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(iters, frc_ratio, "o-")
    axes[2].axhline(0.05, color="grey", linestyle=":", label="validate gate")
    axes[2].set_xlabel("Stage B iteration")
    axes[2].set_ylabel("kz-FRC high/low ratio")
    axes[2].set_title("Depth-direction structure")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    out_dir = Path("validation_plots")
    out_dir.mkdir(exist_ok=True)
    out_png = out_dir / "lcurve.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"\nWrote {out_png}")

    # Knee detection: kz-FRC ratio peaks then drops as noise wins
    knee_ix = int(np.argmax(frc_ratio))
    print(f"\nkz-FRC ratio peaks at iter {iters[knee_ix]} "
          f"(ratio={frc_ratio[knee_ix]:.4f}). "
          f"This is the candidate Niter for production.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stack-dir", type=Path, default=None,
                    help="obj_phase_roi_stack/ dir (default: auto-find most recent)")
    args = ap.parse_args()
    stack_dir = args.stack_dir or _find_default_stack_dir()
    analyse(stack_dir)


if __name__ == "__main__":
    main()
