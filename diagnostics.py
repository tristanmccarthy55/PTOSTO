# Run with: C:\Users\Trist\HyperSpy-bundle\python.exe diagnostics.py
"""Phase 0 diagnostics for the PTO/STO 4D-STEM -> multislice-ptychography pipeline.

Read-only. Runs each sub-test, prints a PASS/FAIL table, and writes a
``diagnostics_report.md`` next to this file plus a few PNG plots into
``diagnostics_plots/``.

Usage:
    python diagnostics.py                  # all checks
    python diagnostics.py --only 0a 0e     # subset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Constants from the existing notebook (Cell 0). Single source of truth here so
# the diagnostics script does not depend on importing the notebook.
# ---------------------------------------------------------------------------
ENERGY_EV = 300e3
CONVERGENCE_MRAD = 100.0
OVERFOCUS_NM = 1.0
OVERFOCUS_A = OVERFOCUS_NM * 10.0
HAADF_INNER = 110.0
HAADF_OUTER = 250.0
BF_MAX_ANGLE = 250.0  # outer angle the existing PixelatedDetector uses
TARGET_DETECTOR_PIXELS = 64
TARGET_OVERLAP = 0.75
SCAN_WIDTH_A = 40.0
TILE_SIZE_A = 4.0
CENTER_X_A = 23.0
CENTER_Y_A = 35.0

PROJECT_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = PROJECT_ROOT.parent
STRUCTURE_FILE = PROJECT_ROOT / "PTO6_STO6_18_18_labyrinthPoscar.vasp"
EXISTING_RECON_NPZ = PARENT_ROOT / "ptycho_3d_reconstruction.npz"

PLOTS_DIR = PROJECT_ROOT / "diagnostics_plots"
REPORT_PATH = PROJECT_ROOT / "diagnostics_report.md"


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------
@dataclass
class Check:
    label: str
    value: str
    verdict: str  # PASS / WARN / FAIL / INFO
    note: str = ""


@dataclass
class Section:
    name: str
    checks: list[Check] = field(default_factory=list)
    body: str = ""

    def add(self, label: str, value: str, verdict: str, note: str = "") -> None:
        self.checks.append(Check(label, value, verdict, note))

    def render(self) -> str:
        lines = [f"## {self.name}", ""]
        if self.checks:
            lines.append("| Check | Value | Verdict | Note |")
            lines.append("|---|---|---|---|")
            for c in self.checks:
                lines.append(f"| {c.label} | {c.value} | **{c.verdict}** | {c.note} |")
            lines.append("")
        if self.body:
            lines.append(self.body)
            lines.append("")
        return "\n".join(lines)


SECTIONS: list[Section] = []


def section(name: str) -> Section:
    s = Section(name)
    SECTIONS.append(s)
    return s


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------
def relativistic_wavelength_a(energy_ev: float) -> float:
    """Return the relativistic electron wavelength in angstroms."""
    h = 6.62607015e-34
    m = 9.1093837015e-31
    e = 1.602176634e-19
    c = 299_792_458.0
    p = np.sqrt(2.0 * m * energy_ev * e * (1.0 + energy_ev * e / (2.0 * m * c * c)))
    return float(h / p * 1e10)


WAVELENGTH_A = relativistic_wavelength_a(ENERGY_EV)


# ---------------------------------------------------------------------------
# 0a -- Geometry & beam direction
# ---------------------------------------------------------------------------
def check_0a_geometry() -> None:
    s = section("0a — Geometry & beam direction")

    if not STRUCTURE_FILE.exists():
        s.add("Structure file", str(STRUCTURE_FILE), "FAIL",
              "Cannot find POSCAR; downstream checks skipped.")
        return

    try:
        import ase.io
        import abtem
    except ImportError as exc:  # pragma: no cover -- defensive
        s.add("ASE/abTEM import", str(exc), "FAIL",
              "Activate the conda env that has hyperspy + abtem.")
        return

    atoms_raw = ase.io.read(str(STRUCTURE_FILE))
    species_counts = {}
    for sym in atoms_raw.get_chemical_symbols():
        species_counts[sym] = species_counts.get(sym, 0) + 1
    raw_lengths = atoms_raw.cell.lengths()
    s.add("POSCAR atoms", f"{len(atoms_raw)}", "INFO",
          ", ".join(f"{k}={v}" for k, v in sorted(species_counts.items())))
    s.add("Pre-rotate cell", f"{raw_lengths[0]:.3f} × {raw_lengths[1]:.3f} × {raw_lengths[2]:.3f} Å", "INFO")

    atoms = atoms_raw.copy()
    atoms.rotate(-90, "y", rotate_cell=True)
    atoms = abtem.orthogonalize_cell(atoms)
    atoms.center(axis=2, vacuum=2.0)
    L = atoms.cell.lengths()
    angles = atoms.cell.angles()
    s.add("Post-pipeline cell",
          f"{L[0]:.3f} × {L[1]:.3f} × {L[2]:.3f} Å",
          "INFO",
          f"angles {angles[0]:.1f},{angles[1]:.1f},{angles[2]:.1f}")
    s.add("Atom count after orthogonalize", f"{len(atoms)}", "INFO")

    # Candidate "thickness along beam" for each axis convention.
    thickness_candidates = {"X": L[0], "Y": L[1], "Z": L[2]}
    for axis, val in thickness_candidates.items():
        s.add(f"Candidate thickness if beam=={axis}", f"{val:.3f} Å", "INFO")

    # Cross-check against existing recon's metadata if available.
    if EXISTING_RECON_NPZ.exists():
        try:
            with np.load(EXISTING_RECON_NPZ, allow_pickle=True) as data:
                if "real_space_thickness" in data.files:
                    rst = float(data["real_space_thickness"])
                    closest_axis = min(thickness_candidates,
                                       key=lambda k: abs(thickness_candidates[k] - rst))
                    closest_val = thickness_candidates[closest_axis]
                    delta = abs(rst - closest_val)
                    verdict = "PASS" if delta < 0.05 else "WARN"
                    s.add("Existing recon real_space_thickness",
                          f"{rst:.3f} Å",
                          verdict,
                          f"closest to axis {closest_axis} ({closest_val:.3f} Å, Δ={delta:.3f})")
                    if closest_axis != "Z":
                        s.add("Beam-axis convention",
                              f"existing recon used axis {closest_axis}, abTEM beam=Z",
                              "FAIL",
                              "Set REAL_SPACE_THICKNESS = atoms.cell.lengths()[2] in params.py")
                    else:
                        s.add("Beam-axis convention", "axis Z (matches abTEM)", "PASS")
        except Exception as exc:  # pragma: no cover -- defensive
            s.add("Existing recon load", str(exc), "WARN")
    else:
        s.add("Existing recon npz", "not found", "WARN",
              "Cannot cross-check thickness without the prior result.")

    # --- Vortex axis from atomic displacements ---
    # Pb columns lie at one Wyckoff site; in a polar-vortex labyrinth their
    # in-plane displacement vectors curl about the vortex axis. Whichever axis
    # we project ALONG (sum) shows tight Pb dots (the vortex is along that
    # axis); axes perpendicular to it show the swirl.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        s.add("matplotlib", "missing", "WARN", "Plots skipped.")
        return

    PLOTS_DIR.mkdir(exist_ok=True)
    pos = atoms.get_positions()
    syms = np.array(atoms.get_chemical_symbols())
    pb_mask = syms == "Pb"
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax_idx, (proj_axis, name) in enumerate(zip([0, 1, 2], ["X", "Y", "Z"])):
        keep = [i for i in range(3) if i != proj_axis]
        a, b = keep
        axs[ax_idx].scatter(pos[~pb_mask, a], pos[~pb_mask, b],
                            s=0.1, c="lightgrey", rasterized=True)
        axs[ax_idx].scatter(pos[pb_mask, a], pos[pb_mask, b],
                            s=2, c="black", rasterized=True)
        axs[ax_idx].set_xlabel(f"{['X','Y','Z'][a]} (Å)")
        axs[ax_idx].set_ylabel(f"{['X','Y','Z'][b]} (Å)")
        axs[ax_idx].set_title(f"Projection ALONG {name}")
        axs[ax_idx].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "0a_projections.png", dpi=140)
    plt.close(fig)
    s.body = ("![projections](diagnostics_plots/0a_projections.png)\n\n"
              "Inspect: the projection along the vortex axis shows Pb columns "
              "as tight dots; perpendicular projections show the swirl. "
              "Currently abTEM beam = +Z (third panel).")


# ---------------------------------------------------------------------------
# 0b -- Parameter sanity sums
# ---------------------------------------------------------------------------
def check_0b_params() -> None:
    s = section("0b — Parameter sanity sums")

    lam = WAVELENGTH_A
    alpha = CONVERGENCE_MRAD * 1e-3
    expected_lam = (0.0197, 0.0197)  # narrow window for 300 keV
    s.add("WAVELENGTH (300 keV)",
          f"{lam:.4f} Å",
          "PASS" if 0.019 < lam < 0.020 else "FAIL",
          "h / sqrt(2 m E (1 + E/(2 m c²)))")

    dz_optical = lam / (alpha ** 2)
    s.add("Depth resolution dz = λ/α²",
          f"{dz_optical:.2f} Å",
          "PASS" if dz_optical < 4.0 else "FAIL",
          f"PTO unit cell ≈ 3.9 Å, sub-UC achievable iff dz < 3.9 Å")

    fwhm_focused = 0.61 * lam / alpha
    s.add("PROBE_FWHM_FOCUSED (Airy)",
          f"{fwhm_focused:.3f} Å",
          "PASS" if 0.10 < fwhm_focused < 0.14 else "WARN",
          "0.61 · λ / α")

    geom = alpha * OVERFOCUS_A
    s.add("GEOMETRIC_SPREAD at overfocus",
          f"{geom:.3f} Å",
          "INFO",
          "α · Δf with Δf = 10 Å overfocus")

    fwhm_eff = float(np.sqrt(fwhm_focused ** 2 + geom ** 2))
    s.add("PROBE_FWHM_EFFECTIVE",
          f"{fwhm_eff:.3f} Å",
          "INFO",
          "quadrature sum of focused FWHM and geometric spread")

    scan_step = (1.0 - TARGET_OVERLAP) * fwhm_eff
    s.add("SCAN_STEP",
          f"{scan_step:.3f} Å",
          "PASS" if scan_step < 0.5 else "WARN",
          f"(1 − {TARGET_OVERLAP}) · FWHM_eff → {TARGET_OVERLAP*100:.0f}% overlap")

    # Slicing for the simulation: λ/(2*max_angle) is a safe upper bound, but
    # for 100 mrad we can typically use 1 Å with no loss.
    slice_thickness_recommended = max(0.5, lam / (2.0 * np.sin(BF_MAX_ANGLE * 1e-3)))
    s.add("Recommended sim SLICE_THICKNESS",
          f"{slice_thickness_recommended:.3f} Å",
          "INFO",
          "Existing 0.31 Å is over-fine; 0.8–1.0 Å is sufficient for 100 mrad.")

    s.add("VRAM cap (user)",
          "8 GB",
          "INFO",
          "Drives detector-size and batch caps (see 0e).")


# ---------------------------------------------------------------------------
# 0c -- Phonon sigma literature cross-check
# ---------------------------------------------------------------------------
def check_0c_phonons() -> None:
    s = section("0c — Phonon σ literature cross-check (300 K, isotropic)")

    # Sigma values used in the proposed FrozenPhonons call. These are starting
    # numbers; user should verify against PTO/STO Debye-Waller measurements.
    proposed = {"Pb": 0.092, "Sr": 0.085, "Ti": 0.072, "O": 0.085}

    # Reference range from neutron / X-ray PTO/STO literature, conservative.
    # Sources (consult before relying on these numbers):
    #   - Sani et al., Phys Rev B 65 214104 (2002): PbTiO3 DWFs
    #   - Abramov et al., Acta Cryst B51 942 (1995): SrTiO3 DWFs
    reference = {
        "Pb": (0.080, 0.110),
        "Sr": (0.075, 0.100),
        "Ti": (0.060, 0.085),
        "O":  (0.075, 0.100),
    }
    for sp, val in proposed.items():
        lo, hi = reference[sp]
        verdict = "PASS" if lo <= val <= hi else "WARN"
        s.add(f"σ({sp})", f"{val:.3f} Å", verdict, f"literature range {lo:.3f}–{hi:.3f} Å")
    s.body = ("Reminder: sigmas are placeholders. Confirm against the user's "
              "preferred Debye-Waller table or DFPT phonon DOS before the production run.")


# ---------------------------------------------------------------------------
# 0d -- Existing zarr + npz inspection
# ---------------------------------------------------------------------------
def check_0d_existing() -> None:
    s = section("0d — Existing-data inspection")

    try:
        import zarr
    except ImportError:
        s.add("zarr import", "missing", "FAIL")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None  # type: ignore

    PLOTS_DIR.mkdir(exist_ok=True)

    # Find any existing _bf.zarr in the parent dir or this dir.
    candidates = sorted(list(PARENT_ROOT.glob("tile*_bf.zarr"))
                        + list(PROJECT_ROOT.glob("tile*_bf.zarr")))
    if not candidates:
        s.add("BF tile zarrs", "none found", "WARN",
              "Phase 0d/1 plots skipped; existing sim outputs not on disk.")
    else:
        sample = candidates[len(candidates) // 2]
        try:
            arr = zarr.open(str(sample), mode="r")
            shp = tuple(arr.shape)
            s.add(f"Sample tile {sample.name}",
                  f"shape {shp}, dtype {arr.dtype}",
                  "INFO")
            if len(shp) == 4:
                # Sum over scan to get the average CBED.
                mean_cbed = np.array(arr[:]).mean(axis=(0, 1))
                # Disk centre and effective pix-mrad estimated from disk radius.
                ny, nx = mean_cbed.shape
                cy, cx = ny // 2, nx // 2
                # Azimuthal profile (rough -- uses isotropic radius in pixels).
                yy, xx = np.indices(mean_cbed.shape)
                rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
                radial = np.bincount(rr.ravel(), weights=mean_cbed.ravel()) / np.maximum(np.bincount(rr.ravel()), 1)
                # Disk edge: largest derivative of radial profile.
                deriv = np.diff(radial)
                disk_radius_px = int(np.argmin(deriv) + 1) if deriv.size else 0
                s.add("Average BF disk radius (pixels)",
                      f"{disk_radius_px}",
                      "PASS" if disk_radius_px >= 8 else "WARN",
                      "≥8 px diameter required for ePIE updates")
                if plt is not None:
                    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
                    axs[0].imshow(mean_cbed, cmap="magma")
                    axs[0].set_title(f"Mean CBED, {ny}×{nx}")
                    axs[1].plot(radial)
                    axs[1].axvline(disk_radius_px, color="red", linestyle="--",
                                   label=f"r={disk_radius_px}")
                    axs[1].set_yscale("log")
                    axs[1].legend()
                    axs[1].set_title("Azimuthal profile")
                    one_frame = np.array(arr[shp[0] // 2, shp[1] // 2])
                    axs[2].imshow(one_frame, cmap="magma")
                    axs[2].set_title(f"Single CBED at scan ({shp[0]//2},{shp[1]//2})")
                    fig.tight_layout()
                    fig.savefig(PLOTS_DIR / "0d_existing_bf.png", dpi=140)
                    plt.close(fig)
        except Exception as exc:  # pragma: no cover -- defensive
            s.add("BF tile read", str(exc), "FAIL")

    if EXISTING_RECON_NPZ.exists():
        try:
            with np.load(EXISTING_RECON_NPZ, allow_pickle=True) as d:
                files = list(d.files)
                s.add("Existing recon npz keys",
                      ", ".join(files[:8]) + ("…" if len(files) > 8 else ""),
                      "INFO")
                if "object_phase" in files:
                    phase = d["object_phase"]
                    s.add("object_phase shape",
                          f"{phase.shape}",
                          "INFO")
                    std_z = phase.std(axis=tuple(range(1, phase.ndim)))
                    ratio = float(std_z.max() / max(std_z.min(), 1e-12))
                    s.add("XY-std ratio across Z slices",
                          f"{ratio:.2f}",
                          "PASS" if ratio > 2.0 else "FAIL",
                          "want >2× contrast between sample and vacuum")
                    # |FFT_z|: power vs kz.
                    fz = np.abs(np.fft.rfft(phase, axis=0)).mean(axis=tuple(range(1, phase.ndim)))
                    if fz.size > 1:
                        ratio_kz = float(fz[1:].max() / max(fz[0], 1e-12))
                        s.add("FFT_z power away from kz=0",
                              f"{ratio_kz:.2f}× of DC",
                              "PASS" if ratio_kz > 0.05 else "FAIL",
                              "FAIL means reconstruction has no axial structure")
                    if plt is not None:
                        PLOTS_DIR.mkdir(exist_ok=True)
                        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
                        axs[0].plot(std_z)
                        axs[0].set_xlabel("Z slice index")
                        axs[0].set_ylabel("phase std (rad)")
                        axs[0].set_title("Per-slice contrast")
                        axs[1].plot(fz)
                        axs[1].set_yscale("log")
                        axs[1].set_xlabel("kz bin")
                        axs[1].set_title("Mean |FFT_z(phase)|")
                        # XZ slice through phase volume's middle row.
                        mid_y = phase.shape[1] // 2
                        xz = phase[:, mid_y, :]
                        axs[2].imshow(xz, aspect="auto", cmap="twilight")
                        axs[2].set_xlabel("X (px)")
                        axs[2].set_ylabel("Z slice")
                        axs[2].set_title("XZ slice (mid Y)")
                        fig.tight_layout()
                        fig.savefig(PLOTS_DIR / "0d_existing_recon.png", dpi=140)
                        plt.close(fig)
        except Exception as exc:  # pragma: no cover -- defensive
            s.add("recon npz read", str(exc), "FAIL")
    else:
        s.add("ptycho_3d_reconstruction.npz", "not present", "INFO",
              "Skip existing-recon plots if first run.")


# ---------------------------------------------------------------------------
# 0e -- Detector design study
# ---------------------------------------------------------------------------
def check_0e_detector(box_dims_a: tuple[float, float, float] | None = None) -> None:
    s = section("0e — Detector design study")

    if box_dims_a is None:
        try:
            import ase.io
            import abtem
            atoms = ase.io.read(str(STRUCTURE_FILE))
            atoms.rotate(-90, "y", rotate_cell=True)
            atoms = abtem.orthogonalize_cell(atoms)
            atoms.center(axis=2, vacuum=2.0)
            box_dims_a = tuple(float(x) for x in atoms.cell.lengths())
        except Exception as exc:  # pragma: no cover -- defensive
            s.add("Box dim resolution", f"{exc}", "WARN",
                  "Falling back to (47.88, 70.01, 73.93)")
            box_dims_a = (47.88, 70.01, 73.93)

    bx, by, bz = box_dims_a
    s.add("Box dims used (X,Y,Z)", f"{bx:.2f} × {by:.2f} × {bz:.2f} Å", "INFO")

    # Native reciprocal pixel = 1/L (per axis) -> mrad via lambda.
    pix_mrad_x = (1.0 / bx) * WAVELENGTH_A * 1000.0
    pix_mrad_y = (1.0 / by) * WAVELENGTH_A * 1000.0
    s.add("Native pixel mrad (X / Y)",
          f"{pix_mrad_x:.3f} / {pix_mrad_y:.3f}",
          "INFO",
          "abTEM detector pixel size before binning")

    # The CBED array is asymmetric: BOX_X ≈ 48 Å gives coarser native pixels
    # (pix_mrad_x > pix_mrad_y). K must be chosen so the coarser (X) axis
    # reaches N_target pixels; the finer (Y) axis will be larger and py4DSTEM
    # resamples to square during preprocess. We report disk/DF px on the coarser
    # axis (worst case). K is derived as n_native_small // N_target where
    # n_native_small = min(n_native_x, n_native_y).
    pix_mrad_coarse = float(max(pix_mrad_x, pix_mrad_y))   # X-axis (BOX_X small)
    pix_mrad_fine = float(min(pix_mrad_x, pix_mrad_y))     # Y-axis (BOX_Y large)

    rows: list[dict] = []
    selected = None
    for theta_max in (150.0, 200.0, 250.0):
        n_native_coarse = int(np.ceil(2.0 * theta_max / pix_mrad_coarse))
        n_native_fine = int(np.ceil(2.0 * theta_max / pix_mrad_fine))
        for n_target in (48, 64, 96, 128):
            k = max(1, n_native_coarse // n_target)
            n_actual = n_native_coarse // k  # trim_excess: coarser axis
            n_actual_fine = n_native_fine // k     # finer axis is larger
            pix_mrad_eff = pix_mrad_coarse * k     # worst-case (coarser axis)
            bf_disk_px = int(round(2.0 * CONVERGENCE_MRAD / pix_mrad_eff))
            df_radius_px = int(round((theta_max - CONVERGENCE_MRAD) / pix_mrad_eff))
            # Storage: n_actual × n_actual_fine (asymmetric) before py4DSTEM crop.
            kb_per_pos = (n_actual * n_actual_fine) * 4 / 1024.0
            gb_full = (n_actual * n_actual_fine) * 4 * 16 * 16 * 9 / 1e9
            # R1: BF disk pixel diameter >= 28. ePIE-class updates need ~12 px
            # radius; 28 gives a bit of margin.
            # R2: dark-field radius >= 12. Multislice depth-sectioning needs
            # adequate sampling of the DF region; 12 px is a sane minimum.
            r1 = bf_disk_px >= 28
            r2 = df_radius_px >= 12
            r3 = gb_full <= 1.0
            # R4: VRAM working set heuristic for cropped-FFT backprop.
            r4_bytes = (n_actual ** 2) * 64 * 8 * 50  # batch 64 × 8 B × 50 slices
            r4 = r4_bytes <= 2e9
            passes = r1 and r2 and r3 and r4
            rows.append({
                "theta_max": theta_max,
                "n_target": n_target,
                "K": k,
                "n_actual": n_actual,
                "pix_mrad": pix_mrad_eff,
                "bf_disk_px": bf_disk_px,
                "df_radius_px": df_radius_px,
                "kb_per_pos": kb_per_pos,
                "gb_full": gb_full,
                "r1": r1, "r2": r2, "r3": r3, "r4": r4,
                "passes": passes,
            })
            if passes and (selected is None or n_actual < selected["n_actual"]):
                selected = rows[-1]

    body = ["", "Sweep over detector outer angle and target detector size:", ""]
    body.append("Note: CBED is asymmetric (X coarser). N_act = coarse-axis pixels; fine axis is larger.")
    body.append("py4DSTEM resamples to square during preprocess; disk/DF quoted on worst (coarse) axis.")
    body.append("")
    body.append("| θ_max | N_tgt | K | N_coarse | mrad/px | BF_disk_px | DF_r_px | KB/pos | GB(9 tiles) | R1 | R2 | R3 | R4 | pick |")
    body.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        is_pick = (selected is not None and r is selected)
        body.append(
            f"| {r['theta_max']:.0f} | {r['n_target']} | {r['K']} | {r['n_actual']} | "
            f"{r['pix_mrad']:.2f} | {r['bf_disk_px']} | {r['df_radius_px']} | "
            f"{r['kb_per_pos']:.1f} | {r['gb_full']:.3f} | "
            f"{'✓' if r['r1'] else '✗'} | {'✓' if r['r2'] else '✗'} | "
            f"{'✓' if r['r3'] else '✗'} | {'✓' if r['r4'] else '✗'} | "
            f"{'**←**' if is_pick else ''} |"
        )

    body += [
        "",
        "Selection rules in priority order:",
        "  R1: BF disk diameter ≥ 28 px (ePIE updates need ~12 px radius + margin)",
        "  R2: dark-field radius ≥ 12 px (axial sectioning needs DF info)",
        "  R3: total stored 4D-STEM ≤ 1 GB (compressed will be ~5–10× smaller)",
        "  R4: cropped-FFT working set (≈ N²·batch·8 B·num_slices) ≤ 2 GB on 8 GB GPU",
        "",
    ]
    if selected:
        body.append(
            f"**Recommended detector**: θ_max = {selected['theta_max']:.0f} mrad, "
            f"N = {selected['n_actual']}, bin K = {selected['K']}, "
            f"pix_mrad = {selected['pix_mrad']:.2f}, "
            f"BF disk = {selected['bf_disk_px']} px diameter, "
            f"DF radius = {selected['df_radius_px']} px, "
            f"~{selected['gb_full']*1000:.1f} MB across all 9 tiles."
        )
        s.add("Recommended config",
              f"θ_max={selected['theta_max']:.0f} mrad, N={selected['n_actual']}, K={selected['K']}",
              "PASS")
    else:
        body.append("**No config satisfies R1–R4.** Relax R3 (storage) "
                    "or accept smaller DF margin.")
        s.add("Recommended config", "none", "FAIL")

    s.body = "\n".join(body)


# ---------------------------------------------------------------------------
# 0f -- Storage & wall-clock projection
# ---------------------------------------------------------------------------
def check_0f_budget() -> None:
    s = section("0f — Storage & wall-clock projection")

    # Existing run baseline: ~8–10 h / tile, no phonons, slice 0.31 Å.
    base_lo, base_hi = 8.0, 10.0
    old_slice = 0.31

    body = ["", "Per-tile wall-clock projection (relative to existing 8–10 h baseline):", ""]
    body.append("| n_configs | sim slice | scale | per-tile h | 9-tile h | verdict |")
    body.append("|---|---|---|---|---|---|")
    pick = None
    options = [
        (4, 1.0, "4 phonons, 1 Å slice"),
        (4, 1.5, "4 phonons, 1.5 Å slice"),
        (8, 1.0, "8 phonons, 1 Å slice"),
        (8, 1.5, "8 phonons, 1.5 Å slice"),
        (1, 1.0, "no phonons, 1 Å slice (sanity baseline)"),
    ]
    for n_cfg, slice_a, label in options:
        scale = (n_cfg / 1.0) * (old_slice / slice_a)
        per_lo, per_hi = base_lo * scale, base_hi * scale
        nine_lo, nine_hi = 9 * per_lo, 9 * per_hi
        verdict = "PASS" if per_hi <= 24 else ("WARN" if per_hi <= 48 else "FAIL")
        body.append(f"| {n_cfg} | {slice_a:.1f} Å | ×{scale:.2f} | "
                    f"{per_lo:.1f}–{per_hi:.1f} | {nine_lo:.0f}–{nine_hi:.0f} | {verdict} |")
        if pick is None and verdict == "PASS":
            pick = (n_cfg, slice_a, label, per_lo, per_hi)

    body.append("")
    if pick is not None:
        n_cfg, slice_a, label, per_lo, per_hi = pick
        body.append(f"**Recommended: {label}** "
                    f"(per-tile {per_lo:.1f}–{per_hi:.1f} h, "
                    f"9-tile {9*per_lo:.0f}–{9*per_hi:.0f} h)")
        s.add("Recommended sim config",
              label,
              "PASS",
              f"per-tile {per_lo:.1f}–{per_hi:.1f} h")
    else:
        body.append("**No config fits within 24 h/tile.** Reduce n_configs further "
                    "or relax slice thickness to 2 Å.")
        s.add("Recommended sim config", "none under 24 h/tile", "FAIL")

    body.append("")
    body.append("Toy (1 tile, 4×4 scan positions, n_configs=2, slice=2 Å) target: ≤ 1 h")
    s.add("Toy budget", "≤ 1 h target", "INFO",
          "1 tile × 4×4 scan × n_configs=2 × slice=2.0 Å for the gating end-to-end run")
    s.body = "\n".join(body)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
ALL_CHECKS = {
    "0a": check_0a_geometry,
    "0b": check_0b_params,
    "0c": check_0c_phonons,
    "0d": check_0d_existing,
    "0e": check_0e_detector,
    "0f": check_0f_budget,
}


def main(argv: Iterable[str] | None = None) -> int:
    # Windows console defaults to cp1252; coerce to utf-8 so unicode in the
    # PASS/FAIL table prints cleanly.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Phase 0 diagnostics for the PTO/STO ptycho pipeline")
    parser.add_argument("--only", nargs="+", choices=list(ALL_CHECKS), default=None)
    args = parser.parse_args(argv)

    SECTIONS.clear()
    PLOTS_DIR.mkdir(exist_ok=True)

    keys = args.only or list(ALL_CHECKS)
    for k in keys:
        try:
            ALL_CHECKS[k]()
        except Exception as exc:  # pragma: no cover -- defensive
            s = section(f"{k} — UNCAUGHT ERROR")
            s.add("error", repr(exc), "FAIL")

    # Top-line verdict.
    fail_count = sum(1 for s in SECTIONS for c in s.checks if c.verdict == "FAIL")
    warn_count = sum(1 for s in SECTIONS for c in s.checks if c.verdict == "WARN")
    if fail_count == 0 and warn_count == 0:
        verdict = "READY-TO-SIM"
    elif fail_count == 0:
        verdict = "READY-WITH-WARNINGS"
    else:
        verdict = "BLOCKED"

    parts = [
        "# Phase 0 diagnostics report",
        "",
        f"**Verdict: {verdict}** ({fail_count} FAIL, {warn_count} WARN)",
        "",
        "Generated by `diagnostics.py`. Plots in `diagnostics_plots/`.",
        "",
    ]
    parts += [s.render() for s in SECTIONS]
    REPORT_PATH.write_text("\n".join(parts), encoding="utf-8")

    # Print compact stdout summary.
    for s in SECTIONS:
        print(f"\n=== {s.name} ===")
        for c in s.checks:
            print(f"  [{c.verdict:4s}] {c.label}: {c.value}" + (f"  -- {c.note}" if c.note else ""))
    print(f"\nVerdict: {verdict}")
    print(f"Report:  {REPORT_PATH}")
    print(f"Plots:   {PLOTS_DIR}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
