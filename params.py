"""Single source of truth for the PTO/STO 4D-STEM + ptychography pipeline.

Encodes the experimental constants and derives every other parameter (probe,
scan, detector, slicing) from them. Importable from the simulation, recon,
and validation scripts so they cannot drift.

Beam-axis convention: abTEM propagates along **+Z** after `orthogonalize_cell`
+ `center(axis=2)`. ``REAL_SPACE_THICKNESS`` is therefore `cell.lengths()[2]`.
The notebook's old `BOX_Y` choice was wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Fundamental experimental constants (Cell 0 of the original notebook).
# ---------------------------------------------------------------------------
ENERGY_EV: float = 300e3
CONVERGENCE_MRAD: float = 100.0
OVERFOCUS_NM: float = 1.0  # positive = probe focused before sample (overfocus)

HAADF_INNER_MRAD: float = 110.0
HAADF_OUTER_MRAD: float = 250.0

# Detector: outer angle and target detector array size are derived in
# diagnostics.py (Phase 0e). Defaults below are the picked config for 8 GB VRAM.
DETECTOR_MAX_ANGLE_MRAD: float = 200.0
DETECTOR_TARGET_PIXELS: int = 64

# Scan: the existing notebook nominally declares 40 × 40 Å but tiles only 12 ×
# 12 Å. We keep the centre and the 9-tile grid so existing zarr files remain
# usable; the production run can be expanded later if compute budget allows.
CENTER_X_A: float = 23.0
CENTER_Y_A: float = 35.0
SCAN_WIDTH_A: float = 40.0   # nominal scan window (only central 12 Å tiled)
TILE_SIZE_A: float = 4.0     # one tile is TILE_SIZE_A × TILE_SIZE_A
TARGET_OVERLAP: float = 0.75  # fraction of probe FWHM to overlap

# Multislice simulation slice thickness. Existing 0.31 Å is over-fine; 1.0 Å
# is sufficient at 100 mrad and gives a ~3× speedup.
SIM_SLICE_THICKNESS_A: float = 1.0

# Frozen phonons. Sigmas are starting values from perovskite literature;
# verify against DWFs before production. n_configs=4 keeps total per-tile
# wall-clock under ~13 h on the user's hardware.
PHONON_NUM_CONFIGS: int = 4
PHONON_SIGMAS_A: dict[str, float] = {
    "Pb": 0.092, "Sr": 0.085, "Ti": 0.072, "O": 0.085,
}

# File system.
PROJECT_ROOT: Path = Path(__file__).resolve().parent
PARENT_ROOT: Path = PROJECT_ROOT.parent
STRUCTURE_FILE: Path = PROJECT_ROOT / "PTO6_STO6_18_18_labyrinthPoscar.vasp"


# ---------------------------------------------------------------------------
# Atom & cell geometry helper.
# ---------------------------------------------------------------------------
def load_atoms():
    """Read POSCAR, rotate so the original Z is the long in-plane axis,
    orthogonalize, centre with vacuum on the beam axis. Returns an ASE Atoms.
    """
    import ase.io
    import abtem
    atoms = ase.io.read(str(STRUCTURE_FILE))
    atoms.rotate(-90, "y", rotate_cell=True)
    atoms = abtem.orthogonalize_cell(atoms)
    atoms.center(axis=2, vacuum=2.0)
    return atoms


# ---------------------------------------------------------------------------
# Derived parameters (everything you need for sim + recon).
# ---------------------------------------------------------------------------
def relativistic_wavelength_a(energy_ev: float = ENERGY_EV) -> float:
    """Relativistic electron wavelength in angstroms."""
    h = 6.62607015e-34
    m = 9.1093837015e-31
    e = 1.602176634e-19
    c = 299_792_458.0
    p = np.sqrt(2.0 * m * energy_ev * e * (1.0 + energy_ev * e / (2.0 * m * c * c)))
    return float(h / p * 1e10)


@dataclass(frozen=True)
class Params:
    """Frozen, fully-derived parameter bundle.

    Build with :func:`derive`. All units are angstroms / mrad / eV unless the
    field name says otherwise.
    """
    # Direct constants (echoed for completeness).
    energy_ev: float
    convergence_mrad: float
    overfocus_a: float
    haadf_inner_mrad: float
    haadf_outer_mrad: float
    detector_max_angle_mrad: float
    detector_target_pixels: int
    target_overlap: float
    sim_slice_thickness_a: float
    phonon_num_configs: int
    phonon_sigmas_a: dict

    # Geometry (after rotate/orthogonalize/center).
    box_x_a: float
    box_y_a: float
    box_z_a: float
    real_space_thickness_a: float  # along beam = box_z_a

    # Optics.
    wavelength_a: float
    probe_fwhm_focused_a: float
    probe_fwhm_effective_a: float

    # Sampling / scan.
    scan_step_a: float
    scan_window_a: float
    tile_size_a: float
    center_x_a: float
    center_y_a: float
    num_tiles_per_side: int

    # Multislice.
    num_slices_sim: int  # ceil(thickness / slice)

    # Detector binning chosen so the saved CBED has ~detector_target_pixels per
    # axis after coarsening. Computed from the long-axis native pixel mrad.
    bf_bin: int
    bf_native_pixel_mrad: float
    bf_eff_pixel_mrad: float
    bf_disk_diameter_px: int
    bf_df_radius_px: int

    # Reconstruction defaults.
    recon_num_slices: int
    recon_slice_thickness_a: float
    recon_max_batch_size: int
    recon_diff_intensities_shape: Tuple[int, int]


def _native_pixel_mrad(box_a: float, wavelength_a: float) -> float:
    """abTEM reciprocal-pixel size in mrad for a real-space box of length box_a."""
    return (1.0 / box_a) * wavelength_a * 1000.0


def derive(
    *,
    energy_ev: float = ENERGY_EV,
    convergence_mrad: float = CONVERGENCE_MRAD,
    overfocus_nm: float = OVERFOCUS_NM,
    detector_max_angle_mrad: float = DETECTOR_MAX_ANGLE_MRAD,
    detector_target_pixels: int = DETECTOR_TARGET_PIXELS,
    sim_slice_thickness_a: float = SIM_SLICE_THICKNESS_A,
    phonon_num_configs: int = PHONON_NUM_CONFIGS,
    target_overlap: float = TARGET_OVERLAP,
    box_dims_a: Tuple[float, float, float] | None = None,
    recon_num_slices: int | None = None,
    recon_max_batch_size: int = 64,
    recon_diff_intensities_shape: Tuple[int, int] = (64, 64),
) -> Params:
    """Compute the derived Params bundle.

    If ``box_dims_a`` is None, the POSCAR is read and processed; pass an
    explicit triple to avoid the disk read (used by tests and diagnostics).
    """
    if box_dims_a is None:
        atoms = load_atoms()
        box_dims_a = tuple(float(x) for x in atoms.cell.lengths())
    box_x_a, box_y_a, box_z_a = box_dims_a
    real_space_thickness_a = box_z_a  # beam = +Z in abTEM

    overfocus_a = overfocus_nm * 10.0

    wavelength_a = relativistic_wavelength_a(energy_ev)
    alpha_rad = convergence_mrad * 1e-3
    probe_fwhm_focused_a = 0.61 * wavelength_a / alpha_rad
    geom_spread_a = alpha_rad * overfocus_a
    probe_fwhm_effective_a = float(np.sqrt(probe_fwhm_focused_a ** 2 + geom_spread_a ** 2))

    scan_step_a = (1.0 - target_overlap) * probe_fwhm_effective_a
    num_tiles_per_side = int(np.ceil(SCAN_WIDTH_A / TILE_SIZE_A))

    num_slices_sim = int(np.ceil(real_space_thickness_a / sim_slice_thickness_a))

    # Detector design.
    # The cell is asymmetric (BOX_X ≈ 48 Å, BOX_Y ≈ 70 Å) so the CBED native
    # pixel size differs between axes: pix_mrad_x > pix_mrad_y.  A single K
    # applied to both axes via da.coarsen({2:K,3:K}) gives different output
    # sizes: n_y = ceil(2*theta/pix_y)//K, n_x = ceil(2*theta/pix_x)//K.
    # We must choose K from the SMALLER native dimension (X-axis) so that the
    # coarser axis has at least N_target pixels; the finer axis (Y) ends up
    # larger and py4DSTEM crops it to square during preprocess resampling.
    pix_mrad_x = _native_pixel_mrad(box_x_a, wavelength_a)
    pix_mrad_y = _native_pixel_mrad(box_y_a, wavelength_a)
    n_native_y = int(np.ceil(2.0 * detector_max_angle_mrad / pix_mrad_y))
    n_native_x = int(np.ceil(2.0 * detector_max_angle_mrad / pix_mrad_x))
    n_native_small = min(n_native_x, n_native_y)  # drives K to not under-sample X
    bf_bin = max(1, n_native_small // detector_target_pixels)
    # Effective mrad/px and disk sizes quoted on the COARSER axis (X here).
    bf_native_pixel_mrad = float(max(pix_mrad_x, pix_mrad_y))  # coarser = X
    bf_eff_pixel_mrad = bf_native_pixel_mrad * bf_bin
    bf_disk_diameter_px = int(round(2.0 * convergence_mrad / bf_eff_pixel_mrad))
    bf_df_radius_px = int(round((detector_max_angle_mrad - convergence_mrad)
                                / bf_eff_pixel_mrad))

    # Recon slice count: target ~1 Å per slice, but ensure dz_optical = λ/α²
    # is at least 2× over-sampled.
    if recon_num_slices is None:
        recon_num_slices = int(round(real_space_thickness_a / 1.0))
    recon_slice_thickness_a = float(real_space_thickness_a / recon_num_slices)

    return Params(
        energy_ev=energy_ev,
        convergence_mrad=convergence_mrad,
        overfocus_a=overfocus_a,
        haadf_inner_mrad=HAADF_INNER_MRAD,
        haadf_outer_mrad=HAADF_OUTER_MRAD,
        detector_max_angle_mrad=detector_max_angle_mrad,
        detector_target_pixels=detector_target_pixels,
        target_overlap=target_overlap,
        sim_slice_thickness_a=sim_slice_thickness_a,
        phonon_num_configs=phonon_num_configs,
        phonon_sigmas_a={"Pb": 0.092, "Sr": 0.085, "Ti": 0.072, "O": 0.085},
        box_x_a=box_x_a,
        box_y_a=box_y_a,
        box_z_a=box_z_a,
        real_space_thickness_a=real_space_thickness_a,
        wavelength_a=wavelength_a,
        probe_fwhm_focused_a=probe_fwhm_focused_a,
        probe_fwhm_effective_a=probe_fwhm_effective_a,
        scan_step_a=scan_step_a,
        scan_window_a=SCAN_WIDTH_A,
        tile_size_a=TILE_SIZE_A,
        center_x_a=CENTER_X_A,
        center_y_a=CENTER_Y_A,
        num_tiles_per_side=num_tiles_per_side,
        num_slices_sim=num_slices_sim,
        bf_bin=bf_bin,
        bf_native_pixel_mrad=bf_native_pixel_mrad,
        bf_eff_pixel_mrad=bf_eff_pixel_mrad,
        bf_disk_diameter_px=bf_disk_diameter_px,
        bf_df_radius_px=bf_df_radius_px,
        recon_num_slices=recon_num_slices,
        recon_slice_thickness_a=recon_slice_thickness_a,
        recon_max_batch_size=recon_max_batch_size,
        recon_diff_intensities_shape=recon_diff_intensities_shape,
    )


def toy_params() -> Params:
    """Tiny config for the < 1 h gating test."""
    return derive(
        sim_slice_thickness_a=2.0,
        phonon_num_configs=2,
        recon_num_slices=20,
        recon_max_batch_size=16,
        recon_diff_intensities_shape=(48, 48),
    )


def summary(p: Params) -> str:
    return (
        f"Params summary\n"
        f"  Beam:        {p.energy_ev/1e3:.0f} keV  λ={p.wavelength_a:.4f} Å\n"
        f"  Convergence: {p.convergence_mrad:.0f} mrad  overfocus={p.overfocus_a:.1f} Å\n"
        f"  Cell:        {p.box_x_a:.2f} × {p.box_y_a:.2f} × {p.box_z_a:.2f} Å"
        f"  (beam=Z, thickness={p.real_space_thickness_a:.2f} Å)\n"
        f"  Probe:       FWHM_focused={p.probe_fwhm_focused_a:.3f} Å, "
        f"effective={p.probe_fwhm_effective_a:.3f} Å\n"
        f"  Scan:        step={p.scan_step_a:.3f} Å  "
        f"({p.target_overlap*100:.0f}% overlap)\n"
        f"  Sim slices:  {p.num_slices_sim} × {p.sim_slice_thickness_a:.2f} Å\n"
        f"  Phonons:     n_configs={p.phonon_num_configs}, "
        f"σ={p.phonon_sigmas_a}\n"
        f"  Detector:    θ_max={p.detector_max_angle_mrad:.0f} mrad, "
        f"N≈{p.detector_target_pixels}, K={p.bf_bin}, "
        f"eff_pix={p.bf_eff_pixel_mrad:.2f} mrad/px\n"
        f"               BF disk = {p.bf_disk_diameter_px} px diameter, "
        f"DF radius = {p.bf_df_radius_px} px\n"
        f"  Recon:       {p.recon_num_slices} slices × "
        f"{p.recon_slice_thickness_a:.2f} Å, "
        f"batch={p.recon_max_batch_size}, "
        f"crop={p.recon_diff_intensities_shape}\n"
    )


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        p = derive()
    except Exception as exc:
        print(f"Falling back to placeholder box dims (no abTEM available): {exc}")
        p = derive(box_dims_a=(47.88, 70.01, 73.93))
    print(summary(p))
