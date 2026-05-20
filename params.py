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
# Positive = probe focused before sample (overfocus). W1 Phase C: 5.0 -> 20.0 nm.
# The W1 fold_slice port at 5 nm overfocus hit kz-FRC 0.087 — ~13000x better than
# py4DSTEM on the same data but still under-determined (LABBOOK § 12). 20 nm
# overfocus matches the Lei & Wang paper recipe: geometric spread α·Δf = 0.1·200
# Å = 20 Å effective probe footprint (vs ~5 Å before), so each object pixel sees
# ~16× more scan positions => much better conditioning. Requires TILE_X=TILE_Y=2
# below — at α=100 mrad the probe reach α·(Δf+t) ≈ 27 Å plus half_scan would wrap
# around the original 48×70 Å cell. Still POSITIVE (focused above the sample);
# focusing inside the sample is a hard user constraint.
OVERFOCUS_NM: float = 20.0

HAADF_INNER_MRAD: float = 110.0
# 185 mrad is safely within the ~197 mrad simulated angular range (grid-limited
# by the 0.05 Å sampling floor on the 47.88 Å cell). 250 mrad would exceed it.
HAADF_OUTER_MRAD: float = 185.0

# Detector: outer angle and target detector array size are derived in
# diagnostics.py (Phase 0e). Defaults below are the picked config for 8 GB VRAM.
DETECTOR_MAX_ANGLE_MRAD: float = 200.0
DETECTOR_TARGET_PIXELS: int = 64

# Scan: the simulated/stitched region is an N_TILES_TILED × N_TILES_TILED grid
# of TILE_SIZE_A tiles, centred on (CENTER_X_A, CENTER_Y_A). 5×5 (20×20 Å) since
# W2 (~26×26 Å is what the reference paper used). W1 Phase C: scan placed on
# the LOWER of the two polarisation vortex cores in the POSCAR (the upper one
# at ~28,50 and the lower at ~23,22 per GroundTruth.ipynb streamlines). After
# STO_X_PAD_UC=5 perovskite UCs of pure STO are added on each X-edge, the
# lower vortex sits at x = 5·3.99 + 23 = 42.95 Å in the padded cell
# (87.78 × 70.01 Å untiled). TILE_Y=2 → 87.78 × 140.02 Å; the scan goes on the
# upper Y-replica at (42.95, 22+70.01 = 92.01) because the y=22 copy hard
# against the y=0 edge would alias. Clearance at 20 nm overfocus + 5×5 × 4 Å
# scan: probe reach 37.4 Å; X-margins (42.95, 44.83), Y-margins (92.01,
# 48.01) — all positive, tightest = +5.55 Å on the left X-edge. Slightly
# better headroom than scanning the upper vortex (+2.4 Å) because x=23 is
# more centred in the padded cell than x=28.
CENTER_X_A: float = 42.95
CENTER_Y_A: float = 92.01
N_TILES_TILED: int = 5        # tiles per side of the simulated grid (was 3)
TILE_SIZE_A: float = 4.0      # one tile is TILE_SIZE_A × TILE_SIZE_A

# Cell replication (W1 Phase C). The labyrinth is periodic in X (PTO/STO
# sandwich, 12 perovskite UCs; X-edges are pure STO) and Y (in-plane
# "footlong", 18 perovskite UCs). Z is the beam axis and looks down a
# polarisation vortex core — DO NOT tile Z.
# Phase C strategy: X is widened by asymmetric padding with pure STO on each
# edge (see STO_X_PAD_UC below and load_atoms), NOT by full-cell tiling. The
# X-edges of the natural cell are already STO (Sr-only at x=0..15 and x=37..48
# per Phase 0 load_atoms verification), so adding more STO on each side
# preserves the periodic wrap (STO→STO, polarisation = 0 on both sides).
# This is ~35% cheaper FFT than a full TILE_X=3 because the pad atoms are
# pure perovskite (no labyrinth modulation to replicate). Y is straight-tiled
# (TILE_Y=2) because the y≈50 vortex is too close to the y=70.01 top edge of
# the untiled cell to give the probe its 37.4 Å clearance.
TILE_X: int = 1   # X widened via STO padding (see STO_X_PAD_UC); no cell tiling
TILE_Y: int = 2   # footlong axis straight tiling

# Perovskite-UC pad of pure STO added on each X-edge of the POSCAR before any
# tiling. 0 = disabled (use plain TILE_X-only tiling instead). 5 UCs each side
# → 87.78 Å total X (vs 95.77 Å of TILE_X=2 or 143.64 Å of TILE_X=3) with
# clearance margin +2.4 Å on the right X-edge at 20 nm overfocus + 5×5 × 4 Å
# scan. The pad UC is extracted from the leftmost UC of the POSCAR (which
# sits in the natural STO_3 flank — see load_atoms below).
STO_X_PAD_UC: int = 5
SCAN_WIDTH_A: float = 40.0    # legacy nominal window; derive() now uses
                              # N_TILES_TILED·TILE_SIZE_A as the real window
# Fraction of (effective) probe FWHM that successive scan positions overlap.
# W1 Phase C: 0.95 -> 0.99 to drive scan density up alongside the wider probe.
# At 20 nm overfocus FWHM_eff ≈ 20 Å, so step = (1-0.99)·20 = 0.20 Å => 400
# pos/tile × 25 tiles = 10000 positions, realized linear overlap ≈ 99.5 %.
# Each object pixel is constrained by ~16× more positions than the W1 6400-pos
# / 5 nm baseline. Sim ~15-18 h, recon ~6-8 h.
TARGET_OVERLAP: float = 0.99

# Multislice simulation slice thickness. Existing 0.31 Å is over-fine; 1.0 Å
# is sufficient at 100 mrad and gives a ~3× speedup.
SIM_SLICE_THICKNESS_A: float = 1.0

# Frozen phonons. Sigmas are starting values from perovskite literature.
# 8 configs: the GPU sim runs ~10-20 min/tile (the handover's "8-13 h/tile"
# was a wild over-estimate), and dark-field signal quality (suspected
# depth-sectioning bottleneck) scales with TDS sampling, so 8 > 4 is worth it.
PHONON_NUM_CONFIGS: int = 8
PHONON_SIGMAS_A: dict[str, float] = {
    "Pb": 0.092, "Sr": 0.085, "Ti": 0.072, "O": 0.085,
}
# Multiplies every phonon σ. 1.0 = room temperature. ~0.65 ≈ LN2 / cryo (W3):
# the paper notes depth resolution is "sensitive to lattice vibrations" and
# cryo helps (handover §5, §6.4). **Cryo is now the default** (we're committed
# to a cryo experiment) — override per run with simulate_4dstem.py --room for
# the room-temperature ablation. --cryo is still accepted (it's the default).
PHONON_DWF_SCALE_DEFAULT: float = 0.65

# File system.
PROJECT_ROOT: Path = Path(__file__).resolve().parent
PARENT_ROOT: Path = PROJECT_ROOT.parent
STRUCTURE_FILE: Path = PROJECT_ROOT / "PTO6_STO6_18_18_labyrinthPoscar.vasp"


# ---------------------------------------------------------------------------
# Atom & cell geometry helper.
# ---------------------------------------------------------------------------
def load_atoms():
    """Read POSCAR, rotate so the original Z is the long in-plane axis,
    orthogonalize, apply STO X-pad + TILE_X/TILE_Y replication, then centre
    with vacuum on the beam axis. Returns an ASE Atoms.

    When STO_X_PAD_UC > 0 the cell is widened in X by adding `STO_X_PAD_UC`
    perovskite unit cells of pure STO on each X-edge. The pad UC is extracted
    from the leftmost UC of the POSCAR (which sits in the natural STO_3 flank,
    verified via Phase 0 load_atoms output: Pb concentrated at x=15..37, Sr
    only at x<15 and x>37). The result is [STO pad | POSCAR | STO pad] in X
    with continuous lattice (3.99 Å perovskite period) and STO→STO periodic
    wrap on the X-axis. TILE_X tiling is then applied on top of the padded
    cell (typically TILE_X=1 when padding handles the X-widening).
    """
    import ase.io
    import abtem
    atoms = ase.io.read(str(STRUCTURE_FILE))
    atoms.rotate(-90, "y", rotate_cell=True)
    atoms = abtem.orthogonalize_cell(atoms)

    # Asymmetric STO padding in X (W1 Phase C).
    if STO_X_PAD_UC > 0:
        bx, by, bz = atoms.cell.lengths()
        a_pto = bx / 12.0   # POSCAR is PTO6/STO6 = 12 perovskite UCs along X
        # Extract one UC of pure STO from the left X-edge of the POSCAR.
        sto_uc_mask = atoms.positions[:, 0] < a_pto
        sto_uc = atoms[sto_uc_mask].copy()
        sto_uc.set_cell([a_pto, by, bz])
        sto_uc.pbc = [True, True, True]
        # Build the left pad by replicating the UC; right pad is a copy.
        sto_pad_left = sto_uc * (STO_X_PAD_UC, 1, 1)
        sto_pad_right = sto_pad_left.copy()
        # Shift the POSCAR right past the left pad.
        atoms_shifted = atoms.copy()
        atoms_shifted.translate([STO_X_PAD_UC * a_pto, 0.0, 0.0])
        # Shift the right pad past the POSCAR.
        sto_pad_right.translate([STO_X_PAD_UC * a_pto + bx, 0.0, 0.0])
        # Combine: [left_pad] + [original] + [right_pad].
        atoms = sto_pad_left + atoms_shifted + sto_pad_right
        new_bx = 2.0 * STO_X_PAD_UC * a_pto + bx
        # set_cell with scale_atoms=False (default) preserves atom positions
        # — we only resize the cell vectors.
        atoms.set_cell([new_bx, by, bz])
        atoms.pbc = [True, True, True]

    # Tile BEFORE center(axis=2) so the Z-vacuum padding survives the tile.
    if TILE_X != 1 or TILE_Y != 1:
        atoms = atoms * (TILE_X, TILE_Y, 1)
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
    phonon_sigmas_a: dict          # already scaled by phonon_dwf_scale
    phonon_dwf_scale: float

    # Geometry (after rotate/orthogonalize/center).
    box_x_a: float
    box_y_a: float
    box_z_a: float
    real_space_thickness_a: float  # along beam = box_z_a

    # Optics.
    wavelength_a: float
    probe_fwhm_focused_a: float
    probe_fwhm_effective_a: float
    probe_disk_diameter_a: float   # geometric shadow disk of the defocused probe = 2·α·Δf

    # Sampling / scan.
    scan_step_a: float
    scan_window_a: float
    overlap_linear: float          # realized linear probe overlap = 1 − scan_step / probe footprint Ø
    tile_size_a: float
    center_x_a: float
    center_y_a: float
    n_tiles_tiled: int   # simulated/stitched grid is n_tiles_tiled × n_tiles_tiled

    # Multislice.
    num_slices_sim: int  # ceil(thickness / slice)

    # Detector binning chosen so the saved CBED has ~detector_target_pixels per
    # axis after coarsening. Computed from the long-axis native pixel mrad.
    bf_bin: int
    bf_native_pixel_mrad: float
    bf_eff_pixel_mrad: float        # coarser axis (X-direction CBED ≈ 6.17 mrad/px)
    bf_eff_pixel_mrad_y: float      # finer axis (Y-direction CBED ≈ 4.22 mrad/px)
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
    n_tiles_tiled: int = N_TILES_TILED,
    dwf_scale: float = PHONON_DWF_SCALE_DEFAULT,
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
    # geom_spread = α·Δf is the geometric shadow-disk *radius* of the
    # defocused probe; the derived scan step uses it (via probe_fwhm_effective)
    # as a conservative probe-size proxy. The actual probe *footprint diameter*
    # is ≈ 2·α·Δf, so the realized overlap (below) comes out a bit *higher*
    # than `target_overlap` — fine, more overlap diversity is the goal (W2).
    geom_spread_a = alpha_rad * overfocus_a
    probe_fwhm_effective_a = float(np.sqrt(probe_fwhm_focused_a ** 2 + geom_spread_a ** 2))
    probe_disk_diameter_a = 2.0 * alpha_rad * overfocus_a

    scan_step_a = (1.0 - target_overlap) * probe_fwhm_effective_a
    # The real scan window is the tiled grid, centred on (CENTER_X_A, CENTER_Y_A).
    scan_window_a = float(n_tiles_tiled * TILE_SIZE_A)
    # Realized linear overlap of adjacent probe footprints. Use the larger of
    # the focused FWHM and the geometric disk diameter as the footprint Ø.
    probe_footprint_a = max(probe_fwhm_focused_a, probe_disk_diameter_a)
    overlap_linear = float(max(0.0, 1.0 - scan_step_a / probe_footprint_a))

    phonon_sigmas_scaled = {k: v * dwf_scale for k, v in PHONON_SIGMAS_A.items()}

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
    # Per-axis effective mrad/px. fold_slice consumes both (no resample);
    # py4DSTEM consumes only the coarser value after the square-resample.
    bf_eff_pixel_mrad_y = float(min(pix_mrad_x, pix_mrad_y)) * bf_bin   # finer = Y
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
        phonon_sigmas_a=phonon_sigmas_scaled,
        phonon_dwf_scale=dwf_scale,
        box_x_a=box_x_a,
        box_y_a=box_y_a,
        box_z_a=box_z_a,
        real_space_thickness_a=real_space_thickness_a,
        wavelength_a=wavelength_a,
        probe_fwhm_focused_a=probe_fwhm_focused_a,
        probe_fwhm_effective_a=probe_fwhm_effective_a,
        probe_disk_diameter_a=probe_disk_diameter_a,
        scan_step_a=scan_step_a,
        scan_window_a=scan_window_a,
        overlap_linear=overlap_linear,
        tile_size_a=TILE_SIZE_A,
        center_x_a=CENTER_X_A,
        center_y_a=CENTER_Y_A,
        n_tiles_tiled=n_tiles_tiled,
        num_slices_sim=num_slices_sim,
        bf_bin=bf_bin,
        bf_native_pixel_mrad=bf_native_pixel_mrad,
        bf_eff_pixel_mrad=bf_eff_pixel_mrad,
        bf_eff_pixel_mrad_y=bf_eff_pixel_mrad_y,
        bf_disk_diameter_px=bf_disk_diameter_px,
        bf_df_radius_px=bf_df_radius_px,
        recon_num_slices=recon_num_slices,
        recon_slice_thickness_a=recon_slice_thickness_a,
        recon_max_batch_size=recon_max_batch_size,
        recon_diff_intensities_shape=recon_diff_intensities_shape,
    )


def toy_params(*, dwf_scale: float = PHONON_DWF_SCALE_DEFAULT) -> Params:
    """Tiny config for the < 1 h gating test (and the ~few-h --mini run).

    n_tiles_tiled=2 ⇒ a 2×2 tile block; the toy/mini CLIs pick a 2×2 subset.
    """
    return derive(
        sim_slice_thickness_a=2.0,
        phonon_num_configs=2,
        n_tiles_tiled=2,
        dwf_scale=dwf_scale,
        recon_num_slices=20,
        recon_max_batch_size=16,
        recon_diff_intensities_shape=(48, 48),
    )


def _pos_per_tile_axis(p: Params) -> int:
    """Approx scan positions along one tile axis (abTEM GridScan rounds
    extent/sampling)."""
    return max(1, int(round(p.tile_size_a / p.scan_step_a)))


def summary(p: Params) -> str:
    return (
        f"Params summary\n"
        f"  Beam:        {p.energy_ev/1e3:.0f} keV  λ={p.wavelength_a:.4f} Å\n"
        f"  Convergence: {p.convergence_mrad:.0f} mrad  overfocus={p.overfocus_a:.1f} Å\n"
        f"  Cell:        {p.box_x_a:.2f} × {p.box_y_a:.2f} × {p.box_z_a:.2f} Å"
        f"  (beam=Z, thickness={p.real_space_thickness_a:.2f} Å)\n"
        f"  Probe:       FWHM_focused={p.probe_fwhm_focused_a:.3f} Å, "
        f"effective(FWHM)={p.probe_fwhm_effective_a:.3f} Å, "
        f"footprint Ø(=2·α·Δf)={p.probe_disk_diameter_a:.2f} Å\n"
        f"  Scan:        {p.n_tiles_tiled}×{p.n_tiles_tiled} tiles × "
        f"{p.tile_size_a:.1f} Å = {p.scan_window_a:.0f}×{p.scan_window_a:.0f} Å, "
        f"step={p.scan_step_a:.3f} Å\n"
        f"  Overlap:     {p.overlap_linear*100:.1f}% linear "
        f"(step {p.scan_step_a:.3f} Å / footprint {p.probe_disk_diameter_a:.2f} Å); "
        f"target was {p.target_overlap*100:.0f}% of effective FWHM\n"
        f"               ≈{_pos_per_tile_axis(p)**2} pos/tile ⇒ "
        f"≈{(_pos_per_tile_axis(p)*p.n_tiles_tiled)**2} positions total\n"
        f"  Sim slices:  {p.num_slices_sim} × {p.sim_slice_thickness_a:.2f} Å\n"
        f"  Phonons:     n_configs={p.phonon_num_configs}, "
        f"DWF×{p.phonon_dwf_scale:.2f}, σ={p.phonon_sigmas_a}\n"
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
