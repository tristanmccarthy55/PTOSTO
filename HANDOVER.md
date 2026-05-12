# PTO/STO 4D-STEM → Multislice Ptychography — Handover (2026-05-12)

Continuation handover for a new chat. Goal, status, what was tried, what the
data says, and ranked ways forward. Read this first, then the referenced files.

---

## 0. The goal

Simulate 4D-STEM of a PTO6/STO6 ferroelectric **vortex labyrinth** (POSCAR:
`PTO6_STO6_18_18_labyrinthPoscar.vasp`, ~19k atoms, ~74 Å along the beam),
reconstruct a **3D phase volume** with multislice ptychography, and resolve the
polarisation "tornado" — i.e. see the ferroelectric displacements *change with
depth* along the column. Target: sub-unit-cell axial resolution (~2 Å).

We are **improving on** `TileDataAnalysis_WORKS copy 2.ipynb`, whose 3D
reconstruction is axially blurred (atom columns smear through the whole
thickness — see `image.png` for its output: "3D Ptychography 20.5×15.2×70.0 Å³,
98 slices").

---

## 1. TL;DR status

**The pipeline now runs end-to-end, cleanly, with verified-correct parameters —
but it does NOT depth-section.** The production reconstruction
(`ptycho_recon.zarr`, `validation_plots/ptycho_recon_summary.png`) is a **2D
projected phase replicated across all 74 z-slices with a smooth envelope
weighting**. The depth-direction FRC is a single delta at kz=0 with nothing at
finite kz — textbook signature of a projection. Per-slice std "ratio 7.98"
*passes* the validator but is meaningless: it's the reconstruction-confidence
envelope (low at the surfaces, hump in the middle), not depth-resolved structure.
The XZ panel shows full-thickness vertical streaks — the *same failure mode* as
the old notebook, despite fixing every item in the original plan.

This is not a bug. It's the physics/engine: a single perpendicular 4D-STEM raster
of a 7 nm crystal, reconstructed with py4DSTEM's gradient-descent MSPT, does not
contain / cannot invert enough depth information. The reference paper achieves
atomic-layer 3D — but with **PtychoShelves (LSQML)**, ultrahigh dose, and
typically tilt-coupling for sub-nm depth.

---

## 2. What the pipeline is (current state of the `.py` files)

All in `C:\Users\Trist\HyperSpy-bundle\MY HYPERSPY CODE\PTOSTO\`. Python:
`C:\Users\Trist\HyperSpy-bundle\python.exe` (abtem 1.0.5, py4DSTEM 0.14.18,
zarr 2.18.4, CUDA 13.1, 8 GB VRAM, Win10).

- **`params.py`** — single source of truth. Derives probe/scan/detector/slicing
  from experimental constants. Beam axis = **+Z** (`real_space_thickness =
  cell.lengths()[2]` = 73.93 Å — the notebook's `BOX_Y` was wrong).
  Key constants: 300 keV, **CONVERGENCE 100 mrad**, **OVERFOCUS 1.0 nm (=10 Å)**,
  HAADF 110–185 mrad (185, not 250 — the abTEM grid only reaches ~197 mrad),
  detector max 200 mrad, target 64 px, **PHONON_NUM_CONFIGS = 8**, sim slice
  thickness 1.0 Å. `derive()` = production; `toy_params()` = fast (2 configs,
  2 Å slices, 20 recon slices, batch 16, crop 48). `--mini` also uses
  `toy_params()` but 4 tiles.
- **`simulate_4dstem.py`** — abTEM sim. FrozenPhonons (σ: Pb 0.092, Sr 0.085,
  Ti 0.072, O 0.085 Å, room-temp DWFs). **`probe.aberrations.defocus =
  -p.overfocus_a`** (NEGATIVE — verified to be the overfocus sign, see §4).
  PixelatedDetector at 200 mrad → native (976,1425) CBED → coarsen by K=15 →
  (65,95) saved per scan position. HAADF outer is runtime-clamped to 0.97× the
  actual sim angle. CLI: `--tile I J`, `--all`, `--toy`, `--mini`, `--overwrite`,
  `--num-configs N`. Sim is FAST: ~2.6 min/tile toy, ~10–20 min/tile production
  on the GPU (the original handover's "8–13 h/tile" was a ~50× over-estimate —
  probably a CPU figure).
- **`reconstruct_ptycho.py`** — py4DSTEM `MultislicePtychography`. Loads tile
  zarrs, **resamples the CBED (65,95)→(65,65)** by scipy zoom (py4DSTEM rejects
  non-uniform Q resampling factors; the non-square cell makes the two diffraction
  axes have different mrad/px). Q calibration = `bf_eff_pixel_mrad / (1000·λ)`
  (physically derived, not the notebook's `BF_DOWNSAMPLE/REAL_SPACE_THICKNESS`
  hack). **`polar_parameters = {"C10": +p.overfocus_a}`** (POSITIVE — matches the
  sim's negative abTEM defocus since abTEM `defocus = -C10`; both put C10 = +10
  in the χ formula). Two stages:
  - Stage A: 24 iter, `fix_probe=True`, `step_size=0.025`, kz-reg OFF,
    gaussian σ=0.1, `fix_potential_baseline=True`. (kz-reg in Stage A *diverges
    to NaN* on thin data — keep it off here.)
  - Stage B: 96 iter, `step_size=0.05`, `fix_probe=False`,
    `fit_probe_aberrations=True`, **`fit_probe_aberrations_using_scikit_image=
    False`** (the scikit-image unwrapper has a documented kernel-hang bug — this
    was the "Stage B hangs at 0 iterations" we hit), `constrain_probe_amplitude=
    True`, **`pure_phase_object=True`** (PTO/STO is ~pure phase; pins |obj|=1,
    kills the amplitude/phase degeneracy that was eating depth info → |obj| went
    from 0.0007..1.0 garbage to a clean 1/1/1), **`kz_regularization_filter=True,
    gamma=0.2`** (ablated up from 0.02; 0.02→0.1→0.2 took std ratio 1.07→1.49→
    1.81 but never moved the kz-FRC ratio — so gamma is NOT the depth lever),
    **`tv_denoise=True, tv_denoise_weights=[1e-3, 1e-4]`** ([kz, kxy]; piecewise-
    depth prior — added for production, made the envelope smoother, didn't create
    real depth structure). CUDA mempool flushed between stages (else py4DSTEM can
    deadlock at iter 0 on a fragmented heap). `--stage B` alone is *blocked*
    (raises) — Stage A must precede it in the same process (the ptycho object
    isn't serialisable; running B alone gave 100% NaN).
- **`validate.py`** — code-path gate (NaN check) → per-slice std ratio → kz-FRC
  (even/odd object rows, ratio of high-kz to kz=0 power) → column centroids CSV +
  `validation_plots/<name>_summary.png` (per-slice std / kz-FRC / XZ slice).
  Modes: `production` (std ≥2.0, kz ratio >0.05), `mini` (≥1.3, >0.02), `toy`
  (≥1.0, >0.0 — code-path only). NOTE: the std-ratio test is weak (envelope, not
  structure); the kz-FRC is the meaningful one and it fails hard.
- **`diagnostics.py` / `diagnostics_report.md`** — Phase-0 sanity checks (all
  pass except a pre-existing data FAIL). Geometry, parameter arithmetic, detector
  design study. Worth reading for the *why* behind the detector binning.
- **`test_probe_focus.py`** — standalone: propagates a vacuum probe through free
  space, tracks peak intensity vs z, prints whether `defocus=+10` re-focuses
  inside the sample (underfocus) or spreads monotonically (overfocus). Use the
  PEAK column, not RMS (RMS is junk on a 0.12 Å focus / 0.15 Å grid).
- **`run_production.ps1`** — unattended runner: `simulate_4dstem.py --all
  --overwrite` → `reconstruct_ptycho.py --stage both` → `validate.py`, logged via
  Start-Transcript, aborts on non-zero exit. ~3.5–4 h total. **`--overwrite` is
  mandatory** there because `tile00..tile11` on disk are mini-test tiles
  (toy params) that must be regenerated with production params.

### How to run things
```
# toy code-path gate (~18 min, 1 tile)
python simulate_4dstem.py --tile 1 1 --toy
python reconstruct_ptycho.py --toy --stage both
python validate.py --toy

# mini quality gate (~3 h sim + ~13 min recon, 4 tiles, 8×8 Å)
python simulate_4dstem.py --mini --overwrite
python reconstruct_ptycho.py --mini --stage both
python validate.py --mini

# production (~3.5–4 h)
powershell -ExecutionPolicy Bypass -File run_production.ps1
```

---

## 3. The result, and the comparison to the old notebook

Production output: `ptycho_recon.zarr` (object_complex / object_phase / probe),
`ptycho_recon_metadata.json`, `validation_plots/ptycho_recon_summary.png`.

| | Old notebook (`image.png`) | Our pipeline (`ptycho_recon_summary.png`) |
|---|---|---|
| Slices | 98 (from wrong BOX_Y thickness) | 74 (correct BOX_Z, 1 Å/slice) |
| Object amplitude | varies (no pure-phase) → artifacts | pinned |obj|=1, clean |
| Probe sign | `defocus=-10`/`C10=+10` (correct, by luck) | `defocus=-10`/`C10=+10` (correct, verified) |
| kz-reg / TV-z | none | gamma 0.2 / TV-z on |
| XY-at-depth | look different across z — but that's mostly the envelope weighting (bottom slices higher contrast ⇒ "sharper"), not depth structure | — |
| XZ panel | vertical streaks, **asymmetric envelope** skewed to z≈40–70 (probably the BOX_Y mislabel + amplitude artifacts) | vertical streaks, **symmetric hump** envelope |
| kz-FRC | (not computed) | **single delta at kz=0, nothing else** — pure projection |
| Verdict | axially blurred | axially blurred, but *cleaner* (honest projection vs artefactual one) |

**Bottom line: neither depth-sections.** Our pipeline is a genuine improvement in
*correctness and cleanliness* (right beam axis, verified probe sign, pure-phase
object, no amplitude artifacts, no NaN, defensible detector/calibration) but it
did **not** fix the axial blur, which was the whole point. The plan's hypothesis
(blur ⇐ missing phonons + wrong C10 sign + kz-reg-off + over-fine slices) was
**wrong** — we addressed all of it and the blur remains.

---

## 4. The probe-sign question (settled)

abTEM (`abtem/transfer.py:807-814`): `defocus = -C10`; the setter does `C10 =
-value`. Both abTEM and py4DSTEM use `χ(α) = ½·α²·C10`. Empirically
(`test_probe_focus.py`): `abtem.defocus = +10` re-focuses the probe ~10 Å
*downstream* (inside the crystal) = **underfocus**; `abtem.defocus = -10` spreads
the probe monotonically from z=0 = crossover is *above* the sample = **overfocus
(focused before entry)**. Convention chain agrees (negative C10 = underfocus in
the Krivanek convention; abTEM defocus = −C10 ⇒ negative defocus = overfocus).

So for "probe focused 10 Å before the sample":
- sim: `probe.aberrations.defocus = -OVERFOCUS_A`  ⇒ abTEM C10 = +OVERFOCUS_A
- recon: `polar_parameters['C10'] = +OVERFOCUS_A`  (same coefficient)

Both currently set this way. The original plan doc said `C10 = -OVERFOCUS_A` —
**that was backwards**. The notebook had it right. Earlier in this project's
history the rebuild briefly had sim `defocus=+10` (underfocus) + recon `C10=-10`
(self-consistent but wrong physical direction); now fixed.

**Constraint from the user (important for ways-forward):** we *cannot* focus the
probe inside the material — at any reasonable raster step the overlap of a sharp
probe would be terrible. So "focus in the middle" is OFF the table. (The
reference paper agrees in spirit: it uses a *huge* overfocus, never converges in
the sample — see §5.)

---

## 5. What the reference paper actually does (`paper.pdf`)

**Lei & Wang, "Multislice Hollow Ptychography for Simultaneous Atomic-Layer-
Resolved 3D Structural Imaging and Spectroscopy"** (Warwick). Test material:
**21 nm PrScO₃ (PSO)** + a single Pr atom for depth-resolution characterisation.
They DO get atomic-layer 3D. Their recipe vs ours:

| | Paper | Us |
|---|---|---|
| Convergence semi-angle | sims sweep **20 / 40 / 80 mrad**; expt data 21.4 mrad | 100 mrad |
| Probe defocus | **~20 nm overfocus** (expt) — huge, never converges in sample | 1 nm (10 Å) overfocus |
| Scan | 64×64, step 0.41 Å, 2.6×2.6 nm (≈676 px, ~26×26 Å) | 3×3 tiles of 16×16, step 0.25 Å, 12×12 Å |
| Detector | EMPAD 128×128, optional **hollow mask** (BF disk removed, for EELS) | pixelated 200 mrad → binned (65,95) |
| Sample thickness | 21 nm | 7.4 nm |
| Engine | **PtychoShelves, LSQ-ML** (least-sq max-likelihood) with probe-position / detector-rotation / Fourier-misalignment refinement + momentum acceleration | py4DSTEM `MultislicePtychography`, plain gradient descent |
| Dose for full 3D atomic-layer | **≥10⁸ e⁻/Å²**, ideally **cryo** (depth res is "sensitive to lattice vibrations") | noiseless sim, room-temp DWFs |

Key physics from the paper:
- **Depth resolution is primarily set by the convergence semi-angle (CSA)**:
  ~9.5 Å at 20 mrad → ~3.5 Å at 40 mrad → ~2.8 Å at 80 mrad (FWHM on a single-atom
  test). "Increasing CSA narrows the depth of focus and enhances sensitivity to
  z-localized phase contrast." ⇒ **our 100 mrad is GOOD for depth — not the
  problem.** Theoretical dz ≈ λ/α² ≈ 2 Å for us.
- Hollow geometry (masking the BF disk) is for EELS compatibility; depth
  resolution is "largely independent of the hollow geometry until HSA > 0.95α".
  We don't need to go hollow for depth.
- "To resolve features ... in depth, their separation must exceed the theoretical
  depth resolution, typically **by at least a factor two**." So at ~2 Å theoretical
  we can resolve depth features ≥~4 Å apart. The vortex's polarisation rotates
  *continuously* over ~7 nm — that's a smooth depth modulation, arguably easier
  than discrete 4 Å layers, but it's a different beast.
- Full atomic-layer 3D needs **ultrahigh dose (≥10⁸ e⁻/Å²)** and cryo helps a lot.
  Chen et al. 2023 (Science 380, 633) got only **~3.9 nm** depth res in thick
  strong scatterers. Dong et al. 2025 (Nat Commun 16, 1219): "**sub-nanometer**
  depth resolution ... by **tilt-coupled** multislice electron ptychography" —
  i.e. the state of the art for sub-nm depth needs *tilts*.

**Interpretation:** single-perpendicular-scan MSPT depth sectioning is, in
practice, *coarse* (nm-scale) unless you throw ultrahigh dose + cryo + a strong
engine at it; sub-nm/atomic depth in thick samples generally needs tilt-coupling.
The polarisation tornado over 7 nm might be visible at ~1 nm depth resolution
(that would show the rotation), but our pipeline isn't even getting that — it's a
flat projection. The two most likely culprits are **the reconstruction engine**
(py4DSTEM GD vs PtychoShelves LSQML with position/rotation refinement + momentum)
and possibly **too little scan area** (12×12 Å vs the paper's ~26×26 Å).

---

## 6. Assessment — why it's a projection, ranked by likelihood

1. **Reconstruction engine.** py4DSTEM's `MultislicePtychography` is plain
   gradient descent with a kz filter; no probe-position correction, no detector-
   rotation refinement, no momentum/LSQML. The paper (and most of the depth-
   sectioning literature) uses **PtychoShelves / fold_slice (LSQML)**. The depth
   information is subtle (it's in the Fresnel-propagation differences between
   slices); a weak optimiser may simply settle into the projection minimum that a
   stronger one escapes. *We have hit four separate py4DSTEM rough edges this
   project (scikit-image hang, mempool deadlock, NaN on under-constrained data,
   the kz-reg-in-Stage-A divergence) — the tool is fightable but not robust.*
2. **Too little scan area / overlap diversity.** 12×12 Å (3×3 tiles, 48×48
   positions) vs the paper's ~26×26 Å (64×64). More positions ⇒ better-conditioned
   inversion ⇒ more room for depth structure to be constrained rather than
   regularised flat. Cheap-ish to test (more tiles).
3. **Overfocus too small.** 10 Å vs the paper's ~20 nm. A bigger defocus spreads
   the probe over many unit cells ⇒ heavier overlap ⇒ better conditioning. At
   α=100 mrad even a 100–200 Å overfocus keeps the probe ~10–20 Å wide (fine for
   ptychography; resolution comes from the diffraction angle, not probe size).
   Cheap to test (one param + re-sim).
4. **Lattice vibrations (room-temp DWFs).** The paper: depth res is "sensitive to
   lattice vibrations", cryo helps. Our σ are room-temp. Halving σ (simulate ~LN2)
   might sharpen depth contrast. Cheap to test.
5. **Single perpendicular scan is just not enough.** Per the literature, sub-nm
   depth in thick samples needs tilt-coupling. If 1–4 don't crack it, the honest
   answer is the experiment design needs tilts (or the goal is nm-scale, not 2 Å).

---

## 7. Ways forward — recommended order

**(W1) Switch the reconstruction engine to fold_slice / PtychoShelves.** Highest
expected payoff. fold_slice (Yi Jiang's electron-ptychography fork of
PtychoShelves) is the code the depth-sectioning papers actually use. Cost: ~1–2
days (MATLAB; re-export the stitched 4D data to their HDF5 layout; learn the
param struct). Keep the abTEM sim outputs as-is — the tile zarrs are fine. This
is the move I'd make first, because §6.1 is the most likely root cause and no
amount of py4DSTEM-knob-twiddling has touched the kz-FRC.

**(W2) Bigger scan area + bigger overfocus, still in py4DSTEM.** Before/instead of
W1 if you'd rather stay in Python: (a) extend the scan to ~25×25 Å — the
`simulate_4dstem.py` tile machinery already supports a bigger grid; you'd want
~6×6 tiles of 4 Å or larger tiles; (b) bump `OVERFOCUS_NM` to ~10–20 nm in
`params.py` (re-sim needed — note the scan step is derived from the effective
probe size, so it'll auto-coarsen, which also cuts position count); (c) keep
pure-phase + TV-z + kz-reg as-is. Cheap-ish (a few hours). If this still gives a
kz-FRC delta at zero, that's strong evidence the engine (W1) is the blocker.

**(W3) Cryo DWFs.** Halve the phonon σ in `params.py` (`PHONON_SIGMAS_A`) and
re-sim a `--mini`. Quick. Per the paper this can meaningfully sharpen depth
contrast. Stack with W2.

**(W4) Tilt-coupled MSPT.** If W1–W3 disappoint, this is the literature answer for
sub-nm/atomic depth in thick samples (Dong et al. 2025). Simulate 2–3 tilts
(e.g. 0°, ±10–15° about an in-plane axis), reconstruct jointly. abTEM can do the
tilted sims; the joint reconstruction needs PtychoShelves/fold_slice anyway, so
this naturally follows W1. Bigger lift, but it's the surest path to actually
seeing the tornado in depth.

**(W5) Re-scope.** If the goal is "see the polarisation rotate with depth", maybe
~1 nm depth resolution is enough (it would resolve the rotation over a ~7 nm
column into ~7 layers). That's more achievable than 2 Å. Decide what "good
enough" means before spending more cycles chasing atomic-layer depth.

**Not recommended:** more kz-reg gamma (exhausted — moved std ratio, not kz-FRC),
more TV-z weight (smooths the envelope, doesn't create structure), focusing the
probe inside the sample (user constraint: kills lateral overlap), more recon
iterations (it's converged — to a projection).

---

## 8. File map / references

In `C:\Users\Trist\HyperSpy-bundle\MY HYPERSPY CODE\PTOSTO\`:
- `params.py`, `simulate_4dstem.py`, `reconstruct_ptycho.py`, `validate.py`,
  `diagnostics.py` — the pipeline (see §2).
- `test_probe_focus.py` — vacuum-probe focus/sign test (see §4).
- `run_production.ps1` — unattended production runner.
- `diagnostics_report.md` — Phase-0 sanity checks output.
- `paper.pdf` — Lei & Wang, *Multislice Hollow Ptychography* (the reference; §5).
- `PTO6_STO6_18_18_labyrinthPoscar.vasp` — the structure (vortex displacements =
  validation ground truth).
- `TileDataAnalysis_WORKS copy 2.ipynb` — the notebook we're improving from
  (read-only scratch). `image.png` = its 3D reconstruction output (the "before").
- `ptycho_recon.zarr` + `ptycho_recon_metadata.json` +
  `validation_plots/ptycho_recon_summary.png` — current production output (the
  "after" — still a projection).
- `ptycho_recon_mini.zarr` / `ptycho_recon_toy.zarr` (+ `_metadata.json`,
  `validation_plots/*_summary.png`) — gating-run outputs.
- `tile{ij}_bf.zarr` (ij ∈ 00..22) — 4D-STEM BF data per tile. **After the last
  `run_production.ps1` these are production params (8 configs, 1 Å slices)** for
  all 9; before that they were mini params for 00/01/10/11.
- `WORKSTOP/WORKS/TileDataAnalysis_WORKS.ipynb` (parent dir) — the other "WORKS"
  notebook the original plan cited for the C10 sign.

Outside the project dir:
- `C:\Users\Trist\.claude\plans\you-are-a-senior-cozy-fairy.md` — the original
  improvement plan (note: its C10-sign claim was backwards; otherwise still a good
  context doc for the root-cause analysis and the detector design study).

Environment: Python `C:\Users\Trist\HyperSpy-bundle\python.exe`, abtem 1.0.5,
py4DSTEM 0.14.18, zarr 2.18.4, CUDA 13.1, 8 GB VRAM, Windows 10. GPU sim is fast
(~10–20 min/tile production). Recon Stage A ~3 min, Stage B ~10 min (toy/mini);
production recon (74 slices, batch 64, crop 64) ~20–40 min.

---

## 9. One-paragraph summary for the next assistant

The `.py` pipeline is correct and clean (verified probe sign, right beam axis,
pure-phase object, no NaN, defensible calibration) but it produces a *projection*,
not a depth-sectioned volume — same axial-blur failure as the notebook it
replaced. We've ruled out the original plan's hypotheses (phonons, C10 sign,
kz-reg, slice count) and the kz-reg/TV-z knobs (they move the meaningless std
ratio, not the meaningful kz-FRC, which stays a delta at kz=0). The reference
paper gets atomic-layer 3D with **PtychoShelves/LSQML + much bigger overfocus +
bigger scan + ultrahigh dose + (for sub-nm) tilt-coupling**; our 100 mrad
convergence is fine (it's actually *good* for depth). Recommended next step: port
the reconstruction to fold_slice/PtychoShelves (W1), optionally first try a bigger
scan area + bigger overfocus + cryo DWFs in py4DSTEM (W2+W3) as a cheap check; if
depth still won't come, the experiment likely needs tilts (W4) or the goal should
be re-scoped to ~1 nm depth (W5). Do **not** focus the probe inside the sample
(kills lateral overlap — hard constraint from the user).
