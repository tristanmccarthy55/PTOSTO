# Lab book — PTO/STO 4D-STEM multislice ptychography

**Author:** Tristan McCarthy &middot; **Date:** 2026-05-13 &middot; **For:** supervisor meeting

---

## 1. Goal

Simulate 4D-STEM of a **PTO6/STO6 ferroelectric labyrinth** (POSCAR
`PTO6_STO6_18_18_labyrinthPoscar.vasp`, ~19 140 atoms, ~74 Å along the beam),
reconstruct a 3D phase volume with multislice electron ptychography, and
resolve the polarisation "tornado" — i.e. see the ferroelectric displacements
*change with depth*. Target: ~2 Å axial resolution. Reference: Lei & Wang,
*Multislice Hollow Ptychography for Simultaneous Atomic-Layer-Resolved 3D
Structural Imaging and Spectroscopy* (`paper.pdf`).

The starting point was `TileDataAnalysis_WORKS copy 2.ipynb`, whose 3D
reconstruction (`ptycho_3d_final.png`) is well-resolved in XY but **axially
blurred** — atom columns smear through the whole thickness.

## 2. What we did: notebook → modular `.py` pipeline

The notebook was monolithic, mixed parameters with execution, and didn't allow
clean ablation. Rebuilt as six scripts with a single source of truth:

| File | Role |
|---|---|
| `params.py` | All experimental constants and a `derive()` that produces every sampling / detector / slicing param. Frozen dataclass shared by every other script. |
| `simulate_4dstem.py` | abTEM 4D-STEM tile simulation, GPU, with `FrozenPhonons` (cryo σ ×0.65 default). |
| `reconstruct_ptycho.py` | py4DSTEM `MultislicePtychography`, two-stage (probe-fixed warm-up → release + priors). |
| `validate.py` | Quantitative gates: NaN check, per-slice std ratio, depth-direction FRC, column-centroid CSV, plotting (XY slice montage, strip-averaged XZ/YZ, deviation-from-projection panels). |
| `diagnostics.py` | Phase-0 sanity checks (geometry, beam direction, parameter arithmetic, detector design study). |
| `derive_scan.py` | **Analytical** scan-design sweep + probe-vs-cell clearance check. Runs in seconds, decides params *before* committing 4 h+ of GPU. |
| `run_production.ps1` | Unattended sim → recon → validate, logging to `production_run_<stamp>.log`. |

Everything is reproducible from `python params.py` + the one-line PowerShell
command — no notebook state, no hidden config.

## 3. Notebook bugs we found and fixed

Quantitative, not stylistic — these all affected the result:

1. **Beam-axis bug.** Notebook used `BOX_Y` (≈ 70 Å) as the through-beam thickness; abTEM propagates along +Z after `orthogonalize_cell + center(axis=2)`, so the correct thickness is `BOX_Z` ≈ 73.93 Å. This propagated into the recon's slice count and Q-pixel calibration.
2. **Probe defocus sign.** Verified empirically (`test_probe_focus.py`, vacuum-probe through free space): abTEM uses `defocus = −C10`. A *negative* abTEM defocus puts the probe crossover above the entrance surface = **overfocus**. py4DSTEM's `C10` is the same coefficient → C10 = `+overfocus_a`. Notebook had the sign right by luck; an earlier rebuild had it backwards.
3. **No frozen phonons.** Notebook simulated at 0 K. We now run 8 configurations with cryo Debye-Waller σ (Pb 0.060, Sr 0.055, Ti 0.047, O 0.055 Å, ×0.65 of room-temperature literature values).
4. **Amplitude/phase degeneracy was eating depth info.** Object was complex with free amplitude; depth info leaked into the amplitude channel. Fixed with `pure_phase_object=True` + `fix_potential_baseline=True` — pins |obj| = 1. |obj| went from a 0.0007..1.0 mess to a clean 1/1/1.
5. **Detector calibration.** Q-pixel size was computed as `BOX_Y / bin` (geometric guess); now derived from `bf_eff_pixel_mrad / (1000·λ)` (physically correct).
6. **Detector binning.** Auto-derived (K = 15) so the saved CBED is ≈ 64 px on the coarser axis — matches 8 GB VRAM after recon-time crop.
7. **HAADF outer angle clamping.** Notebook asked for 250 mrad; abTEM grid only supported ~197 mrad, so the detector silently truncated. Now runtime-clamped to 0.97 × actual sim max.
8. **scikit-image kernel hang.** py4DSTEM's `fit_probe_aberrations_using_scikit_image=True` (default) has a documented unwrapper hang. Disabled it.
9. **CUDA mempool fragmentation.** Stage B can deadlock at iter 0 on a fragmented heap left by Stage A. Now flushed (`cp.get_default_memory_pool().free_all_blocks()`) between stages.
10. **Probe-vs-cell wraparound.** abTEM uses a periodic cell. The defocused probe is widest at the *exit* surface (radius = α·(Δf + t)); if it exceeds the cell short axis (x ≈ 48 Å) it wraps and aliases the diffraction patterns. Now checked analytically in `derive_scan.py` *before* every parameter change.
11. **py4DSTEM auto-rotation.** `preprocess()` auto-fitted a 4°–5° rotation between scan and detector axes. abTEM builds them in the same frame, so the true rotation is 0 — the auto-fit was being biased by anisotropic CBED resampling. Forced to 0.0.

## 4. Improvements we tried (W2 + W3 + ablation) and what each told us

The original handover's "fix list" (defocus sign, phonons, kz-reg, slice count) had already been applied. The remaining ranked levers were W2 (bigger scan / bigger probe), W3 (cryo DWFs), W1 (switch engine), W4 (tilts), W5 (re-scope).

### 4a — W2: bigger scan, bigger overfocus

|  | notebook / handover baseline | now (W2) |
|---|---|---|
| Tile grid | 3×3 of 4 Å = 12×12 Å | **5×5 of 4 Å = 20×20 Å** |
| Overfocus | 1 nm | **5 nm** (probe footprint Ø 10 Å entrance, 25 Å exit) |
| Target overlap | 75 % | **95 %** (= 97.5 % realized linear overlap) |
| Scan step | 0.25 Å | **0.25 Å** (same — the bigger probe and higher overlap balance) |
| Positions per tile / total | 256 / 2304 | **256 / 6400** (above the paper's 64×64 = 4096) |
| Convergence | 100 mrad | 100 mrad (gives dz ≈ λ/α² ≈ 2 Å — **already good for depth**, vs the paper's 2.8–9.5 Å) |

**Why not the paper's 20 nm overfocus:** the geometric probe footprint scales
as 2·α·Δf. At our 100 mrad (5× the paper's 21 mrad) the same defocus gives a
5× wider probe. At 20 nm overfocus our probe Ø would be 55 Å at the exit
surface — **wider than the 48 Å abTEM cell short axis**, so it wraps and
aliases. The cell width, not numerical taste, is the binding constraint.
`derive_scan.py` produces the clearance table:

```
ovf nm | probeØ exit | x-overhang | verdict
   1.0 |     16.8 Å  |     0      | OK — fits
   5.0 |     24.8 Å  |     0      | OK — fits   ← chosen
  10.0 |     34.8 Å  |   4.4 Å    | MARGINAL — aliasing likely
  20.0 |     54.8 Å  |  14.4 Å    | UNSAFE — wraps
```

### 4b — W3: cryo phonons

Phonon σ × 0.65 (room-temp → ~LN2). Baked in as the default; `--room` for the
ablation.

### 4c — Production-recon results (W2 + W3 together)

After 4.3 h of GPU simulation and 1.2 h of GPU reconstruction:

```
object_phase: (74, 484, 484) complex64, NaN = 0%
|obj| min/med/max  =  1 / 1 / 1                  (pure-phase, clean)
phase min/max       =  -0.019 / 0.020 rad        (per slice)
per-slice std ratio =  5.30   (PASS — but see below)
kz-FRC ratio        =  6.8e-6 (FAIL — want > 0.05; off by ~4 orders of mag)
```

**The per-slice std PASS is meaningless** — it's the *reconstruction-confidence
envelope* (low contrast at the surfaces, hump where probe + thickness give the
best statistics), not depth-resolved structure. The handover explicitly flagged
this test as weak; we kept it for back-compat.

**The phase magnitude is consistent with a projection:** ±0.02 rad/slice ×
74 slices ≈ 1.5 rad integrated, which is what the projected potential of
this crystal should give. The recon is finding the right *total* phase and
smearing it uniformly over z.

### 4d — Ablation: kz-regularisation + TV-denoise OFF

py4DSTEM's `kz_regularization_filter` is literally a low-pass in kz —
suppresses the high-kz content that *would* encode atomic layers. Hypothesis:
the priors are killing depth signal we'd otherwise see. Run Stage B with
both off, otherwise identical, no re-sim needed.

```
kz-FRC ratio: 1.8e-5 (priors on)  →  5e-4 (priors off)    ~30× up
per-slice std ratio:    5.30           →   7.41           more contrast
phase min/max:    ±0.020 rad           →   ±0.024 rad
```

Higher than with priors on, but still ~100× below threshold. **The regularisation
wasn't what was pinning the projection** — turning it off didn't unlock depth
structure, just made the same z-uniform projection slightly less smooth.

## 5. The interesting discovery: the **anti-symmetric depth mode**

After upgrading `validate.py` to produce strip-averaged XZ/YZ, a per-pixel
"std-along-Z" map, and (most usefully) the **(phase − z-mean)** deviation
panels, a clean pattern emerged that wasn't visible in the old single-line XZ
plot:

- **Lateral resolution is excellent.** Individual perovskite columns are
  resolved at z = 29, 55, 69 Å — the perovskite lattice is unmistakable.
- **Columns do NOT move with z.** Same (x, y) in every slice. The recon is
  laterally a projection, not a depth-sectioned volume.
- **But the deviation panels are not flat.** A genuine projection would give
  `phase − z-mean = 0` everywhere. Ours gives a coherent, column-aligned
  pattern with a **clean sign-flip at z ≈ 40**: deviation is positive at the
  column centres for z = 20–35, *negative* at the same column centres for
  z = 45–65. The same anti-symmetry shows in the 1-D per-slice std curve
  (peak at z = 28, dip at z = 42, peak at z = 55–70).

**Interpretation.** The data contains some depth information — otherwise the
deviation would be zero, not anti-symmetric. Plain gradient descent files
that information into the *lowest-frequency non-projection mode* (the
anti-symmetric one) because:

1. The projected phase is well-constrained → z-mean is locked.
2. Atomic-layer depth modes are weakly constrained (inter-slice Fresnel
   propagation differences are small at this thickness / dose).
3. Between the two, the smoothest non-projection answer is anti-symmetric
   about mid-z — and the optimiser dumps the residual depth signal there
   because that minimises the smoothness penalty.

**Why "the weak slices" (z = 3, 16, 42) look bad** is the same artefact: they
sit at the *nodes* of the anti-symmetric mode where deviation ≈ 0, so the
displayed phase ≈ z-mean ≈ small. z = 29, 55, 69 sit at the *antinodes* and
get |deviation| ≈ peak. One artefact, multiple manifestations.

This is exactly the failure mode that **LSQ-ML + momentum + position
refinement** (PtychoShelves / fold_slice) is designed to break: the momentum
term resists low-kz attractors, and probe-position refinement adds constraints
that the smooth anti-symmetric mode cannot satisfy.

## 6. What has now been ruled out

| Hypothesis | Status |
|---|---|
| Frozen phonons missing | Fixed (8 configs). No improvement. |
| Beam axis wrong (BOX_Y) | Fixed (BOX_Z). No improvement. |
| C10 sign flipped | Verified correct. No improvement. |
| Slice count wrong | Now 74 slices × 1 Å (matches dz_optical = 2 Å with 2× oversampling). No improvement. |
| kz-regularisation off | Toggled on (γ 0.02 → 0.2) and off. Moves the envelope, not the kz-FRC. |
| TV-z denoise off | Toggled on and off. Same. |
| Amplitude/phase degeneracy | Fixed (`pure_phase_object`). Cleaner output, same projection. |
| Scan area too small | Doubled (12 Å → 20 Å), 2.8× more positions. Same. |
| Overfocus too small | 1 nm → 5 nm (the cell-safe maximum). Same. |
| Detector design | Analytically optimised. Same. |
| Room-temp DWFs | Cryo σ × 0.65. Same. |
| Auto-rotation calibration | Forced to 0°. Same. |

Every accessible knob in the abTEM + py4DSTEM domain has now been exercised.
The result is robust: **a clean, well-conditioned, cryo, big-scan, big-probe
4D-STEM dataset of a 74 Å labyrinth produces, in py4DSTEM
`MultislicePtychography`, a z-uniform phase projection plus an anti-symmetric
depth-mode artefact**.

## 7. What this tells us

1. The **data are information-bearing in z** — the anti-symmetric deviation is
   proof that depth information is in the diffraction patterns. The bottleneck
   is *extraction*, not *acquisition*.
2. **py4DSTEM's plain gradient-descent multislice ptychography cannot extract
   atomic-layer depth from a single perpendicular scan of a thick strong
   scatterer.** It settles into the smoothest non-projection minimum
   regardless of conditioning, regularisation, scan size, or probe shape.
3. This matches the literature. The reference paper (Lei & Wang) and every
   atomic-layer-depth-sectioning paper we've found (Chen *et al.* Science 2023,
   Dong *et al.* Nat Commun 2025) uses **PtychoShelves / fold_slice with
   LSQ-ML, position refinement, detector-rotation refinement, and momentum
   acceleration** — none of which exists in py4DSTEM MSPT.
4. Our 100 mrad convergence gives dz_optical ≈ 2 Å — *better* than the
   reference paper's setup. The depth resolution ceiling is not the
   experiment; it's the optimiser.

## 8. Recommended next step: W1 — fold_slice / PtychoShelves port

Yi Jiang's electron-ptychography fork of PtychoShelves (MATLAB) is the engine
the depth-sectioning papers actually use. Estimated effort **1–2 days**.

- The 25 cryo abTEM tile zarrs (`tile00..tile44_bf.zarr`, ~200 MB total) are
  **reusable as-is**. Only the export bridge and the parameter struct need to
  be built.
- Export bridge: load the stitched 4D array (`load_and_stitch` in
  `reconstruct_ptycho.py` already does this), write it to fold_slice's HDF5
  layout (`dp` / `probe_params` / `param`), with per-axis Q-pixel calibration
  so we *don't* need the anisotropic 65×95 → 65×65 zoom that's currently
  biasing py4DSTEM's auto-rotation.
- First reconstruction: replicate Stage B (release probe, fit aberrations,
  momentum on, position refinement on) and look at the same XZ-deviation
  panels. The question we want answered: does the anti-symmetric mode collapse
  into something layer-localised when LSQML + momentum get hold of it?

If single-scan-still-fails after W1 → **W4 (tilt-coupled MSPT)**, which needs
fold_slice anyway. Dong *et al.* 2025 is the recipe — 2–3 tilts at ±10–15°,
joint reconstruction.

## 9. Open questions for discussion

1. **Target depth resolution.** Atomic-layer (~2 Å) is paper-class and what
   we've been aiming for. The polarisation rotates *continuously* over the 7 nm
   tornado — so resolving it at ~1 nm would already split the rotation into
   ~7 angular bins. If ~1 nm is enough, the W1 single-scan port has a good
   shot. If we need true atomic-layer depth → likely W1 + W4 (tilts).
2. **MATLAB licensing for PtychoShelves.** I have access via the campus
   network; need to confirm a stable build path on the local Windows machine
   (vs. the Linux cluster).
3. **Re-scoping (W5).** Is "resolve the polarisation rotation, not necessarily
   atomic columns in z" an acceptable refinement of the goal if W1
   single-scan turns out to be ~1 nm not ~2 Å? My read: yes — the science is
   the rotation, not the atomic-layer demonstration.
4. **Tilt simulation budget.** A single 6400-position abTEM tile run is ~4 h;
   3 tilts × 25 tiles is ~12 days of GPU time, or a few days on a
   bigger box. Worth it only after W1 shows what single-scan can do.

## 10. Files referenced

In `c:\Users\Trist\HyperSpy-bundle\MY HYPERSPY CODE\PTOSTO\`:

- **Pipeline:** [params.py](params.py), [simulate_4dstem.py](simulate_4dstem.py), [reconstruct_ptycho.py](reconstruct_ptycho.py), [validate.py](validate.py), [diagnostics.py](diagnostics.py), [derive_scan.py](derive_scan.py), [run_production.ps1](run_production.ps1).
- **Strategic doc:** [HANDOVER.md](HANDOVER.md) (original problem statement and ranked options).
- **Reference:** `paper.pdf` (Lei & Wang).
- **Structure:** `PTO6_STO6_18_18_labyrinthPoscar.vasp`.
- **Production output (this run):** `ptycho_recon.zarr` + `..._metadata.json` (6400 positions, cryo, 95 % overlap).
- **Ablation output:** `ptycho_recon_nodepthreg.zarr` (kz-reg + TV off).
- **Plots:** [validation_plots/ptycho_recon_summary.png](validation_plots/ptycho_recon_summary.png) (2×3 grid with the depth-mode deviation panel), [validation_plots/ptycho_recon_slices.png](validation_plots/ptycho_recon_slices.png) (XY montage at 6 z-values with deviation row).
- **Tile data (reusable for W1):** `tile00_bf.zarr` … `tile44_bf.zarr`, each (16, 16, 65, 95) float32 zstd-compressed.
- **Run log:** `production_run_20260512_234211.log`.

## 11. Wall-clock and cost summary

| Run | Sim time | Recon time | Notes |
|---|---|---|---|
| Toy gate (1 tile, toy params) | ~3 min | ~18 min | Code-path test; PASSES |
| W2 diagnostic (5×5, 1600 pos, cryo, 0.90 overlap) | 1.08 h | 19 min | Projection |
| W2 production (5×5, 6400 pos, cryo, 0.95 overlap) | 4.28 h | 73 min | Projection + anti-symmetric mode |
| Ablation (no kz-reg, no TV, 6400 pos) | 0 | ~73 min | Same projection, slightly less smooth |

All on the local 8 GB VRAM GPU (CUDA 13.1) under Windows 10 / PowerShell.
Total disk for production tiles: ~200 MB. Stitched 4D in RAM: 0.16 GB.

---

## 12. W1 fold_slice port — smoke ladder → production (2026-05-18/19)

Same 6400-position cryo data as W2 production, exported to fold_slice's HDF5
schema via `export_to_foldslice.py --pipeline`. CBED square-resampled 65×95 →
65×65 at the coarse mrad/px (= what py4DSTEM was already seeing — clean
engine-vs-engine A/B). Data rescaled to ~10⁴ counts/frame (abTEM emits
probability units, fold_slice's L1 metric needs counts or it NaNs at iter 1).

### 12.1 Smoke ladder — what each test isolated

Each smoke ran 3 iter / stage (≤ 5 min). The point was diagnosing the
NaN-at-iter-1 failure mode of the early full-config attempts, by collapsing
the recipe one knob at a time.

| Config | Nprobe | Nlayers | δz (Å) | Outcome | What it ruled out |
|---|---:|---:|---:|---|---|
| `baseline` | 1 | 1 | 73.9 | ✅ both stages clean | data + probe + scan setup all OK |
| `multimode` | 4 | 1 | 73.9 | ✅ (untested in full but inferred) | (mixed-mode probe is not the issue) |
| `multislice` (74 layers) | 1 | 74 | 1.0 | ❌ NaN iter 1 | **fine slices break Fresnel propagator at our 100 mrad α** |
| `multislice37` | 1 | 37 | 2.0 | ✅ both stages clean | confirms 2 Å = dz_optical = λ/α² is the floor |
| **Full Yu recipe** | 4 | 37 | 2.0 | ✅ both stages clean (5.2 min) | pipeline verified end-to-end |

Round-trip of every successful smoke through `validate.py` passed code-path
(no NaN) and the per-slice-std gate (1.7–2.7 ratio).

### 12.2 Production attempts and what each crash taught us

All at Nprobe=4, Nlayers=37, δz=2.0 Å, Stage A 40 iter + Stage B 200 iter.
**~3.3 h** wall-clock on the 8 GB 2070 SUPER (~42 s/iter).

| # | `probe_position_search` | `probe_change_start` (Stage B) | Outcome | Diagnosis |
|---|---:|---:|---|---|
| 1 | **20** | 1 | ❌ crashed Stage B iter 20 in `gradient_projection_solver` | Position refinement at iter 20 diverges on perfect (sim) positions — Yu's sim recipe: `=inf` |
| 2 | inf | **1** | ❌ NaN Stage B iter 1 (LSQML) | 40 iter of fixed-probe Stage A over-converged the object; first probe-release step in Stage B overshot to NaN. Yu's sim recipe: probe also `=inf` (perfect probe = never refine) |
| 3 | inf | inf | ✅ both stages, **2 h 48 min**, .mat saved | Full Yu sim recipe — fixed probe + fixed positions, object-only refinement |

So the operative recipe for this sim-data setup is the strictly conservative
end of Yu's reply: **don't refine what you already know**. With abTEM-constructed
probe and grid-perfect positions, both refinement loops are net-negative.

### 12.3 fold_slice patches required to get our data through

- `+engines/+GPU_MS/+shared/find_geom_correction.m` lines 60, 71, 81: wrapped
  `self.Np_p/2` in `floor()` so an **odd CBED size** (65) doesn't break
  `gpuArray/zeros` (which requires integer dims). PSO_science example used
  Ndp=128 so the bug never fired upstream.
- Project-side `recon.m`: `eng.save_results_every = eng.number_iterations`
  (fold_slice's default `save_every=50` was suppressing all saves when
  smoke runs had Niter < 50).

### 12.4 W1 result

`python validate.py --recon ptycho_recon_foldslice.zarr`:

- **code-path**: PASS (no NaN)
- **per-slice std ratio**: 1.76 (FAIL, gate ≥ 2.0 — but always the weaker metric per the handover, "envelope, not structure")
- **kz-FRC ratio**: **0.087** (PASS, gate ≥ 0.05). py4DSTEM's MSPT run on the *identical data* gave **6.8 × 10⁻⁶**. So fold_slice's LSQ-ML engine is **~13 000× better on the strongest depth-structure metric** without changing the data, the cell, or the slice count.

Visually (`validation_plots/ptycho_recon_foldslice_summary.png` and
`_slices.png`): the XZ-deviation panel now shows **column-localised vertical
streaks** rather than the smooth anti-symmetric envelope py4DSTEM produced;
the per-slice XY montages show atomic-column-resolved positive/negative dots
in the phase-minus-z-mean view. Some residual anti-symmetric character is
still visible (red-dominant z≈1–7 → blue-dominant z≈27–34) but is *carried
on top of* real per-column structure, not replacing it.

**Interpretation**: the engine port did exactly what § 8 hypothesised — LSQ-ML
+ mixed-mode probe + variable-probe broke through the gradient-descent failure
that pinned py4DSTEM to a z-uniform projection. The residual symmetric pattern
is the kind of artefact that **tilts (W4, ±10–15°)** are designed to break.
Single-perpendicular-scan at 100 mrad / 74 Å gets to "real-but-not-clean"
3D structure; clean atomic-layer separation likely needs tilts on top.

### 12.5 L-curve regularisation experiments (2026-05-19)

Three L-curve sweeps (Stage A 40 iter + Stage B 10/20/.../80 iter,
save_results_every=10, TIFF analysis via `lcurve.py`):

| Sweep | reg_mu | regularize_layers | kz-FRC @10 | kz-FRC @80 | Shape |
|---|---:|---:|---:|---:|---|
| #1 | 0 | 0 | 0.66 | 0.63 | monotonic decay |
| #2 | 1e-3 | 0.01 | 0.55 | 0.53 | interior peak at iter 20 (kz-FRC) / 40 (std) |
| #3 | 1e-3 | 0 | 0.66 | 0.63 | monotonic decay (= #1) |

Findings:

1. **`reg_mu` alone has no measurable effect** at 1e-3 (sweeps #1 and #3
   are statistically identical). Either the setting is too weak, or
   Tikhonov on the object phase isn't the right regulariser for this
   problem.
2. **`regularize_layers = 0.01` IS the only thing that introduces a clean
   L-curve knee** (sweep #2), but it bought that knee by suppressing
   the depth signal we're trying to detect — direct validate.py round-trip
   of the iter-30 .mat gave kz-FRC = 0.047 (vs 0.087 for the no-reg
   200-iter prod). Visually atomic columns are cleaner in the XY slices
   but the XZ-deviation panel becomes mottled noise instead of the
   column-localised vertical streaks the no-reg result showed.
3. **The "monotonic decay" in #1/#3 is a real problem, not a measurement
   artefact**: the engine is over-fitting multislice gauge swaps on our
   noiseless abTEM data (no Poisson noise to break the degeneracy as it
   would in Yu's experimental data). But the cure (layer symmetrisation)
   is worse than the disease.

**Conclusion: recon-side tuning has reached its limit on this data.** The
recon is fundamentally under-determined (~10:1 unknowns:constraints per
slice), so spurious gauge swaps amplify with iter and there's no
regularisation that suppresses them without also killing the real
depth-localised structure. Better data conditioning is required — i.e.
Phase C of the original W1 plan (cell extension + 20 nm overfocus +
denser scan to get the same overlap-diversity-per-pixel as the Lei & Wang
paper). Handover spec: [`W1C_PROMPT.md`](W1C_PROMPT.md).

### 12.6 New wall-clock entries

| Run | Sim time | Recon time | Notes |
|---|---|---|---|
| W1 smoke ladder (5× ~5 min each) | 0 | ~25 min | Diagnosis of NaN modes |
| W1 production attempt 1 | 0 | ~33 min (crashed at Stage B iter 20) | Position-refinement diverged |
| W1 production attempt 2 | 0 | ~32 min (crashed at Stage B iter 1) | Probe-release overshot |
| W1 production attempt 3 (success) | 0 | **2.80 h** | kz-FRC PASS, .mat saved |
| W1 L-curve sweep #1 (no reg, 80 iter) | 0 | ~91 min | monotonic decay confirmed |
| W1 L-curve sweep #2 (reg_mu + layer_reg) | 0 | ~91 min | knee at iter 20-40 but kz-FRC dropped |
| W1 L-curve sweep #3 (reg_mu only) | 0 | ~91 min | same as #1 — reg_mu no-op at 1e-3 |
| W1 optimised re-run (Stage A 40 + Stage B 30 + reg #2) | 0 | ~52 min | validated: std-ratio improved but kz-FRC dropped |

CUDA 11.2 (MATLAB R2022b bundled) + MSVC v142 toolset; `mexcuda` chain
required ~6 hours of debugging (Visual Studio Build Tools registration
broken on install → manual `HKLM\SOFTWARE\Microsoft\VisualStudio\SxS\VS7`
write, MATLAB mexopts XMLs hard-coded for VS Enterprise/Pro/Community
only → patched to recognise BuildTools, CUDA 11.2 vs MSVC v143 toolset
incompatibility → forced v142 via `vcvars_ver=14.29` + uninstall v143 +
drop a `vcvars64.bat` shim at the path nvcc 11.2 expects). Worth knowing
that "MATLAB + CUDA + fold_slice on Windows" is real work, not a default.

---

*End of lab book — last update 2026-05-19 (W1 added).*
