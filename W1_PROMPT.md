# W1 prompt — for a fresh chat in plan mode

*Copy the block below into a new chat that has plan mode enabled. The files it
references all live in this project root.*

---

I want to port my multislice ptychography reconstruction from py4DSTEM to **fold_slice / PtychoShelves** (Yi Jiang's electron-ptychography fork of PtychoShelves, MATLAB — repo: https://github.com/yijiang1/fold_slice). The abTEM simulation side is done and the data is sound. This is the **W1** step from `HANDOVER.md`.

Read in order:

1. `LABBOOK.md` — full technical record: what we built, what we've ruled out, and the **anti-symmetric depth-mode** finding that's the strongest evidence the py4DSTEM engine (not the data) is the bottleneck.
2. `meeting.md` — same story in plainer prose; useful for context.
3. `HANDOVER.md` — strategic context and ranked options (W1 in particular).
4. `params.py`, `simulate_4dstem.py`, `reconstruct_ptycho.py` — the existing pipeline. `load_and_stitch()` in `reconstruct_ptycho.py` already produces a stitched 4D-STEM array + calibration metadata; that's the natural export-bridge starting point.

Plan the following work in a single cohesive plan file. Run in plan mode. Use AskUserQuestion to resolve ambiguity in platform or scope.

## Phase A — build / install

fold_slice on Windows vs the campus Linux cluster. MATLAB version requirements. Whether to use Yi Jiang's main branch or a paper-specific fork.

**Important contact:** the first authors of the reference paper `paper.pdf` — **Yu Lei and Peng Wang** — are in my research group. I can ask them directly for a known-good install path and a parameter struct for a similar system (PrScO₃). The plan should explicitly note where to defer to them rather than re-deriving from scratch (engine config, position-refinement strength, momentum settings, detector-rotation refinement, multi-config probe).

## Phase B — export bridge

Take the stitched 4D-STEM datacube from `load_and_stitch()` and write it to fold_slice's expected HDF5 layout, including:

- Real-space pixel size: 0.25 Å (5×5 tiles × 4 Å scan, 0.25 Å step, 6400 positions, cryo).
- **Per-axis Q-pixel calibration** so we keep the native 65×95 CBED and skip the anisotropic 65×95→65×65 zoom that's been biasing py4DSTEM's auto-rotation. (Coarse axis: `bf_eff_pixel_mrad` ≈ 6.17 mrad/px; fine axis: derived from the other native pixel size.)
- Probe init: complex64, abTEM convention `defocus = −overfocus_a`; PtychoShelves equivalent `C10 = +overfocus_a` after sign-convention map.
- Scan-position list in Å (50×50 in-plane = 2500 positions… *actually* check this: the 6400 figure is `5×5 tiles × 16×16 = 6400` positions; verify the stitched scan grid is `40×40 = 1600` or `80×80 = 6400` from the load_and_stitch shape).

## Phase C — abTEM cell extension for bigger overfocus

Our current cell is **47.88 × 70.01 × 73.93 Å** with beam along Z. At our 100 mrad convergence the geometric probe footprint is `2·α·(Δf + t)` at the exit surface, and the 47.88 Å short axis caps us at ~5 nm overfocus before the periodic-cell probe wraparound kicks in (see the clearance table in `derive_scan.py`). The reference paper used ~20 nm overfocus at ~21 mrad convergence — we want to get close to that.

**The structure can be tiled in both in-plane directions, but NOT along the beam.** The beam looks down a polarisation **vortex core**, so:

- **Z (74 Å, beam):** **DO NOT extend.** That would extend the physical sample thickness, changing the physics, and we can't add atoms along the vortex axis we don't have ground truth for. Z stays as-is.
- **X (47.88 Å, the "sandwich" direction — PTO/STO layering normal):** the binding short axis. The labyrinth pattern of vortices is periodic along this axis, so the supercell is genuinely periodic here — tiling `(n_x, 1, 1)` adds more PTO/STO repeats and gives us more cell width for the probe to fit in. **This is the priority extension** since it's the one limiting overfocus.
- **Y (70.01 Å, the "bread" direction — perpendicular in-plane, along the perovskite-only periodicity, matches the "18_18" in the POSCAR filename = 18 × 3.905 Å):** already roomier. Tiling `(1, n_y, 1)` here is also fine (the structure is periodic) and gives extra margin / lets the probe footprint approach the box width on this axis too.

Concrete plan for the extension:

- One-line ASE op after `rotate(-90, 'y')` + `orthogonalize_cell` + `center(axis=2)`: `atoms = atoms * (n_x, n_y, 1)`.
- Verify by inspecting `load_atoms()` output that **X is the sandwich axis** (12 PTO+STO layers ≈ 48 Å) and **Y is bread** (18 perovskite UCs ≈ 70 Å) — the filename `PTO6_STO6_18_18_labyrinthPoscar.vasp` says 18×18 perovskite in-plane, and 18 × 3.905 ≈ 70.3 Å matches Y, not X. If X turns out to be bread and Y sandwich, swap the tile multipliers accordingly.
- Add `BREAD_REPLICATIONS` and `SANDWICH_REPLICATIONS` (or just `TILE_X`, `TILE_Y`) knobs to `params.py`. Suggested starting point: `(n_x, n_y, n_z) = (2, 1, 1)` ⇒ cell ~96 × 70 × 74 Å, enough for 10–15 nm overfocus on the sandwich axis with margin. Worth checking `(2, 2, 1)` too if VRAM allows (~96 × 140 × 74 Å, abTEM grid roughly 4× larger, sim time roughly 4× longer per tile).
- Update `derive_scan.py`'s probe-clearance check with the new cell so the table reads the safe overfocus limit automatically.

## Phase D — minimal first reconstruction in fold_slice

The equivalent of our Stage A + Stage B:

- Engine: **LSQ-ML**.
- Probe-position refinement: ON. Detector-rotation refinement: ON. Momentum: ON.
- Slices: 74 × 1 Å (matches the optical dz ≈ 2 Å with 2× oversampling).
- Probe init: `C10 = +overfocus_a`, semiangle 100 mrad, 300 keV.
- Object: pure-phase if available (equivalent to our `pure_phase_object=True`).

Defer the exact param values to Yu Lei / Peng Wang. Plan should produce a `recon.m` (or equivalent) skeleton with placeholder values for the params they'll fill in.

## Phase E — round-trip back to our validation

Save the fold_slice output object phase to `ptycho_recon_foldslice.zarr` in the same `(z, y, x)` layout that `validate.py` already expects, so we can run

```
python validate.py --recon ptycho_recon_foldslice.zarr
```

and get the same six-panel summary (per-slice std, log kz-FRC, per-pixel std along Z, strip-averaged XZ, strip-averaged YZ, **XZ deviation from z-mean**) and the XY slice montage. The success criterion is binary: **does the anti-symmetric depth-mode (the sign-flip-at-mid-z pattern visible in the current XZ-deviation panel) collapse into atomic-layer-localised structure under LSQ-ML + position refinement + momentum?**

## Deliverable

Write the plan to your plan file. Do not implement yet. Specifically pin down:

1. fold_slice install / build path (Windows vs cluster, MATLAB version).
2. The export-bridge file format (we want exact HDF5 dataset names + per-axis Q calibration).
3. The supercell-extension code change (which axis is sandwich vs bread, what `params.py` API).
4. The starting recon parameter struct (placeholders, to be filled by Yu/Peng).
5. The validation round-trip (zarr layout + how to feed it back to `validate.py`).

Beam-axis sanity: the previous notebook had a beam-axis bug (used BOX_Y as thickness when beam is +Z). The current pipeline is correct (beam = Z = box_z = 73.93 Å). Any plan that proposes extending Z is wrong — re-read the supercell-extension section above.
