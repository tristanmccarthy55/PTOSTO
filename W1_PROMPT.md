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

**Important sequencing note:** the very first deliverable of this plan must be **Phase 0 (below)** — a concrete question list / short email draft for Yu Lei. He's down the hall and Phases A–E all depend on inputs only he can give cheaply (recommended fold_slice branch, MATLAB version, working param struct for a similar perovskite system). Get that human-element ask out the door *before* working through the technical phases, so his reply is already in the inbox by the time the plan needs it. Do not bury this at the end.

## Phase 0 — what to ask Yu Lei (FIRST deliverable, today)

Produce a 5-minute email draft (or talking-points if I'd rather walk to his desk) — tight, scannable, prioritised. The plan should write this as a separate small markdown file (`yu_lei_questions.md`) so I can edit lightly and send. Refine and reorder the seed list below; don't just paste it — the plan agent should think about what's actually unknowable from the paper alone.

**Seed question list (refine, don't paste verbatim):**

*Install / build (5 min answer):*
- Which fold_slice fork / branch does your group actually run? Yi Jiang's `main`, or an internal copy with patches?
- MATLAB version that works (R2021a-ish? newer?).
- Do you run on the campus Linux cluster or a local Windows / Linux box? Is there a working install I could borrow access to before I build my own?

*Recon params for a similar system (the real ask):*
- For a thick (74 Å) PTO6/STO6 labyrinth at 300 keV, 100 mrad convergence, 5 nm overfocus, cryo phonons, 6400 scan positions over 20 × 20 Å, what's your starting engine struct? Specifically: LSQ-ML variant, Stage A / Stage B iteration counts, position-refinement step size & schedule, momentum β, mixed-state probe modes, gaussian / TV regularisation.
- Detector-rotation refinement — free or fixed? (Our abTEM data has known-zero detector rotation by construction.)
- Per-axis Q-pixel calibration in fold_slice — is the HDF5 struct OK with non-square CBED (ours is natively 65 × 95 px at ~6.17 mrad/px vs ~4.22 mrad/px), or do you pre-rectify?

*Science / sanity (gut-check the whole approach):*
- Our py4DSTEM single-perpendicular-scan recon collapses to a z-uniform projection plus an **anti-symmetric depth-mode** (deviation positive in z ≈ 20–35, negative in z ≈ 45–65, at the same column positions). Is that a known py4DSTEM-MSPT signature, and does LSQ-ML routinely break it on similar-thickness samples?
- At our 100 mrad convergence (vs your paper's 21 mrad) — does single-perpendicular-scan have any realistic shot at atomic-layer depth in a 74 Å sample, or should I plan tilts (W4) from the start?
- Cryo σ scaling — your paper notes depth res is "sensitive to lattice vibrations". What σ values did you use, and from what source?

*Practical:*
- Anything specific to ferroelectric / heavy-Pb columns (dynamical scattering) that bit you and isn't in the paper?

Email draft should be ≤ 200 words and lead with the two highest-leverage asks (working param struct + single-scan-vs-tilts gut check). Everything else is a nice-to-have he can answer in a sentence each.

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

Write the plan to your plan file. Do not implement yet. Output in this order:

0. **First**: `yu_lei_questions.md` — short email / talking-points draft for Yu Lei. See Phase 0. This is the only artefact you produce *before* writing the rest of the plan, so I can send it / walk to his desk while you finish.
1. fold_slice install / build path (Windows vs cluster, MATLAB version).
2. The export-bridge file format (we want exact HDF5 dataset names + per-axis Q calibration).111111
3. The supercell-extension code change (which axis is sandwich vs bread, what `params.py` API).
4. The starting recon parameter struct (placeholders, to be filled by Yu/Peng).
5. The validation round-trip (zarr layout + how to feed it back to `validate.py`).

Beam-axis sanity: the previous notebook had a beam-axis bug (used BOX_Y as thickness when beam is +Z). The current pipeline is correct (beam = Z = box_z = 73.93 Å). Any plan that proposes extending Z is wrong — re-read the supercell-extension section above.
