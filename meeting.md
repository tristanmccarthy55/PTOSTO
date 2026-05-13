# Supervisor meeting notes

*PTO/STO labyrinth — 4D-STEM + multislice ptychography*

---

## Where I started

The previous attempt lived in a single notebook (`TileDataAnalysis_WORKS copy 2.ipynb`). It ran, it produced a 3D reconstruction, but the 3D volume was **blurry along the beam direction** — atom columns smeared through the whole thickness instead of changing with depth, which is the whole point. There was no easy way to test individual changes because everything was tangled together in one file.

## What I changed about how I work

I refactored the notebook into a set of small, single-purpose Python scripts:

- one file that owns every experimental parameter and derives everything else from it
- one file for the abTEM simulation
- one file for the py4DSTEM reconstruction
- one for validation / plotting
- one for sanity-check diagnostics
- one for the analytical "before I run a 4-hour sim, will this even fit?" pass

The big win is that now I can change one thing at a time and re-run cleanly — overnight, unattended, logged — and I can ablate hypotheses (turn one knob off and see what happens). I also wrote a clearance check that catches problems on paper before I commit a full GPU night to them.

## Things I fixed in the science before the new experiments

A pile of small-to-medium bugs in the notebook setup, the ones that mattered:

- The beam axis was wrong — the notebook used the wrong cell dimension as the through-beam thickness, so the reconstruction was being asked to invert ~4 Å too much sample.
- The probe defocus sign — I verified empirically (with a vacuum-probe-through-free-space test) which sign convention abTEM uses vs. py4DSTEM, and confirmed both ends now agree that the probe is genuinely overfocused (focused above the sample).
- **Frozen phonons** — the notebook didn't use them. I enabled them (8 thermal configurations per scan) with Debye-Waller σ values from the perovskite literature, and ran them at cryo temperatures (× 0.65 of room-temp σ) because the reference paper notes that depth resolution is "sensitive to lattice vibrations" and cryo helps.
- The reconstruction was letting the object's amplitude run free, which lets depth information leak into the (meaningless) amplitude channel. Pinned the object to be pure-phase. Huge cleanup of artefacts.
- Detector calibration was an ad-hoc geometric guess; now derived from the actual angular pixel size.
- A handful of py4DSTEM rough edges — auto-rotation calibration was getting biased by anisotropic CBED resampling, the scikit-image phase unwrapper has a documented hang, the CUDA memory pool fragments between stages — all worked around or forced to known values.

## What I actually tried (after the bug fixes)

Two ideas, both informed by the reference paper (Lei & Wang, *Multislice Hollow Ptychography*):

1. **Bigger scan area + a more defocused probe.** The paper used a ~26 × 26 Å scan with a probe defocused by ~20 nm. I went from a 12 × 12 Å scan with 1 nm defocus to a **20 × 20 Å scan with 5 nm defocus** (~6400 scan positions, above the paper's ~4096). I couldn't match the paper's 20 nm because our convergence angle (100 mrad) is ~5× larger than theirs, which makes the defocused probe ~5× bigger — at 20 nm the probe would have wrapped around the periodic abTEM cell and aliased the diffraction patterns. 5 nm is the largest defocus that still fits the cell cleanly.
2. **Cryo phonons.** As above.

Then, when those didn't crack it, an ablation:

3. **Turned the depth-direction regularisation off.** py4DSTEM applies a low-pass filter and a TV-denoiser along the depth direction by default — which, the more I thought about it, sounded like the *last* thing you want when you're trying to recover depth structure. So I turned them both off and re-ran the reconstruction (no new simulation needed).

## Best result I got

The honest version: **it's still a projection.** The reconstruction puts the right *total* phase into the volume but smears it uniformly along the depth direction — exactly the same failure mode as the original notebook. The depth-direction Fourier shell correlation (the test for "is there real depth structure?") is still a delta at zero. All three things above improved the *cleanliness* of the reconstruction, but none of them gave us atomic columns localised in z.

## What I think I discovered, and why it's interesting

When I improved the visualisation — averaging the XZ depth view over 2/3 of a perovskite unit cell in the orthogonal direction, and plotting the **(slice − projection)** instead of just raw slices — a clean pattern appeared:

- The lateral resolution is great. You can see individual perovskite columns in several of the slices.
- But the columns don't *move* with depth. They sit at the same (x, y) at every z. The reconstruction is laterally a projection.
- The deviation-from-projection panel is **not flat** (which is what a true projection would give). It shows a coherent, column-aligned pattern with a clean **sign flip across mid-thickness**: deviation positive at column centres in the upper half, negative at the same column centres in the lower half.

What that's telling me: the data **do contain depth information** — otherwise the deviation would be zero, not anti-symmetric. The reconstruction algorithm is "aware" of the depth signal but doesn't know where to put it, so it dumps it into the smoothest possible non-projection answer (a single sign change across the slab). It's not real layer-by-layer structure; it's the algorithm's response to having information it can't localise.

This is genuinely useful to know because it tells us the bottleneck is the *algorithm*, not the *experiment*. Every depth-sectioning paper I can find uses a different reconstruction code — **PtychoShelves** (and its electron-ptychography fork, **fold_slice**) with a more powerful optimiser called LSQ-ML, plus probe-position refinement and momentum acceleration. None of those features exist in py4DSTEM's multislice ptychography. The smooth anti-symmetric mode we're seeing is exactly what those features are designed to break.

## What I'd like to do next

Port the reconstruction step to **fold_slice / PtychoShelves**, keeping the abTEM simulation data exactly as it is (the 25 tile files I have on disk are reusable — only the reconstruction step changes). It's MATLAB rather than Python, which is a small overhead, but it's the code the reference paper actually used. Best guess: 1–2 days of work to get a first reconstruction out, after which we'll know whether the depth signal we're seeing in the deviation panel collapses into real atomic-layer structure with a better optimiser.

If after that we still can't get depth structure from a single perpendicular scan, the literature answer for thick strong scatterers (Dong *et al.* 2025) is **multiple tilts** — but the multi-tilt reconstruction needs fold_slice anyway, so that's a natural follow-on, not a parallel option.

## Things I'd like to discuss

1. **What depth resolution we actually need.** The polarisation rotates continuously over the ~7 nm tornado. Atomic-layer (~2 Å) is the paper-class target, but ~1 nm would already split the rotation into ~7 angular bins — possibly enough for the science we want to tell. Worth deciding before I sink time into tilt-coupling.
2. **MATLAB / PtychoShelves access** — I think this is fine via the campus network but want to confirm before I start the port.
3. **Whether the negative result we have now is worth writing up regardless.** It's a clean, well-conditioned dataset with a controlled set of ablations that pin down where the failure mode lives. Even if W1 succeeds, the "py4DSTEM MSPT can't depth-section a 74 Å labyrinth no matter what you do to the data" finding might be useful to the community.
4. **Time budget for the port.** I'd want to give it ~1 week of focused effort before deciding whether the single-perpendicular-scan story works or we need tilts.

---

*See `LABBOOK.md` in the same folder for the technical detail, run logs, and specific parameter choices — this is the short version for our meeting.*
