# Handover — W1 Phase C: cell extension + higher overfocus re-sim

The fold_slice / PtychoShelves port (W1) is **working end-to-end** on the
existing 5 nm-overfocus 6400-position cryo data, but the reconstruction is
**under-constrained**: the dominant headwind is overlap diversity, not
engine choice. Yu Lei's group runs ~20 nm overfocus precisely because the
wide probe footprint creates many redundant constraints per object pixel.
We can't push above ~5 nm overfocus on the current data because the probe
wraps around the periodic abTEM cell (47.88 Å short axis). This handover
is to **re-simulate with the cell tiled** and the overfocus pushed to ~20
nm, then re-run the existing fold_slice recon on the new data.

This was **Phase C of the original W1 plan** ([`W1_PROMPT.md`](W1_PROMPT.md)
§ "Phase C — abTEM cell extension for bigger overfocus") — deferred so we
could prove the engine port worked first. We've now proven it; § 12 of
[`LABBOOK.md`](LABBOOK.md) is the full narrative of that proof and lays out
exactly why this Phase C is the next move.

## Read in order

1. **`LABBOOK.md` § 12** — what's been done in W1 fold_slice port, the
   L-curve sweep findings, and the explicit hypothesis at the end of § 12.4
   that "**residual symmetric pattern is the kind of artefact that tilts
   (W4) are designed to break**" — Phase C tries to break it first via
   better data conditioning before resorting to tilts.
2. **`W1_PROMPT.md` § "Phase C"** — the original spec for cell extension,
   including which axis is sandwich vs bread and the exact one-line ASE op
   needed.
3. **`HANDOVER.md`** — strategic context, paper reference (Lei & Wang
   PSO_science), and the ranked W1/W2/W3/W4 ladder this Phase C sits on.
4. **`params.py`** — `TILE_X`, `TILE_Y` knobs already exist (defaults =
   1, 1). `load_atoms()` already applies `atoms * (TILE_X, TILE_Y, 1)`
   between `orthogonalize_cell` and `center(axis=2)`. Just needs values
   bumped.
5. **`derive_scan.py`** — the probe-vs-cell clearance check that decides
   the safe overfocus given the tiled cell.
6. **`simulate_4dstem.py`** — tile sim entry point (no edits expected;
   reads `params.load_atoms()`).
7. **`export_to_foldslice.py`** + **`recon.m`** + **`import_foldslice_result.py`**
   + **`validate.py`** — the round-trip pipeline; works as-is, just needs
   to be re-run on the new data.

## Why this is the right next move (one paragraph)

Current best W1 result: kz-FRC ratio = **0.087** (PASS gate ≥ 0.05),
std-ratio 1.76 — the un-regularised 200-iter production
(`ptycho_recon_foldslice.zarr`). py4DSTEM on the *identical data* gave
kz-FRC 6.8e-6 — so the engine port did its job (~13 000× better), but
absolute kz-FRC of 0.087 is not where we need it for clean atomic-plane
depth resolution. Three L-curve sweeps (LABBOOK § 12.5) confirmed that
further recon-side tuning has hit a wall: `reg_mu = 1e-3` was a no-op,
`regularize_layers > 0` introduces a clean knee but suppresses kz-FRC
(0.087 → 0.047). The current `recon.m` defaults (`reg_mu=1e-3`,
`regularize_layers=0`) are equivalent to plain no-reg.
The diagnosis: the recon is **under-determined** at our settings. Counting:
6400 patterns × 65×65 px ≈ 27 M constraints vs 37 slices × 468×468 ≈ 8 M
object pixels per slice → ~10:1 under-determined per slice. Wider probe
(20 nm overfocus instead of 5 nm) sees ~16× the area per scan position →
each object pixel constrained by ~16× more positions → much better
condition number. This is what the paper has and we don't.

## Phase 0 — verify which axis is sandwich (BEFORE editing params)

[`W1_PROMPT.md`](W1_PROMPT.md) § Phase C predicted X = sandwich (12
PTO+STO layers ≈ 48 Å) and Y = bread (18 perovskite UCs ≈ 70 Å), but
flagged that it needs verification. Run this probe first, **before** setting
TILE_X / TILE_Y:

```python
# from project root
from params import load_atoms
import numpy as np
a = load_atoms()
print("cell:", a.cell.lengths(), "atoms:", len(a))
syms = np.array(a.get_chemical_symbols())
for sym in ("Pb", "Sr", "Ti"):
    xs = a.positions[syms == sym, 0]
    ys = a.positions[syms == sym, 1]
    print(f"  {sym}: x {xs.min():.1f}..{xs.max():.1f}   y {ys.min():.1f}..{ys.max():.1f}")
```

If Pb / Sr occupy **disjoint x-ranges** (e.g. Pb at x=0..24, Sr at x=24..48),
X is the sandwich (PTO/STO stacking normal) and `TILE_X = 2` is the right
extension. If they occupy disjoint y-ranges instead, swap the multiplier
to `TILE_Y = 2`.

## Phase 1 — params.py knob updates

Open [`params.py`](params.py). Edits required (with rationale per edit):

| Knob | From | To | Why |
|---|---:|---:|---|
| `TILE_X` | 1 | **2** (or `TILE_Y=2` if Phase 0 swapped) | doubles sandwich axis 48 → 96 Å so probe fits at 20 nm overfocus |
| `OVERFOCUS_NM` | 5.0 | **20.0** | matches Lei & Wang paper recipe; wide probe = more overlap diversity |
| `CENTER_X_A` | 23.0 | **48.0** (= bx/2 of tiled cell ≈ 95.8/2) | re-centre scan on the new cell |
| `TARGET_OVERLAP` | 0.95 | **0.98** | denser scan → more constraints; at 20 Å probe FWHM gives ~0.4 Å step |

Do **not** change anything else without reading the surrounding comments
first — `N_TILES_TILED`, `TILE_SIZE_A`, `PHONON_NUM_CONFIGS`,
`PHONON_DWF_SCALE_DEFAULT` are dialled for current physics and shouldn't
move unless you understand why.

## Phase 2 — derive_scan.py clearance verification

Run [`derive_scan.py`](derive_scan.py). The clearance table should show
**"OK — fits"** for 20 nm overfocus on the new tiled cell:

```
python derive_scan.py
```

Look at the "Probe-vs-cell clearance" section near the bottom. With
`TILE_X=2`, the cell is ~96 × 70 × 74 Å. At 20 nm overfocus and our
current 5×5 × 4 Å = 20 Å scan window, the worst-case probe reach
(half_scan + α·(Δf+t)) should sit comfortably inside the tiled box.

If `derive_scan.py` reports overhang > 0 even at TILE_X=2, bump to
TILE_X=3 (cell becomes 144 × 70 × 74 Å — sim cost goes up ~1.5× per
tile but it's still survivable on the 8 GB GPU).

Also check the predicted scan position count: at 20 nm overfocus, 0.98
overlap, FWHM ≈ 20 Å, step ≈ 0.4 Å → about 50×50 = 2500 positions per
tile, 12500 over 5×5 tiles. Disk budget: ~0.3 GB / tile compressed, ~8 GB
total. Manageable.

## Phase 3 — re-simulate the 25 tiles

Old tile zarrs (`tile00_bf.zarr` through `tile44_bf.zarr`) need to be
moved out of the way OR the new tiles need a different naming scheme.
**Recommendation: move the old ones to an archive subdir** so the
original W1 result (kz-FRC 0.087) remains reproducible:

```bash
mkdir -p archive_w1_5nm_overfocus
mv tile??_bf.zarr archive_w1_5nm_overfocus/
mv tile??_haadf.zarr archive_w1_5nm_overfocus/
```

Then run the sim. Expected cost: each tile at the 2× cell needs ~2× FFT
grid each axis → 4× larger 2D FFTs → ~3-4× slower per tile vs the old
sim. Old was ~15 min/tile = ~6h total; new will be ~20-30 min/tile =
~10-13h total. Strongly suggest **letting it run overnight**.

```bash
python simulate_4dstem.py            # runs all 25 tiles sequentially
```

If you want to run a single tile first as a smoke test:

```bash
python simulate_4dstem.py --tile 2 2   # centre tile only, ~25 min
```

Check the resulting `tile22_bf.zarr` has shape `(Sy, Sx, Dy, Dx)` with
`Sy = Sx = round(TILE_SIZE_A / scan_step_a) ≈ round(4 / 0.4) ≈ 10` per
axis. CBED shape `(Dy, Dx)` should match params: with TILE_X=2 the
cell aspect changes (96 × 70), so new Dy ≈ ceil(2 × 200 / pix_mrad_x_new)
where `pix_mrad_x_new = wavelength * 1000 / 96 ≈ 0.21 mrad/px`
(K-binned with the larger n_native gives a different bf_bin — let
params.derive() compute it; don't second-guess).

## Phase 4 — re-export to fold_slice HDF5

The exporter is data-agnostic; same command works on the new tiles:

```bash
python export_to_foldslice.py --pipeline --out "1/data_roi0_Ndp65_dp.hdf5"
```

(The output filename keeps `_Ndp65_` for legacy; rename to reflect new CBED
size if you want — but [`recon.m`](recon.m) reads from this exact path,
so easier to just keep the name and update the `Ndpx` literal in `recon.m`
if the CBED size changes.)

Likely change: with the bigger cell, the natural CBED size after the
square-resample may shift from 65 → some other value. Update `Ndpx`
and `rbf` in `recon.m` accordingly:

- `Ndpx` = new CBED size (read from the HDF5 `/dp` dataset shape)
- `rbf` = `alpha0 / d_alpha_mrad` where `d_alpha_mrad` comes from the
  `pixel_mrad` attr in the HDF5 root

The export prints all this; just copy from the run log.

## Phase 5 — re-run the fold_slice recon

[`recon.m`](recon.m) is at: `reg_mu=1e-3` (no-op per sweep #3, harmless),
`regularize_layers=0` (anything > 0 suppresses kz-FRC), Stage A = 40 iter,
Stage B = 30 iter (set as `eng.number_iterations` in the prod branch — was
80 during the L-curve sweep, switch back to 30 if you re-run prod
directly), probe + position refinement both off (`probe_change_start = inf`,
`probe_position_search = inf` — sim-recipe per Yu). On Phase C's
better-conditioned data the iter optimum may move — recommend redoing the
L-curve before locking the production recipe. Open MATLAB:

```
cd c:\Users\Trist\fold_slice\ptycho
matlab -batch "addpath('c:\Users\Trist\HyperSpy-bundle\MY HYPERSPY CODE\PTOSTO'); recon"
```

Stage A + Stage B at the new (larger) CBED size and more positions: ~3-5h
total. Watch for OOM on the 8 GB 2070 SUPER — if `eng.grouping = 32`
crashes with VRAM exceeded, drop to 16 or 8.

If you want to redo the L-curve sweep with the new data (recommended —
the optimum iter count may shift with the better-conditioned problem), bump
Stage B `number_iterations = 80` and `save_results_every = 10` in
[`recon.m`](recon.m), then run [`lcurve.py`](lcurve.py) on the resulting
TIFFs. ~91 min.

## Phase 6 — round-trip + validate + compare

```
python import_foldslice_result.py --mat "<path_to_final_recons.mat>" --out ptycho_recon_phaseC.zarr
python validate.py --recon ptycho_recon_phaseC.zarr
```

**Success criterion**: kz-FRC ratio > **0.087** (current W1 best), ideally
above 0.2. std-ratio > 2.5. Visually: atomic columns clear in the XY slice
montage AND the XZ-deviation panel should show column-localised structure
*without* the residual anti-symmetric envelope that the current W1 result
still has.

If kz-FRC < 0.087 despite the better data, the bottleneck is the **engine**
(not data conditioning) and the next move is **W4 — tilt series** (see
[`HANDOVER.md`](HANDOVER.md) § "W4"). If kz-FRC > 0.2 with clean atomic
columns, you have a paper-quality result.

## Update LABBOOK on completion

Append a § 13 to [`LABBOOK.md`](LABBOOK.md) with:

- The verified axis identity (sandwich = X or Y?) and TILE_(X|Y) value used
- Cell dims after tiling, sim cost, total recon cost
- New kz-FRC and std-ratio vs the W1 numbers (table)
- XZ-deviation comparison vs the W1 panel (link to the two
  `_summary.png` plots side-by-side)
- Verdict: does the cell extension + 20 nm overfocus break through to
  atomic-plane depth resolution, or does W4 tilts become the next move?

## Files you should NOT touch (unless you really mean to)

- `tile*_bf.zarr` / `tile*_haadf.zarr` in the project root — these are
  the **original W1 data**. Move to `archive_w1_5nm_overfocus/` before
  the new sim, don't overwrite. The kz-FRC 0.087 W1 result needs to stay
  reproducible from these.
- `analysis/S00000-00999/S00001/*_recons.mat` — the W1 .mat outputs.
  fold_slice may overwrite with new runs; pre-copy to
  `archive_w1_5nm_overfocus/` if you want them preserved.
- `ptycho_recon_foldslice*.zarr` — the validated zarrs from W1. The new
  recon should go to a NEW zarr name (e.g. `ptycho_recon_phaseC.zarr`).
- Anything under `c:\Users\Trist\fold_slice\` — that's Yi Jiang's repo
  with our patches (`find_geom_correction.m` floor fix, and the MATLAB
  XML patches under `C:\Program Files\MATLAB\R2022b\bin\win64\mexopts\`
  and `...\toolbox\parallel\gpu\extern\src\mex\win64\nvcc_msvcpp2019.xml`).
  Document any new patches you add but don't unwind the existing ones —
  they're load-bearing for the MEX+CUDA chain on this Windows install.

## Environment quick-reference

- Python: `C:\Users\Trist\HyperSpy-bundle\python.exe` (has abTEM, py4DSTEM,
  scipy, h5py, zarr, numcodecs, tifffile, matplotlib).
- MATLAB: R2022b at `C:\Program Files\MATLAB\R2022b\bin\matlab.exe`.
- CUDA: MATLAB-bundled 11.2 + system 13.1 (mexcuda uses 11.2). MSVC v142
  is the only supported toolset (v143 was uninstalled — see § 12.5 of
  LABBOOK for the chain of fixes that made this stable).
- GPU: NVIDIA GeForce RTX 2070 SUPER, 8 GB VRAM. Tight at 4 probe modes ×
  37 slices × 6400 positions × `grouping=32`; new data with more positions
  may need `grouping=16`.

Good luck. Make sure to actually verify Phase 0's axis identity before
committing to a ~12 hour resim.
