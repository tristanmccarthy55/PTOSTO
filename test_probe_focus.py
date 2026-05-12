# Run with: C:\Users\Trist\HyperSpy-bundle\python.exe test_probe_focus.py
"""Where does the probe focus?

Builds an abTEM Probe with the SAME call simulate_4dstem.py uses
(``probe.aberrations.defocus = +OVERFOCUS_A``), propagates it through pure
vacuum starting from z=0 (the sample entrance plane in our geometry), and
tracks the probe width / peak intensity vs propagation distance.

Interpretation
--------------
In our sim geometry the probe enters the sample at z=0 and propagates in +Z.

* If the probe **re-focuses** (width minimum / peak maximum) at some z > 0,
  the crossover is *inside* the sample  ->  this is UNDERFOCUS.
  ->  ``defocus = +OVERFOCUS_A`` is the WRONG sign; we'd need ``-OVERFOCUS_A``.

* If the probe width grows monotonically from z=0 (no minimum in the swept
  range), the crossover already happened at z < 0, in the source-side vacuum
  *above* the sample  ->  this is OVERFOCUS (focused before entering).
  ->  ``defocus = +OVERFOCUS_A`` is CORRECT.

We sweep both signs as a cross-check.
"""
from __future__ import annotations

import numpy as np
import abtem
import ase

ENERGY_EV = 300e3
SEMIANGLE_MRAD = 100.0
OVERFOCUS_A = 10.0          # exactly what params.OVERFOCUS_NM * 10 gives
BOX_XY = (47.88, 70.01)     # match the real cell laterally
SAMPLING = 0.15             # Å -- coarse is fine; the defocused probe is ~1 Å
SLICE_TH = 1.0              # Å, vacuum slices
Z_MAX = 40.0                # Å of vacuum to propagate through
Z_STEP = 2.0                # Å

# A "vacuum" potential: an empty cell. If abTEM's Potential rejects an atom-less
# Atoms object we fall back to a single He atom parked in a far corner (its
# potential at the centred probe is utterly negligible).
def _vacuum_potential(thickness_a: float):
    cell = [BOX_XY[0], BOX_XY[1], thickness_a]
    try:
        atoms = ase.Atoms(cell=cell, pbc=True)
        return abtem.Potential(atoms, sampling=SAMPLING,
                               slice_thickness=min(SLICE_TH, thickness_a),
                               device="cpu", parametrization="kirkland")
    except Exception:
        atoms = ase.Atoms("He", positions=[[0.1, 0.1, thickness_a / 2]],
                          cell=cell, pbc=True)
        return abtem.Potential(atoms, sampling=SAMPLING,
                               slice_thickness=min(SLICE_TH, thickness_a),
                               device="cpu", parametrization="kirkland")


def _build_probe(defocus_a: float, potential):
    probe = abtem.Probe(energy=ENERGY_EV, semiangle_cutoff=SEMIANGLE_MRAD,
                        device="cpu")
    probe.aberrations.defocus = defocus_a   # <-- the exact line from the sim
    probe.grid.match(potential)
    return probe


def _intensity_array(waves) -> np.ndarray:
    """Get a 2-D real-space intensity numpy array from an abTEM Waves object."""
    obj = waves
    # Try .intensity() -> Images
    try:
        obj = waves.intensity()
    except Exception:
        pass
    arr = getattr(obj, "array", obj)
    try:
        arr = arr.compute()
    except Exception:
        pass
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        arr = np.abs(arr) ** 2
    # collapse any leading singleton ensemble dims
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def _rms_radius_a(I: np.ndarray) -> float:
    I = I / I.sum()
    yy, xx = np.indices(I.shape)
    cy = (I * yy).sum(); cx = (I * xx).sum()
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return float(np.sqrt((I * r2).sum()) * SAMPLING)


def width_profile(defocus_a: float):
    """Return (z_array, rms_radius_array_A, peak_intensity_array)."""
    zs = np.arange(Z_STEP, Z_MAX + Z_STEP / 2, Z_STEP)
    rms = np.empty_like(zs)
    peak = np.empty_like(zs)
    for k, T in enumerate(zs):
        pot = _vacuum_potential(float(T))
        probe = _build_probe(defocus_a, pot)
        exit_waves = probe.multislice(pot)
        I = _intensity_array(exit_waves)
        rms[k] = _rms_radius_a(I)
        peak[k] = float(I.max() / I.sum())   # normalised peak
    # Width at the entrance plane (z=0) for reference.
    pot0 = _vacuum_potential(SLICE_TH)
    probe0 = _build_probe(defocus_a, pot0)
    I0 = _intensity_array(probe0.build())
    rms0 = _rms_radius_a(I0)
    peak0 = float(I0.max() / I0.sum())
    return zs, rms, peak, rms0, peak0


def report(defocus_a: float):
    zs, rms, peak, rms0, peak0 = width_profile(defocus_a)
    imin = int(rms.argmin())
    ipk = int(peak.argmax())
    print(f"\n=== probe.aberrations.defocus = {defocus_a:+.1f} Å ===")
    print(f"  at entrance (z=0):  RMS radius = {rms0:.3f} Å,  peak = {peak0:.3e}")
    print(f"  profile (z[Å]: RMS[Å] | peak):")
    for z, r, p in zip(zs, rms, peak):
        mark = ""
        if z == zs[imin]: mark += "  <- min RMS"
        if z == zs[ipk]:  mark += "  <- max peak"
        print(f"    {z:5.1f}: {r:6.3f} | {p:.3e}{mark}")
    refocuses = (imin > 0) and (rms[imin] < 0.7 * rms0) and (rms[imin] < rms[-1])
    print(f"  --> {'RE-FOCUSES inside (UNDERFOCUS)' if refocuses else 'spreads monotonically (OVERFOCUS: focused before entry)'}")
    return refocuses


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    print("Probe focus test -- 300 keV, 100 mrad, sweeping defocus sign.")
    print(f"Geometry: probe enters at z=0, propagates +Z through {Z_MAX} Å vacuum.")
    print("(In the real sim the sample top surface sits ~2 Å past z=0.)")

    underfocus_pos = report(+OVERFOCUS_A)
    underfocus_neg = report(-OVERFOCUS_A)
    _ = report(0.0)

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    if (not underfocus_pos) and underfocus_neg:
        print("defocus = +OVERFOCUS_A  ->  OVERFOCUS (probe focused BEFORE the sample). CORRECT.")
        print("Our simulate_4dstem.py setting `probe.aberrations.defocus = +p.overfocus_a` is RIGHT.")
        print("Matching py4DSTEM C10 (since abtem defocus = -C10) = -OVERFOCUS_A. Current recon is consistent.")
    elif underfocus_pos and (not underfocus_neg):
        print("defocus = +OVERFOCUS_A  ->  UNDERFOCUS (probe focuses INSIDE the sample). WRONG SIGN.")
        print("Fix: simulate_4dstem.py should use `probe.aberrations.defocus = -p.overfocus_a`.")
        print("Then matching py4DSTEM C10 = +OVERFOCUS_A (which is what the notebook uses).")
        print("** The existing phononless dataset (sim'd with defocus=-OVERFOCUS_A) is therefore the CORRECT sign already. **")
    else:
        print("Ambiguous -- inspect the profiles above by hand. Neither sign showed a clean re-focus / monotonic-spread split.")


if __name__ == "__main__":
    main()
