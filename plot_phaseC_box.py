"""Visualise the W1 Phase C tiled box so the geometry can be eyeballed before
launching the full sim. Produces a 2-panel figure (XY = beam view, XZ = depth
view) overlaying:

  - heavy atoms (Pb / Sr / Ti) coloured by species
  - the tiled cell outline (TILE_X=2, TILE_Y=2 -> 95.77 x 140.02 A)
  - the original (untiled) cell footprint (47.88 x 70.01 A) shaded
  - the internal tile-replica boundary lines (at x=47.88 and y=70.01)
  - the 5x5 scan window (20x20 A centred on params.CENTER_X_A/Y_A)
  - per-tile sub-grid inside the scan window (4 A spacing)
  - the worst-case exit-surface probe disk at the scan-corner, radius
    alpha*(Delta_f + t) -- this is the actual clearance number from
    derive_scan.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

import params as P


def main() -> Path:
    p = P.derive()
    atoms = P.load_atoms()

    bx, by, bz = atoms.cell.lengths()
    syms = np.array(atoms.get_chemical_symbols())
    pos = atoms.positions

    # Natural POSCAR footprint (before STO X-padding + TILE_Y replication).
    # Cell X composition: [STO pad | POSCAR (47.88 Å) | STO pad]; cell Y = TILE_Y · 70.01.
    poscar_bx = 47.88
    poscar_by = 70.01
    a_pto = poscar_bx / 12.0
    pad_left_end = P.STO_X_PAD_UC * a_pto
    poscar_x_start = pad_left_end
    poscar_x_end = poscar_x_start + poscar_bx
    og_bx = poscar_bx
    og_by = poscar_by

    half_scan = (p.n_tiles_tiled * p.tile_size_a) / 2.0
    sx0, sx1 = p.center_x_a - half_scan, p.center_x_a + half_scan
    sy0, sy1 = p.center_y_a - half_scan, p.center_y_a + half_scan

    alpha = p.convergence_mrad * 1e-3
    df = p.overfocus_a
    t = p.real_space_thickness_a
    r_out = alpha * (df + t)         # exit-surface probe radius (worst case)
    reach = half_scan + r_out         # from scan centre to probe edge

    species_colour = {"Pb": "#1f77b4", "Sr": "#d62728", "Ti": "#2ca02c"}

    fig = plt.figure(figsize=(15, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.65])

    # ============================================================
    # Panel 1: XY view (beam = +Z, looking down)
    # ============================================================
    ax = fig.add_subplot(gs[0, 0])

    # Shade the STO X-pads (left + right) and the embedded POSCAR.
    if P.STO_X_PAD_UC > 0:
        ax.add_patch(Rectangle((0, 0), pad_left_end, by, facecolor="#c8e6f8",
                               edgecolor="none", alpha=0.5, zorder=0,
                               label=f"STO X-pad ({P.STO_X_PAD_UC} UC each side)"))
        ax.add_patch(Rectangle((poscar_x_end, 0), bx - poscar_x_end, by,
                               facecolor="#c8e6f8", edgecolor="none",
                               alpha=0.5, zorder=0))
    # POSCAR footprint at x=[pad_left_end, pad_left_end+47.88], full Y after Y-tile.
    for m in range(P.TILE_Y):
        ax.add_patch(Rectangle((poscar_x_start, m * poscar_by), poscar_bx, poscar_by,
                               facecolor="#ffe9b3", edgecolor="none",
                               alpha=0.55, zorder=0,
                               label=f"POSCAR copy {og_bx:.1f}x{og_by:.1f} A" if m == 0 else None))

    # Vortex positions: the POSCAR has two cores per GroundTruth.ipynb,
    # lower at ~(23, 22) and upper at ~(28, 50). Phase C scans the LOWER one.
    # After STO X-pad both shift right by pad_left_end; TILE_Y replicates them
    # at y + m·70.01 for m = 0..TILE_Y-1. Scan goes on the upper Y-replica of
    # the lower vortex because the y=22 m=0 copy hard-against-y=0 aliases.
    lo_vx, lo_vy = 23.0, 22.0   # lower vortex (scan target)
    hi_vx, hi_vy = 28.0, 50.0   # upper vortex (other one in the POSCAR)
    lo_vx_pad = lo_vx + pad_left_end
    hi_vx_pad = hi_vx + pad_left_end
    lo_Y = lo_vy + np.arange(P.TILE_Y) * og_by
    hi_Y = hi_vy + np.arange(P.TILE_Y) * og_by
    # Plot both vortices' replicas so the user can see both.
    ax.scatter([lo_vx_pad] * P.TILE_Y, lo_Y, marker="*", s=220,
               facecolor="yellow", edgecolor="black", linewidth=1.4,
               zorder=6, label=f"lower vortex (23,22) [scan target]")
    ax.scatter([hi_vx_pad] * P.TILE_Y, hi_Y, marker="*", s=140,
               facecolor="#ffd070", edgecolor="0.4", linewidth=1.0,
               zorder=6, label=f"upper vortex (28,50)")
    # Annotate the lower vortex replicas
    for m, yv in enumerate(lo_Y):
        tag = f"lo m={m}"
        if abs(yv - p.center_y_a) < 1.0:
            tag += "  scan here"
        ax.annotate(tag, (lo_vx_pad, yv), xytext=(8, -10),
                    textcoords="offset points", fontsize=7, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                              ec="0.4", alpha=0.85), zorder=7)

    # Atoms (Pb/Sr/Ti only -- O is 10x more numerous and hides the structure).
    for sym in ("Ti", "Sr", "Pb"):  # Pb drawn last so it sits on top
        m = syms == sym
        ax.scatter(pos[m, 0], pos[m, 1], s=2.0, c=species_colour[sym],
                   alpha=0.55, edgecolors="none", label=f"{sym} ({m.sum()})")

    # Internal boundary lines: STO-pad / POSCAR interfaces in X, Y-tile in Y.
    for xline in (pad_left_end, poscar_x_end):
        ax.axvline(xline, color="0.4", linewidth=0.8, linestyle=":",
                   zorder=2)
    for yline in np.arange(poscar_by, by, poscar_by):
        ax.axhline(yline, color="0.4", linewidth=0.8, linestyle=":",
                   zorder=2)

    # Tiled cell outline.
    ax.add_patch(Rectangle((0, 0), bx, by, facecolor="none",
                           edgecolor="black", linewidth=2.0, zorder=3,
                           label=f"tiled cell {bx:.1f}x{by:.1f} A"))

    # Scan window (filled translucent).
    ax.add_patch(Rectangle((sx0, sy0), sx1 - sx0, sy1 - sy0,
                           facecolor="orange", edgecolor="darkorange",
                           linewidth=1.5, alpha=0.25, zorder=4,
                           label=f"scan window {sx1-sx0:.0f}x{sy1-sy0:.0f} A"))

    # Per-tile sub-grid inside the scan window.
    for xline in np.arange(sx0, sx1 + 1e-3, p.tile_size_a):
        ax.plot([xline, xline], [sy0, sy1], color="darkorange",
                linewidth=0.4, alpha=0.6, zorder=4)
    for yline in np.arange(sy0, sy1 + 1e-3, p.tile_size_a):
        ax.plot([sx0, sx1], [yline, yline], color="darkorange",
                linewidth=0.4, alpha=0.6, zorder=4)

    # Worst-case probe disk at the +X/+Y scan corner (i.e. tile (4,4)
    # corner = (sx1, sy1)). This is the disk the clearance check guards.
    ax.add_patch(Circle((sx1, sy1), r_out, facecolor="none",
                        edgecolor="purple", linewidth=1.6,
                        linestyle="--", zorder=5,
                        label=f"probe disk @exit (r={r_out:.1f} A)"))
    # Centre dot for context.
    ax.plot(p.center_x_a, p.center_y_a, marker="+", color="darkorange",
            markersize=14, mew=2.0, zorder=5)

    ax.set_xlim(-5, bx + 5)
    ax.set_ylim(-5, by + 5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (A)  -- sandwich axis (PTO/STO normal)")
    ax.set_ylabel("Y (A)  -- bread axis (in-plane periodic)")
    title = (f"XY view (beam looks down +Z)\n"
             f"STO_X_PAD_UC={P.STO_X_PAD_UC}, TILE_X={P.TILE_X}, TILE_Y={P.TILE_Y}; "
             f"overfocus={p.overfocus_a/10:.0f} nm; reach@exit-corner={reach:.1f} A; "
             f"min X-margin {min(p.center_x_a, bx-p.center_x_a)-reach:+.1f} A")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.grid(True, linestyle=":", alpha=0.3)

    # ============================================================
    # Panel 2: XZ view (X sandwich axis vs Z beam axis)
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])

    for sym in ("Ti", "Sr", "Pb"):
        m = syms == sym
        ax2.scatter(pos[m, 0], pos[m, 2], s=2.0, c=species_colour[sym],
                    alpha=0.55, edgecolors="none")

    # Tiled cell outline (Z extent will include the 2 A vacuum padding from
    # atoms.center(axis=2, vacuum=2.0) so we use the actual cell length).
    ax2.add_patch(Rectangle((0, 0), bx, bz, facecolor="none",
                            edgecolor="black", linewidth=2.0))
    # Sample slab (atoms span; bounds taken from actual atom z-extrema).
    z_atoms_lo = pos[:, 2].min()
    z_atoms_hi = pos[:, 2].max()
    ax2.axhline(z_atoms_lo, color="grey", linestyle=":", linewidth=0.8)
    ax2.axhline(z_atoms_hi, color="grey", linestyle=":", linewidth=0.8)
    ax2.text(bx + 0.5, (z_atoms_lo + z_atoms_hi) / 2,
             f"sample t={z_atoms_hi-z_atoms_lo:.1f} A", rotation=90,
             va="center", fontsize=8, color="grey")

    # Probe envelope drawn from the crossover (df above entrance) downward.
    # At depth z (measured from entrance), probe radius = alpha*(df + z).
    # The crossover sits at z = z_atoms_lo - df (above the sample).
    z_crossover = z_atoms_lo - df
    z_plot = np.array([z_crossover, z_atoms_lo, z_atoms_hi])
    r_plot = np.array([0.0, alpha * df, alpha * (df + (z_atoms_hi - z_atoms_lo))])
    # Mark crossover.
    ax2.plot(p.center_x_a, z_crossover, "kx", markersize=10, mew=1.5)
    # Probe cone (left + right edges) flaring from crossover downstream.
    ax2.plot([p.center_x_a - r for r in r_plot], z_plot,
             color="purple", linewidth=1.4)
    ax2.plot([p.center_x_a + r for r in r_plot], z_plot,
             color="purple", linewidth=1.4)
    ax2.text(p.center_x_a + 1.0, z_crossover + 1.0,
             f"crossover\n(df={df/10:.0f} nm above entrance)",
             fontsize=8, color="purple")

    ax2.set_xlim(-5, bx + 5)
    z_lo_plot = min(0, z_crossover - 10)
    ax2.set_ylim(z_lo_plot, bz + 5)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("X (A)")
    ax2.set_ylabel("Z (A) -- beam direction (+ is downstream)")
    ax2.set_title("XZ view: probe geometry through sample depth")
    ax2.grid(True, linestyle=":", alpha=0.3)

    out_path = Path(P.PROJECT_ROOT) / "phaseC_box_visualization.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")
    print(f"  tiled cell:    {bx:.2f} x {by:.2f} x {bz:.2f} A  ({len(atoms)} atoms)")
    print(f"  original cell: {og_bx:.2f} x {og_by:.2f} A")
    print(f"  scan window:   [{sx0:.1f}, {sx1:.1f}] x [{sy0:.1f}, {sy1:.1f}] A")
    print(f"  scan centre:   ({p.center_x_a:.1f}, {p.center_y_a:.1f}) A")
    print(f"  probe exit-radius: {r_out:.2f} A (alpha={alpha*1e3:.0f} mrad, df+t={df+t:.1f} A)")
    print(f"  reach@corner:  {reach:.2f} A  vs Y-half {by/2:.1f} A -> margin {by/2-reach:+.2f} A")
    return out_path


if __name__ == "__main__":
    main()
