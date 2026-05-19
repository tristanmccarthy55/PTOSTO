# Run with: C:\Users\Trist\HyperSpy-bundle\python.exe import_foldslice_result.py
"""Round-trip a fold_slice reconstruction back into the project's zarr layout
so `validate.py --recon ptycho_recon_foldslice.zarr` works unchanged.

fold_slice writes its object as a .mat with the complex array in MATLAB's
column-major (Ny, Nx, Nz) convention. We transpose to (z, y, x) complex64
and write a zarr with `object_complex` + `object_phase` datasets, mirroring
what `reconstruct_ptycho._save_zarr` produces.

Usage::

    python import_foldslice_result.py --mat foldslice_run/Niter300_object.mat \\
                                      --out ptycho_recon_foldslice.zarr
    python validate.py --recon ptycho_recon_foldslice.zarr
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import params as P


def _load_object_from_mat(mat_path: Path) -> tuple[np.ndarray, dict]:
    """Read the complex object from a fold_slice .mat. Returns (obj, meta).

    Tries `object` first, then a few common variant names. Drops any leading
    singleton axes MATLAB sometimes adds.
    """
    import scipy.io

    m = scipy.io.loadmat(str(mat_path))
    # Drop MATLAB's metadata keys ("__header__", etc.).
    keys = [k for k in m.keys() if not k.startswith("__")]

    for cand in ("object", "obj", "O", "rec_object"):
        if cand in m:
            obj = m[cand]
            break
    else:
        raise KeyError(
            f"No object array in {mat_path}. Top-level keys: {keys}. "
            "Update _load_object_from_mat() with the correct field name."
        )

    if np.iscomplexobj(obj) is False:
        print(f"[warn] {cand} is real-valued; treating as phase, no amplitude")

    # Squeeze out singleton dims (fold_slice uses (Ny, Nx, Nshare, Nz) where
    # Nshare=1 for non-shared scans).
    obj = np.squeeze(obj)
    if obj.ndim != 3:
        raise ValueError(f"Expected 3D object after squeeze, got shape {obj.shape}")

    meta: dict = {k: m[k] for k in keys if k != cand}
    return obj, meta


def import_result(mat_path: Path, out_zarr: Path,
                  *, mat_axis_order: str = "yxz") -> Path:
    """Convert a fold_slice .mat to a project-compatible zarr.

    mat_axis_order: which two MATLAB axes are in-plane and which is depth.
        "yxz" (default): obj[y, x, z] — most common fold_slice convention.
        "zyx": obj[z, y, x] — no transpose needed.
        Verify against the actual file by inspecting shape: depth axis should
        be ~num_slices (e.g. 74), in-plane axes much larger.
    """
    import zarr
    import numcodecs

    p = P.derive()

    obj_mat, mat_meta = _load_object_from_mat(mat_path)
    print(f"Loaded {mat_path.name}: shape={obj_mat.shape}, dtype={obj_mat.dtype}")

    if mat_axis_order == "yxz":
        obj = np.transpose(obj_mat, (2, 0, 1))  # (Ny, Nx, Nz) -> (z, y, x)
    elif mat_axis_order == "zyx":
        obj = obj_mat
    else:
        raise ValueError(f"unknown mat_axis_order {mat_axis_order!r}")
    obj = obj.astype("complex64")
    Nz, Ny, Nx = obj.shape
    print(f"After transpose to (z,y,x): {obj.shape}")

    if Nz != p.recon_num_slices:
        print(f"[warn] depth axis = {Nz} but params.recon_num_slices = "
              f"{p.recon_num_slices}; check mat_axis_order")

    compressor = numcodecs.Blosc(cname="zstd", clevel=3,
                                 shuffle=numcodecs.Blosc.BITSHUFFLE)
    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(out_zarr), mode="w")
    root.create_dataset("object_complex", data=obj,
                        chunks=(1, Ny, Nx), compressor=compressor)
    root.create_dataset("object_phase",
                        data=np.angle(obj).astype("float32"),
                        chunks=(1, Ny, Nx), compressor=compressor)

    # Pull a recon pixel size if fold_slice saved one; else fall back to params.
    dx_object = None
    for k in ("dx_object", "dx", "pixel_size"):
        if k in mat_meta:
            v = np.asarray(mat_meta[k]).ravel()
            if v.size:
                dx_object = float(v[0])
                break

    metadata = {
        "num_slices": int(p.recon_num_slices),
        "slice_thickness_a": float(p.recon_slice_thickness_a),
        "real_space_thickness_a": float(p.real_space_thickness_a),
        "scan_step_a": float(p.scan_step_a),
        "reconstruction_pixel_size_a": dx_object if dx_object is not None else 0.0,
        "bf_bin": int(p.bf_bin),
        "bf_eff_pixel_mrad": float(p.bf_eff_pixel_mrad),
        "bf_eff_pixel_mrad_y": float(p.bf_eff_pixel_mrad_y),
        "overfocus_a": float(p.overfocus_a),
        "polar_C10": float(+p.overfocus_a),
        "energy_ev": float(p.energy_ev),
        "convergence_mrad": float(p.convergence_mrad),
        "engine": "fold_slice_LSQML",
        "source_mat": str(mat_path),
        "mat_axis_order": mat_axis_order,
    }
    root.attrs.update(metadata)

    sidecar = out_zarr.parent / (out_zarr.stem + "_metadata.json")
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nWrote {out_zarr}  (object_complex {obj.shape}, dtype {obj.dtype})")
    print(f"Sidecar metadata: {sidecar}")
    print(f"\nNext: python validate.py --recon {out_zarr}")
    return out_zarr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mat", type=Path, required=True,
                    help="Path to fold_slice .mat output (e.g. Niter300_object.mat)")
    ap.add_argument("--out", type=Path,
                    default=P.PROJECT_ROOT / "ptycho_recon_foldslice.zarr",
                    help="Output zarr path")
    ap.add_argument("--mat-axis-order", choices=("yxz", "zyx"), default="yxz",
                    help="MATLAB object axis order (default: yxz)")
    args = ap.parse_args()
    import_result(args.mat, args.out, mat_axis_order=args.mat_axis_order)


if __name__ == "__main__":
    main()
