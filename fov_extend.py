#!/usr/bin/env python3
"""
CT DICOM FOV extender (예: FOV 500 mm -> FOV 800 mm)

Bug this file fixes
-------------------
Previous version: only the PixelSpacing / ReconstructionDiameter tags were
changed to 800/512, and the pixel data stayed the 500 mm image as it was
(or it was resized to the same frame and re-saved).
-> In the RTP the same 512x512 image is read as 800 mm, so the whole body is
   enlarged 1.6x (800/500) in physical size. The image looks like "FOV500
   blown up", and outlines, distances and dose calculation are all wrong.

Correct method
--------------
1. Keep the matrix (512) and change the pixel size to 800/512 = 1.5625 mm.
2. Resample the original image into the new grid around the same physical
   centre (= reconstruction centre). The original 500 mm only fills the
   middle 62.5% (500/800) of the new frame, and the outer area is filled
   with air (-1000 HU class).
3. Recompute ImagePositionPatient (top-left pixel coordinate) for the new
   grid, so the patient coordinates (FrameOfReference) match the original
   exactly.

Note: This is not the scanner's extended-FOV reconstruction from raw data
(Siemens HD FOV / FAST eFoV). Anatomy cut off outside the original 500 mm
is not recovered; only the outside is filled with air. Only the image size
and grid are extended to 800 mm.

Usage:
    python fov_extend.py <input_dir> <output_dir> [--fov 800] [--matrix 512]
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from scipy.ndimage import map_coordinates

AIR_HU = -1024.0  # value outside the original FOV / below air


def _geometry(ds):
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]          # direction j increases, direction i increases
    ipp = np.array(ds.ImagePositionPatient, dtype=float)
    dr, dc = (float(v) for v in ds.PixelSpacing)  # [row spacing(between rows), column spacing]
    return ipp, row_dir, col_dir, dr, dc


def extend_slice(ds, fov_mm: float, matrix: int, series_uid: str):
    """Return a new Dataset with one slice extended to fov_mm / matrix."""
    rows, cols = int(ds.Rows), int(ds.Columns)
    ipp, row_dir, col_dir, dr, dc = _geometry(ds)

    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    hu = ds.pixel_array.astype(np.float32) * slope + intercept
    # Outside the recon circle (-2048, -3024 HU etc.) / padding -> air
    hu = np.maximum(hu, AIR_HU)

    # Physical centre of the original image (= reconstruction centre)
    center = ipp + row_dir * dc * (cols - 1) / 2.0 + col_dir * dr * (rows - 1) / 2.0

    new_sp = fov_mm / matrix
    half = (matrix - 1) / 2.0
    new_ipp = center - row_dir * new_sp * half - col_dir * new_sp * half

    # New pixel (i, j) -> fractional index in the original image
    k = (np.arange(matrix) - half) * new_sp
    src_i = k / dr + (rows - 1) / 2.0
    src_j = k / dc + (cols - 1) / 2.0
    ii, jj = np.meshgrid(src_i, src_j, indexing="ij")
    out_hu = map_coordinates(hu, [ii, jj], order=1, mode="constant", cval=AIR_HU)

    # HU -> stored value (keep original slope/intercept and data type)
    dtype = ds.pixel_array.dtype
    info = np.iinfo(dtype)
    stored = np.clip(np.rint((out_hu - intercept) / slope), info.min, info.max).astype(dtype)

    new = copy.deepcopy(ds)
    if new.file_meta.TransferSyntaxUID.is_compressed:
        new.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    new.Rows = matrix
    new.Columns = matrix
    new.PixelSpacing = [f"{new_sp:.6f}", f"{new_sp:.6f}"]
    new.ImagePositionPatient = [f"{v:.4f}" for v in new_ipp]
    new.ReconstructionDiameter = f"{fov_mm:g}"
    new.PixelData = stored.tobytes()
    for tag in ("SmallestImagePixelValue", "LargestImagePixelValue",
                "ReconstructionTargetCenterPatient"):
        if tag in new:
            del new[tag]
    # Fill the outside with the same value as the padding value
    if "PixelPaddingValue" in new:
        new.PixelPaddingValue = int(np.clip(np.rint((AIR_HU - intercept) / slope),
                                            info.min, info.max))

    # New series / instance UIDs (FrameOfReferenceUID is kept -> same coordinates as original)
    new.SeriesInstanceUID = series_uid
    new.SOPInstanceUID = generate_uid()
    new.file_meta.MediaStorageSOPInstanceUID = new.SOPInstanceUID
    desc = str(getattr(ds, "SeriesDescription", "")).strip()
    new.SeriesDescription = (f"{desc} FOV{fov_mm:g}" if desc else f"FOV{fov_mm:g}")[:64]
    if "SeriesNumber" in ds and ds.SeriesNumber is not None:
        new.SeriesNumber = int(ds.SeriesNumber) + 1000
    return new


def _safe(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_.^ " else "_" for c in name).strip()
    return keep or "unknown"


def convert(in_dir: Path, out_dir: Path, fov_mm: float = 800.0, matrix: int | None = None) -> int:
    """Convert every CT image under in_dir.

    - Each original series gets its own new SeriesInstanceUID (patients/series are never merged).
    - Output: out_dir/<original sub folder>/<series>/CT_....dcm
    - Files already inside out_dir are skipped (out_dir may be inside in_dir).
    """
    in_dir, out_dir = in_dir.resolve(), out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    new_uids: dict[str, str] = {}
    counters: dict[str, int] = {}
    n = 0
    for path in sorted(p for p in in_dir.rglob("*") if p.is_file()):
        if out_dir == path.resolve().parent or out_dir in path.resolve().parents:
            continue
        try:
            ds = pydicom.dcmread(path)
        except Exception:
            continue
        if "PixelData" not in ds or "ImagePositionPatient" not in ds:
            continue
        if str(getattr(ds, "Modality", "CT")).upper() not in ("CT", ""):
            continue
        old_uid = str(getattr(ds, "SeriesInstanceUID", "")) or str(path.parent)
        if old_uid not in new_uids:
            new_uids[old_uid] = generate_uid()
        m = matrix or int(ds.Rows)
        try:
            new = extend_slice(ds, fov_mm, m, new_uids[old_uid])
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        rel = path.parent.relative_to(in_dir)
        sub = out_dir / rel / _safe(f"S{getattr(ds, 'SeriesNumber', '')}_FOV{fov_mm:g}")
        sub.mkdir(parents=True, exist_ok=True)
        k = counters.get(str(sub), 0)
        counters[str(sub)] = k + 1
        new.save_as(sub / f"CT_FOV{fov_mm:g}_{k:04d}.dcm", enforce_file_format=True)
        n += 1
    for sub, k in sorted(counters.items()):
        print(f"  {k:4d} slices -> {sub}")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--fov", type=float, default=800.0, help="target FOV (mm), default 800")
    ap.add_argument("--matrix", type=int, default=None, help="output matrix (default: same as original, e.g. 512)")
    a = ap.parse_args(argv)
    n = convert(a.input_dir, a.output_dir, a.fov, a.matrix)
    print(f"{n} slices converted -> {a.output_dir}")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
