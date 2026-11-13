#!/usr/bin/env python3
"""
CT DICOM FOV extender (예: FOV 500 mm -> FOV 800 mm)

What it does
------------
1. Keeps the matrix (default: same as the original, e.g. 512) and changes the
   pixel size to FOV / matrix (800 / 512 = 1.5625 mm) — the same layout a CT
   scanner writes for an extended-FOV (eFOV 800) series.
2. Resamples the original image (bilinear) into the new grid around the same
   physical centre (= reconstruction centre). The outside is filled with air
   (-1024 HU).
3. Recomputes ImagePositionPatient, so the patient coordinates
   (FrameOfReference) match the original exactly.

Safety rules
------------
- The same image stored twice (CD / PACS export with IHE_PDI + Storage copies)
  is converted only once (de-duplicated by SOPInstanceUID).
- One output folder per source series: <out>/<PatientID_Name>/<S{no}_{desc}_FOV800_{id}>/
- Series that already have a FOV >= the target (e.g. earlier eFOV 800 results),
  localizer / scout images and multi-frame images are skipped.
- Files already inside the output folder are skipped (it may live inside the
  input folder).
- UIDs of the output are derived from the source UIDs, so running the
  conversion again overwrites the same files instead of creating duplicates.

Note: This is NOT the scanner's extended-FOV reconstruction from raw data
(Siemens HD FOV / FAST eFoV). Anatomy cut off outside the original FOV is not
recovered; only the image size and grid are extended.

Usage:
    python fov_extend.py <input_dir> [output_dir] [--fov 800] [--matrix 512] [--series-offset 100]
    (output_dir default: <input_dir>/FOV800 completed)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

AIR_HU = -1024.0  # value outside the original FOV / below air


def _geometry(ds):
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]          # direction j increases, direction i increases
    ipp = np.array(ds.ImagePositionPatient, dtype=float)
    dr, dc = (float(v) for v in ds.PixelSpacing)  # [row spacing(between rows), column spacing]
    return ipp, row_dir, col_dir, dr, dc


def current_fov(ds) -> float:
    dr, dc = (float(v) for v in ds.PixelSpacing)
    return max(int(ds.Columns) * dc, int(ds.Rows) * dr)


def _bilinear(img: np.ndarray, ii: np.ndarray, jj: np.ndarray, cval: float) -> np.ndarray:
    """Sample img at fractional (row ii, col jj); outside the image -> cval."""
    h, w = img.shape
    inside = (ii >= -0.5) & (ii <= h - 0.5) & (jj >= -0.5) & (jj <= w - 0.5)
    ic, jc = np.clip(ii, 0, h - 1), np.clip(jj, 0, w - 1)
    i0, j0 = np.floor(ic).astype(int), np.floor(jc).astype(int)
    i1, j1 = np.minimum(i0 + 1, h - 1), np.minimum(j0 + 1, w - 1)
    fi, fj = ic - i0, jc - j0
    top = img[i0, j0] * (1 - fj) + img[i0, j1] * fj
    bot = img[i1, j0] * (1 - fj) + img[i1, j1] * fj
    out = top * (1 - fi) + bot * fi
    out[~inside] = cval
    return out.astype(np.float32)


def _ds_str(v: float) -> str:
    """Decimal String (max 16 characters)."""
    s = f"{v:.10g}"
    return s if len(s) <= 16 else f"{v:.6g}"


def _uid(*parts) -> str:
    """Deterministic UID from the given parts (same input -> same UID)."""
    return generate_uid(entropy_srcs=[str(p) for p in parts])


def extend_slice(ds, fov_mm: float, matrix: int, series_uid: str, series_offset: int = 100):
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
    out_hu = _bilinear(hu, ii, jj, AIR_HU)

    # HU -> stored value (keep original slope/intercept, data type and BitsStored range)
    signed = int(getattr(ds, "PixelRepresentation", 0) or 0) == 1
    bits = int(getattr(ds, "BitsAllocated", 16) or 16)
    stored_bits = int(getattr(ds, "BitsStored", bits) or bits)
    lo, hi = ((-(1 << (stored_bits - 1)), (1 << (stored_bits - 1)) - 1) if signed
              else (0, (1 << stored_bits) - 1))
    dtype = {8: np.int8 if signed else np.uint8, 16: np.int16 if signed else np.uint16,
             32: np.int32 if signed else np.uint32}[bits]
    stored = np.clip(np.rint((out_hu - intercept) / slope), lo, hi).astype(dtype)

    new = copy.deepcopy(ds)
    new.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    new.Rows = matrix
    new.Columns = matrix
    new.PixelSpacing = [f"{new_sp:.10g}", f"{new_sp:.10g}"]
    new.ImagePositionPatient = [_ds_str(v) for v in new_ipp]
    new.ReconstructionDiameter = f"{fov_mm:g}"
    new.PixelData = stored.tobytes()
    new["PixelData"].VR = "OB" if bits == 8 else "OW"
    for tag in ("SmallestImagePixelValue", "LargestImagePixelValue", "SourceImageSequence"):
        if tag in new:
            del new[tag]
    if "PixelPaddingValue" in new:
        new.PixelPaddingValue = int(np.clip(np.rint((AIR_HU - intercept) / slope), lo, hi))
    img_type = list(getattr(new, "ImageType", []) or ["DERIVED", "SECONDARY"])
    img_type[0] = "DERIVED"
    new.ImageType = img_type
    new.DerivationDescription = (f"FOV extended {current_fov(ds):.1f}->{fov_mm:g} mm, matrix {matrix}, "
                                 f"resampled (bilinear), outside air {AIR_HU:g} HU, fov_extend.py")

    # New series / instance UIDs (derived from the source -> stable on re-run).
    # FrameOfReferenceUID is kept -> same coordinates as the original.
    new.SeriesInstanceUID = series_uid
    new.SOPInstanceUID = _uid("fov_extend", ds.SOPInstanceUID, fov_mm, matrix)
    new.file_meta.MediaStorageSOPInstanceUID = new.SOPInstanceUID
    desc = str(getattr(ds, "SeriesDescription", "")).strip()
    new.SeriesDescription = (f"{desc} eFOV {fov_mm:g}" if desc else f"eFOV {fov_mm:g}")[:64]
    if "SeriesNumber" in ds and ds.SeriesNumber is not None:
        new.SeriesNumber = int(ds.SeriesNumber) + series_offset
    return new


def _safe(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_.^ " else "_" for c in name).strip(" ._")
    return keep[:60] or "unknown"


def _skip_reason(ds, fov_mm: float) -> str | None:
    if str(getattr(ds, "Modality", "")).upper() != "CT":
        return "not CT"
    if "Rows" not in ds or "ImagePositionPatient" not in ds or "PixelSpacing" not in ds:
        return "no image geometry"
    if int(getattr(ds, "NumberOfFrames", 1) or 1) > 1:
        return "multi-frame"
    if any("LOCALIZER" in str(v).upper() for v in (getattr(ds, "ImageType", []) or [])):
        return "localizer"
    if current_fov(ds) >= fov_mm - 1.0:
        return f"FOV already {current_fov(ds):.0f} mm"
    return None


def convert(in_dir: Path, out_dir: Path | None = None, fov_mm: float = 800.0, matrix: int | None = None,
            series_offset: int = 100, log=print) -> int:
    """Convert every CT series under in_dir. Returns the number of slices written."""
    in_dir = in_dir.resolve()
    out_dir = (out_dir or in_dir / f"FOV{fov_mm:g} completed").resolve()

    # 1) scan headers, de-duplicate by SOPInstanceUID, group by series
    series: dict[str, dict] = {}
    seen: set[str] = set()
    skipped: dict[str, int] = defaultdict(int)
    for path in sorted(p for p in in_dir.rglob("*") if p.is_file()):
        rp = path.resolve()
        if out_dir == rp.parent or out_dir in rp.parents:
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            continue
        if "SOPInstanceUID" not in ds:
            continue
        sop = str(ds.SOPInstanceUID)
        if sop in seen:
            skipped["duplicate copy"] += 1
            continue
        seen.add(sop)
        why = _skip_reason(ds, fov_mm)
        if why:
            skipped[why] += 1
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "")) or str(path.parent)
        s = series.setdefault(uid, {"files": [], "ds": ds})
        s["files"].append((float(ds.ImagePositionPatient[2]), path))

    # 2) convert series by series
    n = 0
    for src_uid, s in series.items():
        d0 = s["ds"]
        m = matrix or int(d0.Rows)
        new_uid = _uid("fov_extend", src_uid, fov_mm, m)
        pat = _safe(f"{getattr(d0, 'PatientID', '')}_{getattr(d0, 'PatientName', '')}")
        tag = hashlib.sha1(src_uid.encode()).hexdigest()[:6]   # keeps different series with the same name apart
        sname = _safe(f"S{getattr(d0, 'SeriesNumber', '')}_{getattr(d0, 'SeriesDescription', '')}_FOV{fov_mm:g}_{tag}")
        sub = out_dir / pat / sname
        sub.mkdir(parents=True, exist_ok=True)
        for old in sub.glob("CT_*.dcm"):          # re-run: replace previous result of this series
            old.unlink()
        files = sorted(s["files"])
        done = 0
        for k, (_z, path) in enumerate(files):
            try:
                new = extend_slice(pydicom.dcmread(path), fov_mm, m, new_uid, series_offset)
            except Exception as e:
                log(f"  skip {path}: {e}")
                continue
            new.save_as(sub / f"CT_{k + 1:04d}.dcm", enforce_file_format=True)
            done += 1
        n += done
        log(f"  {done:4d} slices  S{getattr(d0, 'SeriesNumber', '')} -> {sub}")
    for why, c in sorted(skipped.items()):
        log(f"  (skipped {c} files: {why})")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path, nargs="?", default=None,
                    help="default: <input_dir>/FOV<fov> completed")
    ap.add_argument("--fov", type=float, default=800.0, help="target FOV (mm), default 800")
    ap.add_argument("--matrix", type=int, default=None, help="output matrix (default: same as original, e.g. 512)")
    ap.add_argument("--series-offset", type=int, default=100,
                    help="new SeriesNumber = original + this (default 100, e.g. 2 -> 102)")
    a = ap.parse_args(argv)
    if not a.input_dir.is_dir():
        print(f"input folder not found: {a.input_dir}")
        return 2
    n = convert(a.input_dir, a.output_dir, a.fov, a.matrix, a.series_offset)
    out = a.output_dir or a.input_dir / f"FOV{a.fov:g} completed"
    print(f"{n} slices converted -> {out}")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
