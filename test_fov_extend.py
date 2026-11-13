"""Synthetic CT test: verifies that after FOV 500 -> 800 the body is smaller in pixels and physical size/position are preserved."""
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from fov_extend import convert


def make_ct(path, z, fov=500.0, n=512):
    sp = fov / n
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = "1.2.3"
    ds.FrameOfReferenceUID = "1.2.3.4"
    ds.Modality = "CT"
    ds.SeriesNumber = 2
    ds.Rows = ds.Columns = n
    ds.PixelSpacing = [sp, sp]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    # Recon centre at (10, -20) mm
    cx, cy = 10.0, -20.0
    ds.ImagePositionPatient = [cx - sp * (n - 1) / 2, cy - sp * (n - 1) / 2, z]
    ds.ReconstructionDiameter = fov
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024
    # Air background + 300 mm diameter "body" (0 HU) centred at the recon centre
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot((x - (n - 1) / 2) * sp, (y - (n - 1) / 2) * sp)
    hu = np.where(r < 150, 0, -1000)
    hu[r > fov / 2] = -3024 + 1024  # outside recon circle (stored 0)
    ds.PixelData = (hu + 1024).clip(0).astype(np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)


def body_width_px(ds):
    hu = ds.pixel_array.astype(float) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    row = hu[ds.Rows // 2]
    return int((row > -500).sum())


def test_fov800(tmp_path):
    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    for k in range(3):
        make_ct(src / f"{k}.dcm", z=k * 3.0)
    assert convert(src, dst, 800.0) == 3

    a = pydicom.dcmread(src / "0.dcm")
    b = pydicom.dcmread(sorted(dst.iterdir())[0])
    assert (b.Rows, b.Columns) == (512, 512)
    assert abs(float(b.PixelSpacing[0]) - 800 / 512) < 1e-4
    assert float(b.ReconstructionDiameter) == 800

    # In pixels, the body shrinks to 500/800 = 0.625x
    wa, wb = body_width_px(a), body_width_px(b)
    assert abs(wb / wa - 0.625) < 0.02, (wa, wb)
    # Physical width (mm) stays the same (~300 mm)
    assert abs(wb * float(b.PixelSpacing[1]) - 300) < 4
    assert abs(wa * float(a.PixelSpacing[1]) - 300) < 4

    # Same reconstruction centre / same FrameOfReference
    def center(ds):
        sp = float(ds.PixelSpacing[0])
        p = np.array(ds.ImagePositionPatient, float)
        return p[:2] + sp * (ds.Rows - 1) / 2
    assert np.allclose(center(a), center(b), atol=1e-3)
    assert b.FrameOfReferenceUID == a.FrameOfReferenceUID
    assert b.SeriesInstanceUID != a.SeriesInstanceUID

    # Outside of new frame = air
    hu_b = b.pixel_array.astype(float) - 1024
    assert hu_b[0, 0] == -1024 and hu_b[256, 5] == -1024
