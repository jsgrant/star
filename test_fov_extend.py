"""Synthetic CT tests for fov_extend.py."""
import collections
import shutil

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from fov_extend import convert


def make_ct(path, z, fov=500.0, n=512, series_uid="1.2.3", series_no=2, image_type=None, sop=None):
    sp = fov / n
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = sop or generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.PatientID = "P1"
    ds.PatientName = "TEST^ONE"
    if image_type:
        ds.ImageType = image_type
    ds.FrameOfReferenceUID = "1.2.3.4"
    ds.Modality = "CT"
    ds.SeriesNumber = series_no
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
    b = pydicom.dcmread(sorted(dst.rglob("*.dcm"))[0])
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


def test_multi_patient_and_out_inside_in(tmp_path):
    src = tmp_path / "imported"
    for pat, uid in (("p1", "1.1"), ("p2", "2.2")):
        (src / pat).mkdir(parents=True)
        for k in range(2):
            make_ct(src / pat / f"{k}.dcm", z=k * 3.0, series_uid=uid)
    out = src / "FOV800 completed"
    assert convert(src, None, 800.0) == 4           # default output folder = <input>/FOV800 completed
    files = sorted(out.rglob("*.dcm"))
    assert len(files) == 4
    uids = collections.Counter(str(pydicom.dcmread(f).SeriesInstanceUID) for f in files)
    assert len(uids) == 2 and set(uids.values()) == {2}


def test_duplicate_copies_converted_once(tmp_path):
    """CD/PACS export: same images in IHE_PDI and Storage -> one series, no duplicated slices."""
    a = tmp_path / "pat" / "IHE_PDI" / "CT"
    b = tmp_path / "pat" / "Storage" / "CT"
    a.mkdir(parents=True); b.mkdir(parents=True)
    for k in range(3):
        make_ct(a / f"{k}", z=k * 3.0)
    for f in a.iterdir():
        shutil.copy(f, b / f.name)
    assert convert(tmp_path / "pat", tmp_path / "out", 800.0) == 3
    files = list((tmp_path / "out").rglob("*.dcm"))
    zs = collections.Counter((str(pydicom.dcmread(f).SeriesInstanceUID),
                              float(pydicom.dcmread(f).ImagePositionPatient[2])) for f in files)
    assert len(files) == 3 and max(zs.values()) == 1


def test_rerun_is_idempotent_and_skips_results(tmp_path):
    src = tmp_path / "imported"
    src.mkdir()
    for k in range(3):
        make_ct(src / f"{k}.dcm", z=k * 3.0)
    convert(src, None, 800.0)
    first = {f.name: str(pydicom.dcmread(f).SOPInstanceUID) for f in (src / "FOV800 completed").rglob("*.dcm")}
    # an earlier eFOV result placed elsewhere inside the input must not be converted again
    other = src / "800 completed"
    other.mkdir()
    for f in (src / "FOV800 completed").rglob("*.dcm"):
        shutil.copy(f, other / f.name)
    assert convert(src, None, 800.0) == 3
    files = list((src / "FOV800 completed").rglob("*.dcm"))
    again = {f.name: str(pydicom.dcmread(f).SOPInstanceUID) for f in files}
    assert again == first                      # same UIDs, same files, nothing added


def test_localizer_and_large_fov_skipped(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    make_ct(src / "scout.dcm", z=0.0, series_uid="9.9", image_type=["ORIGINAL", "PRIMARY", "LOCALIZER"])
    make_ct(src / "big.dcm", z=0.0, series_uid="8.8", fov=800.0)
    make_ct(src / "ok.dcm", z=0.0, series_uid="7.7")
    assert convert(src, tmp_path / "out", 800.0) == 1


def test_series_number_and_order(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    for k, z in enumerate((6.0, 0.0, 3.0)):                # files not in z order
        make_ct(src / f"{k}.dcm", z=z)
    convert(src, tmp_path / "out", 800.0)
    files = sorted((tmp_path / "out").rglob("CT_*.dcm"))
    zs = [float(pydicom.dcmread(f).ImagePositionPatient[2]) for f in files]
    assert zs == sorted(zs)
    d = pydicom.dcmread(files[0])
    assert int(d.SeriesNumber) == 102 and "DERIVED" == d.ImageType[0]
