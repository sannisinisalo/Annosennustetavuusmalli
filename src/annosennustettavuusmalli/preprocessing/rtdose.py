# -*- coding: utf-8 -*-
"""
Luotu To 29.1.2026

Tekijä: Sanni Sinisalo

Koodi RTDose tiedoston upsamplaamiseen.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
import SimpleITK as sitk
from loguru import logger  # type: ignore
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from annosennustettavuusmalli.preprocessing.luokat import AllPatients, Patient


def patient_number(name):
    m = re.search(r"\d+", name)
    return int(m.group()) if m else 999999


def find_dose_file(folder: Path):
    for f in folder.iterdir():
        if f.is_file() and f.name.startswith("RD") and f.name.endswith(".dcm"):
            return f
    return None


def find_ct_files(folder: Path):
    """
    Palauttaa listan CT-viipaleista (täydet polut)
    """
    ct_files = []
    for f in folder.iterdir():
        if f.is_file() and f.name.endswith(".dcm") and not f.name.startswith("RD"):
            try:
                ds = pydicom.dcmread(folder / f, stop_before_pixels=True)
                if ds.Modality == "CT":
                    ct_files.append(folder / f)
            except Exception:
                pass
    return ct_files


def load_ct_series_from_files(ct_files: list[Path]) -> sitk.Image:
    """
    Rakentaa 3D-CT-kuvan listasta 2D-viipaleita
    """
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(ct_files)
    image = reader.Execute()
    logger.debug(f"CT ladattu ({image.GetSize()[2]} viipaletta)")
    return image


@dataclass
class CT3DImage:
    patient: Patient
    forced_spacing: float = 2.0
    original_image: sitk.Image = field(init=False)
    image: sitk.Image = field(init=False)
    size: tuple[int, int, int] = field(init=False)
    positions: list[float] = field(init=False)
    spacing: tuple[float, float, float] = field(init=False)
    origin: tuple[float, float, float] = field(init=False)
    direction: tuple[float, float, float, float, float, float, float, float, float] = (
        field(init=False)
    )
    ref_meta: FileDataset = field(init=False)
    iop: list[float] = field(init=False)
    ipp: list[float] = field(init=False)
    ps: list[float] = field(init=False)
    th: float = field(init=False)
    for_: Optional[str] = field(init=False)

    def __post_init__(self):
        self.ct_files = self.patient.ct_files
        self.original_image = load_ct_series_from_files(self.ct_files)
        self.image = self.original_image
        self.size = self.image.GetSize()
        self.positions = self.get_positions()
        self.spacing = self.image.GetSpacing()
        self.origin = self.image.GetOrigin()
        self.direction = self.image.GetDirection()

        if self.spacing[2] != self.forced_spacing:
            new_spacing = list(self.spacing)
            new_spacing[2] = self.forced_spacing
            self.image.SetSpacing(tuple(new_spacing))
            logger.debug(f"CT spacing pakotettu: {self.image.GetSpacing()}")
            self.spacing = self.image.GetSpacing()

        self.ref_meta = pydicom.dcmread(self.ct_files[0], stop_before_pixels=True)
        self.iop = self.ref_meta.ImageOrientationPatient
        self.ipp = self.ref_meta.ImagePositionPatient
        self.ps = self.ref_meta.PixelSpacing

        thickness = getattr(self.ref_meta, "SliceThickness", None)
        logger.debug(f"SliceThickness raw value: {thickness}")

        if thickness is None:
            thickness = self.spacing[2]

        self.th = float(thickness)
        self.for_ = getattr(self.ref_meta, "FrameOfReferenceUID", None)

    def get_positions(self):
        return sorted(
            [
                pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2]
                for f in self.ct_files
            ]
        )


@dataclass
class Dose3DImage:
    patient: Patient
    ds: FileDataset = field(
        init=False
    )  # koko RTDOSE DICOM dataset, sisältää pixel datan
    ds_meta: FileDataset = field(init=False)  # ilman pixel dataa, vain metadata
    image: sitk.Image = field(init=False)
    scaling: float = field(init=False)
    size: tuple[int, int, int] = field(init=False)
    spacing: tuple[float, float, float] = field(init=False)
    origin: tuple[float, float, float] = field(init=False)
    direction: tuple[float, float, float, float, float, float, float, float, float] = (
        field(init=False)
    )
    slice_thickness: float = field(init=False)
    gfov: Optional[list[float]] = field(init=False)

    def __post_init__(self):
        if self.patient.rd_file is None:
            raise ValueError(f"Potilaalla {self.patient} ei RTDOSE-tiedostoa")

        self.ds_meta = pydicom.dcmread(self.patient.rd_file, stop_before_pixels=True)
        self.scaling = float(self.ds_meta.DoseGridScaling)
        self.ds = pydicom.dcmread(self.patient.rd_file)

        self.image = (
            sitk.ReadImage(self.patient.rd_file, sitk.sitkFloat32) * self.scaling
        )

        self.size = self.image.GetSize()
        self.spacing = self.image.GetSpacing()
        self.origin = self.image.GetOrigin()
        self.direction = self.image.GetDirection()

        thickness = getattr(self.ds_meta, "SliceThickness", None)

        if thickness is None:
            thickness = self.spacing[2]

        self.slice_thickness = float(thickness)

        self.gfov = (
            list(self.ds_meta.GridFrameOffsetVector)
            if "GridFrameOffsetVector" in self.ds_meta
            else None
        )


def unify_dose_with_ct(patient: Patient) -> Optional[FileDataset]:
    """
    Lataa RTDOSE-tiedoston, resamplaa sen CT:n koordinaatistoon ja tallentaa uuden DICOM-tiedoston.
    Palauttaa resamplatun RTDOSE DICOM datasetin.
    """

    if patient.rd_file is None:
        logger.warning(f"Potilaalla {patient} ei RTDOSE-tiedostoa, ohitetaan.")
        return None

    ds = None  # Alustetaan ds, jotta voidaan palauttaa myös virhetilanteessa

    try:
        # Luetaan CT imageksi ja pakotetaan spacing 2.0
        ct_img = CT3DImage(patient=patient, forced_spacing=2.0)

        # Luetaan dose

        dose_img = Dose3DImage(patient=patient)
        ds = dose_img.ds

        new_size = [ct_img.size[0], ct_img.size[1], ct_img.size[2]]
        new_spacing = [ct_img.spacing[0], ct_img.spacing[1], ct_img.forced_spacing]
        new_origin = [ct_img.origin[0], ct_img.origin[1], dose_img.origin[2]]

        reference = sitk.Image(new_size, sitk.sitkFloat32)
        reference.SetSpacing(new_spacing)
        reference.SetOrigin(new_origin)
        reference.SetDirection(dose_img.direction)

        # Resamplataan dose
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)

        dose_resampled = resampler.Execute(dose_img.image)

        # Tallennetaan tiedostot
        dose_array = sitk.GetArrayFromImage(dose_resampled)

        new_scaling = 0.001
        stored_values = np.round(dose_array / new_scaling).astype(np.uint16)

        ds.PixelData = stored_values.tobytes()
        ds.Rows = stored_values.shape[1]
        ds.Columns = stored_values.shape[2]
        ds.NumberOfFrames = stored_values.shape[0]

        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.DoseGridScaling = new_scaling

        ds.PixelSpacing = [float(new_spacing[1]), float(new_spacing[0])]
        ds.SliceThickness = dose_img.slice_thickness

        new_gfov = [i * new_spacing[2] for i in range(ds.NumberOfFrames)]
        ds.GridFrameOffsetVector = new_gfov

        ds.ImagePositionPatient = [
            float(ct_img.origin[0]),
            float(ct_img.origin[1]),
            float(dose_img.origin[2]),
        ]

        if ct_img.for_ is not None:
            ds.FrameOfReferenceUID = ct_img.for_

        if not hasattr(ds, "file_meta") or ds.file_meta is None:
            ds.file_meta = FileMetaDataset()

        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.file_meta.ImplementationClassUID = generate_uid()

    except Exception as e:
        logger.warning(f"Virhe potilaalla {patient}: {e}")

    return ds if ds is not None else None


if __name__ == "__main__":
    from tqdm import tqdm  # type: ignore

    patients = AllPatients(original_root=Path("VN0"), processed_root=Path("VN0ds"))

    total = len(patients)

    for patient in tqdm(
        patients.sort_by_number(),
        total=total,
        desc="Käsitellään potilaita",
    ):
        if patient.rd_file is None:
            logger.warning(f"Potilaalla {patient} ei RTDOSE-tiedostoa, ohitetaan.")
            continue
        ds = unify_dose_with_ct(patient)

        out_path = patient.ds_dose_dir / patient.rd_file.name

        if ds is not None:
            ds.save_as(out_path, write_like_original=False)

            ds2 = pydicom.dcmread(out_path)
            dose_check = ds2.pixel_array * float(ds2.DoseGridScaling)

            logger.success(f"Potilaan {patient} RTDose tallennettu onnistuneesti")
        else:
            logger.error(f"Potilaalla {patient} RTDOSE-käsittely epäonnistui")
