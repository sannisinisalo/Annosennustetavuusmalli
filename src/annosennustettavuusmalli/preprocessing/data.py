# -*- coding: utf-8 -*-
"""
Luotu Ti 24.2.2026
Tekijä: Sanni Sinisalo

Luokkarakenne yhdistämään ja vähentämään toistoa preprocessing koodeissa.
Alkuperäisille ja muokatuille tiedostoille tehty eri tiedostopolut.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pydicom

# Keskitetty juuripolku:
BASE_DIR = Path(r"C:\Users\User01\GRADU\Aineisto")


@dataclass
class Patient:
    original_root: Path  # esim. "VN0", mistä lähdetiedostot haetaan
    patient_folder: Path  # esim. Patient1_VN0
    processed_root: Path  # esim. "VN0ds", mihin tulokset tallennetaan

    def __post_init__(self):
        self.original_root = Path(self.original_root)
        self.patient_folder = Path(self.patient_folder)
        self.processed_root = Path(self.processed_root)
        self.ensure_modified_subdirs()

    # PERUSPOLUT
    @property
    def original_dir(self) -> Path:
        """Alkuperäinen potilaskansio, josta DICOM/RS/RD/RP haetaan."""
        return BASE_DIR / self.original_root / self.patient_folder.name

    @property
    def modified_dir(self) -> Path:
        """Muokattujen tiedostojen kansio (tulokset)."""
        return BASE_DIR / self.processed_root / self.patient_folder.name

    # TIEDOSTOLISTAUS
    def _find_dicom_by_modality(self, modality: str) -> list[Path]:
        """Etsitään kansiosta kaikki tietyn modaliteetin DICOM tiedostot"""
        dicom_files = []
        for f in self.original_dir.iterdir():
            if f.is_file() and f.suffix == ".dcm":
                try:
                    ds = pydicom.dcmread(f, stop_before_pixels=True)
                    if ds.Modality == modality:
                        dicom_files.append(f)
                except Exception:
                    pass
        return dicom_files

    @property
    def ct_files(self) -> list[Path]:
        """Lista CT-viipaleista (täydet polut)"""
        return self._find_dicom_by_modality("CT")

    @property
    def rd_files(self) -> list[Path]:
        """Lista RTDOSE-tiedostoista (täydet polut)"""
        return self._find_dicom_by_modality("RTDOSE")

    @property
    def rp_files(self) -> list[Path]:
        """Lista RTPLAN-tiedostoista (täydet polut)"""
        return self._find_dicom_by_modality("RTPLAN")

    @property
    def rs_files(self) -> list[Path]:
        """Lista RTSTRUCT-tiedostoista (täydet polut)"""
        return self._find_dicom_by_modality("RTSTRUCT")

    # MODIFIROITUIEN TIEDOSTOJEN POLUT
    # Alkuperäisten CT-kuvien kansio

    def ensure_modified_subdirs(self):
        """Varmistetaan, että kaikki tarvittavat alikansiot on luotu"""
        subdirs = [
            "vanha ct",
            "ct",
            "maski",
            "maskids",
            "dose",
            "doseds",
            "struct",
            "plan",
        ]
        for subdir in subdirs:
            dir_path = self.modified_dir / subdir
            dir_path.mkdir(parents=True, exist_ok=True)

    @property
    def ds_org_ct_dir(self) -> Path:
        return self.modified_dir / "vanha ct"

    # Modifioitujen CT-kuvien kansio
    @property
    def ds_ct_dir(self) -> Path:
        return self.modified_dir / "ct"

    # Luotujen rakennemaskien kansio
    @property
    def ds_mask_dir(self) -> Path:
        return self.modified_dir / "maski"

    # Modifioitujen rakennemaskien kansio
    @property
    def ds_maskds_dir(self) -> Path:
        return self.modified_dir / "maskids"

    # Alkuperäisen dose:n kansio
    @property
    def ds_dose_dir(self) -> Path:
        return self.modified_dir / "dose"

    # Modifioidun dose:n kansio
    @property
    def ds_doseds_dir(self) -> Path:
        return self.modified_dir / "doseds"

    # Struct:in kansio
    @property
    def ds_struct_dir(self) -> Path:
        return self.modified_dir / "struct"

    # Plan:in kansio
    @property
    def ds_plan_dir(self) -> Path:
        return self.modified_dir / "plan"

    # NUMERON HAKU JÄRJESTYSTÄ VARTEN
    @property
    def number(self) -> int:
        m = re.search(r"(\d+)", str(self.patient_folder))
        return int(m.group(1)) if m else 0


@dataclass
class AllPatients:
    original_root: Path
    processed_root: Path

    def __post_init__(self):
        self.original_root = Path(self.original_root).resolve()
        self.processed_root = Path(self.processed_root).resolve()
        self._patients = self._find_patients()

    def _find_patients(self):
        patient_dirs = self.original_root.glob("Patient*")

        if not patient_dirs:
            raise FileNotFoundError(f"No patients found in {self.original_root}")

        patients = [
            Patient(
                original_root=self.original_root,
                processed_root=self.processed_root,
                patient_folder=p,
            )
            for p in patient_dirs
        ]

        return patients

    @property
    def patients(self):
        return self._patients

    def sorted_by_number(self):
        return sorted(self._patients, key=lambda p: p.number)
