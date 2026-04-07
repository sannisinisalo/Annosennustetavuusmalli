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
    basedir: Path = (
        BASE_DIR  # Keskitetty juuripolku, johon kaikki suhteelliset polut perustuvat
    )

    def __post_init__(self):
        self.original_root = Path(self.original_root)
        self.patient_folder = Path(self.patient_folder)
        self.processed_root = Path(self.processed_root)
        self.ensure_modified_subdirs()

    # PERUSPOLUT
    @property
    def original_dir(self) -> Path:
        """Alkuperäinen potilaskansio, josta DICOM/RS/RD/RP haetaan."""
        return self.basedir / self.original_root / self.patient_folder.name

    @property
    def modified_dir(self) -> Path:
        """Muokattujen tiedostojen kansio (tulokset)."""
        return self.basedir / self.processed_root / self.patient_folder.name

    # TIEDOSTOLISTAUS
    def _find_dicom_by_modality(self, modality: str) -> list[Path]:
        """Etsitään alkuperäisestä datakansiosta kaikki tietyn modaliteetin DICOM tiedostot"""
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
        """Lista alkuperäisistä CT-viipaleista (täydet polut)"""
        return self._find_dicom_by_modality("CT")

    @property
    def rd_files(self) -> list[Path]:
        """Lista alkuperäisistä RTDOSE-tiedostoista (täydet polut)"""
        return self._find_dicom_by_modality("RTDOSE")

    @property
    def rp_files(self) -> list[Path]:
        """Lista alkuperäisistä RTPLAN-tiedostoista (täydet polut)"""
        return self._find_dicom_by_modality("RTPLAN")

    @property
    def rs_files(self) -> list[Path]:
        """Lista alkuperäisistä RTSTRUCT-tiedostoista (täydet polut)"""
        return self._find_dicom_by_modality("RTSTRUCT")

    @property
    def maski_files(self) -> list[Path]:
        """Lista alkuperäisistä maski-tiedostoista (täydet polut)"""
        return [
            f for f in self.maski_dir.iterdir() if f.is_file() and f.suffix == ".dcm"
        ]

    # MODIFIROITUIEN TIEDOSTOJEN POLUT
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

    # Alkuperäisten CT-kuvien kansio.
    @property
    def ds_org_ct_dir(self) -> Path:
        return self.modified_dir / "vanha ct"

    # Modifioitujen CT-kuvien kansio
    @property
    def ds_ct_dir(self) -> Path:
        """Modifioitujen CT-kuvien kansio"""
        return self.modified_dir / "ct"

    # Luotujen rakennemaskien kansio
    @property
    def maski_dir(self) -> Path:
        """Luotujen rakennemaskien kansio"""
        return self.modified_dir / "maski"

    # Modifioitujen rakennemaskien kansio
    @property
    def ds_maski_dir(self) -> Path:
        """Modifioitujen rakennemaskien kansio"""
        return self.modified_dir / "maskids"

    # Alkuperäisen dose:n kansio
    @property
    def ds_dose_dir(self) -> Path:
        """Alkuperäisen dose:n kansio"""
        return self.modified_dir / "dose"

    # Modifioidun dose:n kansio
    @property
    def ds_doseds_dir(self) -> Path:
        """Modifioidun dose:n kansio"""
        return self.modified_dir / "doseds"

    # Struct:in kansio
    @property
    def ds_struct_dir(self) -> Path:
        """Struct:in kansio, jonne tallennetaan alkuperäiset RTSTRUCT:t, jos niitä halutaan säilyttää"""
        return self.modified_dir / "struct"

    # Plan:in kansio
    @property
    def ds_plan_dir(self) -> Path:
        """Plan:in kansio jonne tallennetaan alkuperäiset RTPLAN:t, jos niitä halutaan säilyttää"""
        return self.modified_dir / "plan"

    @property
    def number(self) -> int:
        """Potilaan numeron kansion nimestä järjestystä varten"""
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

    def _find_patients(self) -> list[Patient]:
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
    def patients(self) -> list[Patient]:
        return self._patients

    def sorted_by_number(self) -> list[Patient]:
        return sorted(self._patients, key=lambda p: p.number)

@dataclass(frozen=True)
class DoseMetricsConfig:
    organ_config: dict = None
    dx_percentages: list = None
    vx_thresholds: list = None
    vx_organs: list = None

    def __post_init__(self):
        object.__setattr__(self, 'organ_config', self.organ_config or {
            "PTV": 1,
            "Heart": 2,
            "Contralateral lung": 3,
            "Ipsilateral lung": 4,
            "Contralateral breast": 5,
        })
        object.__setattr__(
            self, 
            'dx_percentages', 
            self.dx_percentages or 
            [98.5, 95, 90, 75, 50, 25, 10, 2]
        )
        object.__setattr__(
            self, 
            'vx_thresholds', 
            self.vx_thresholds or 
            [35, 16, 8, 4]
        )
        object.__setattr__(
            self, 
            'vx_organs', 
            self.vx_organs or 
            ["Heart", "Ipsilateral lung", "Contralateral lung", "Contralateral breast"]
        )
        
        
        
        
