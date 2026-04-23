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
from typing import Optional
import shutil
import tempfile
import pydicom
from loguru import logger  
from pydicom.dataset import FileDataset  

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
        self._ct_files = self._find_dicom_by_modality("CT")
        self._rd_file = self._find_dicom_by_modality("RTDOSE")
        self._rp_file = self._find_dicom_by_modality("RTPLAN")
        self._rs_file = self._find_dicom_by_modality("RTSTRUCT")

    
    def prepare_ct_dir_for_rtutils(self) -> Path:
        """
        Luo väliaikaisen hakemiston, jossa on vain CT DICOMit.
        Tätä käytetään RTStructBuilderille.
        """
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"ct_only_{self.patient_folder.name}_"))
    
        for ct_file in self.ct_files:
            shutil.copy2(ct_file, tmp_dir / ct_file.name)
    
        logger.debug(
            f"Created temporary CT-only directory for RTUtils: {tmp_dir} "
            f"({len(list(tmp_dir.glob('*.dcm')))} files)"
        )
    
        return tmp_dir


    # PERUSPOLUT
    @property
    def original_dir(self) -> Path:
        """Alkuperäinen potilaskansio, josta DICOM/RS/RD/RP haetaan."""
        return self.basedir / self.original_root.name / self.patient_folder.name

    @property
    def modified_dir(self) -> Path:
        """Muokattujen tiedostojen kansio (tulokset)."""
        return self.basedir / self.processed_root.name / self.patient_folder.name

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
        if not dicom_files:
            logger.warning(
                f"No {modality} DICOM files found for patient {self.patient_folder.name} in {self.original_dir}"
            )

        return sorted(dicom_files)

    @property
    def ct_files(self) -> list[Path]:
        """Lista alkuperäisistä CT-viipaleista (täydet polut)"""
        return self._ct_files

    def get_ct_slices(self) -> list[FileDataset]:
        """Lista alkuperäisistä CT-viipaleista DICOM-dataset muodossa

        Kokeillaan lukea kaikki CT-viipaleet ja suodatetaan ne, jotka kuuluvat samaan sarjaan (SeriesInstanceUID).
        Järjestetään viipaleet InstanceNumber:n mukaan, käänteisessä järjestyksessä, jotta slice 1 on viimeisenä listassa.

        """
        ct_files = self.ct_files

        try:
            ct_slices = [pydicom.dcmread(f) for f in ct_files]
            series_uid = ct_slices[0].SeriesInstanceUID
            logger.debug(f"Found SeriesInstanceUID: {series_uid} in first CT slice.")
            ct_slices = [
                s
                for s in ct_slices
                if getattr(s, "SeriesInstanceUID", None) == series_uid
            ]
            logger.info(
                f"Loaded {len(ct_slices)} CT slices with SeriesInstanceUID: {series_uid}"
            )
            ct_slices = sorted(
                ct_slices, key=lambda x: int(x.InstanceNumber), reverse=True
            )  # Käännetään järjestys, jotta slice 1 on viimeisenä

        except Exception as e:
            logger.warning(f"Failed to read CT files into DICOM datasets: {e}")
            ct_slices = []

        return ct_slices

    @property
    def rd_file(self) -> Optional[Path]:
        """RTDOSE-tiedoston polku, jos löytyy"""
        if len(self._rd_file) > 1:
            logger.warning(
                f"Multiple RTDOSE files found for patient {self.patient_folder.name}. Using the first one: {self._rd_file[0]}"
            )
        return self._rd_file[0] if self._rd_file else None

    @property
    def rp_file(self) -> Optional[Path]:
        """RTPLAN-tiedoston polku, jos löytyy"""
        if len(self._rp_file) > 1:
            logger.warning(
                f"Multiple RTPLAN files found for patient {self.patient_folder.name}. Using the first one: {self._rp_file[0]}"
            )
        return self._rp_file[0] if self._rp_file else None

    @property
    def rs_file(self) -> Optional[Path]:
        """RTSTRUCT-tiedoston polku, jos löytyy"""
        if len(self._rs_file) > 1:
            logger.warning(
                f"Multiple RTSTRUCT files found for patient {self.patient_folder.name}. Using the first one: {self._rs_file[0]}"
            )
        return self._rs_file[0] if self._rs_file else None

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
    base_dir: Path = BASE_DIR

    def __post_init__(self):
        self.original_root = self.ensure_paths(self.original_root)
        self.processed_root = self.ensure_paths(self.processed_root)
        self._patients = self._find_patients()

    def ensure_paths(self, path: Path):
        """Varmistaa, että annettu polku on olemassa, muuten luo sen."""
        logger.debug(f"Ensuring path exists: {path.name}")

        output_path = self.base_dir / path.name

        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        return output_path

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

        return sorted(patients, key=lambda p: p.number)

    @property
    def patients(self) -> list[Patient]:
        return self._patients

    def sort_by_number(self) -> list[Patient]:
        return sorted(self._patients, key=lambda p: p.number)

    def __len__(self):
        return len(self._patients)


