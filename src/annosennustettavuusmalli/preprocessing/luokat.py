# -*- coding: utf-8 -*-
"""
Luotu Ti 24.2.2026
Tekijä: Sanni Sinisalo

Luokkarakenne yhdistämään ja vähentämään toistoa preprocessing koodeissa.
Alkuperäisille ja muokatuille tiedostoille tehty eri tiedostopolut.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List

# Keskitetty juuripolku:
BASE_DIR = Path(r"C:\Users\User01\GRADU\Aineisto")

@dataclass
class Patient:
    processed_dataset: Path   # esim. "VN0ds", mihin tulokset tallennetaan
    original_dataset: Path    # esim. "VN0", mistä lähdetiedostot haetaan
    patient_folder: Path      # esim. Patient1_VN0
    
    def __post_init__(self):
        # Varmistetaan, että kaikki parametrit ovat Path-objekteja
        self.processed_dataset = Path(self.processed_dataset)
        self.original_dataset = Path(self.original_dataset)
        self.patient_folder = Path(self.patient_folder)

    # -------------------------
    # PERUSPOLUT
    # -------------------------

    @property
    def original_dir(self) -> Path:
        """Alkuperäinen potilaskansio, josta DICOM/RS/RD/RP haetaan."""
        return BASE_DIR / self.original_dataset / self.patient_folder

    @property
    def modified_dir(self) -> Path:
        """Muokattujen tiedostojen kansio (tulokset)."""
        return BASE_DIR / self.processed_dataset / self.patient_folder

    # -------------------------
    # TIEDOSTOJEN POLUT
    # -------------------------

    @property
    def ct_dir(self) -> Path:
        return self.modified_dir / "ct"
    
    @property
    def org_ct_dir(self) -> Path:
        return self.modified_dir / "vanha ct"

    @property
    def mask_dir(self) -> Path:
        return self.modified_dir / "maski"
    
    @property
    def maskds_dir(self) -> Path:
        return self.modified_dir / "maskids"

    @property
    def dose_dir(self) -> Path:
        return self.modified_dir / "dose"
    
    @property
    def doseds_dir(self) -> Path:
        return self.modified_dir / "doseds"

    @property
    def struct_dir(self) -> Path:
        return self.modified_dir / "struct"

    @property
    def plan_dir(self) -> Path:
        return self.modified_dir / "plan"

    # -------------------------
    # NUMERON HAKU JÄRJESTYSTÄ VARTEN
    # -------------------------

    @property
    def number(self) -> int:
        m = re.search(r"(\d+)", str(self.patient_folder))
        return int(m.group(1)) if m else 0

@dataclass
class AllPatients:
    processed_dataset: Path
    original_dataset: Path

    def __post_init__(self):
        # Varmistetaan että Path
        self.processed_dataset = Path(self.processed_dataset)
        self.original_dataset = Path(self.original_dataset)
        self._patients = self._find_patients()   

    def _find_patients(self):
        dataset_path = BASE_DIR / self.processed_dataset
        patient_dirs = dataset_path.glob("Patient*")

        patients = [
            Patient(
                processed_dataset=self.processed_dataset,
                original_dataset=self.original_dataset,
                patient_folder=p.name
            )
            for p in patient_dirs
        ]

        if not patients:
            raise FileNotFoundError(f"No patients found in {dataset_path}")

        return patients

    @property
    def patients(self):
        return self._patients

    def sorted_by_number(self):
        return sorted(self._patients, key=lambda p: p.number)