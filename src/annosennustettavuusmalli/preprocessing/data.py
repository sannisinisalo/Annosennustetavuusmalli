import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Patient:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def dir(self) -> Path:
        return self.path

    @property
    def ct_path(self) -> Path:
        return self.dir / "vanha ct"

    @property
    def mask_destination(self) -> Path:
        return self.dir / "maski"

    @property
    def downsampled_destination(self) -> Path:
        """Lisää "ds" peruskansion nimeen, esim. "VN0" -> "VN0ds"."""
        return self.dir.parent.parent / (self.dir.parent.name + "ds") / self.dir.name

    @property
    def struct_path(self) -> Path:
        return self.dir / "struct"


# Järjestetään potilaat numerojärjestykseen, muuten tulisi aakkosjärjestyksessä
def get_patient_number(patient: Patient) -> int:
    """
    Get the numeric value from the patient folder name for sorting purposes.

    Parameters
    ----------
    patient : Patient
        The patient object.

    Returns
    -------
    int
        The numeric value extracted from the patient folder name, or 0 if none found.

    """
    patient_path = patient.dir
    # Eristetään numero nimestä
    m = re.search(r"(\d+)", patient_path.name)
    return int(m.group(1)) if m else 0


@dataclass
class AllPatients:
    data_root: Path

    def __post_init__(self):
        self._patients = self.get_patients()

    def get_patients(self) -> list[Patient]:
        patients = [Patient(path=p) for p in self.data_root.glob("Patient*")]
        if not patients:
            raise FileNotFoundError(
                f"Ei löytynyt potilaskansioita (Patient*) polusta: {self.data_root}"
            )
        return patients

    @property
    def patients(self) -> list[Patient]:
        return self._patients

    def sorted_by_number(self) -> list[Patient]:
        return sorted(self.patients, key=get_patient_number)
