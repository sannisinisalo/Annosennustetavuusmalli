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
    def number(self) -> int:
        return self.get_patient_number()

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

    def get_patient_number(self) -> int:
        """
        Get the numeric value from the patient folder name for sorting purposes.

        Returns
        -------
        int
            The numeric value extracted from the patient folder name, or 0 if none found.

        """
        patient_path = self.dir
        # Eristetään numero nimestä
        m = re.search(r"(\d+)", patient_path.name)
        return int(m.group(1)) if m else 0


@dataclass
class AllPatients:
    data_root: Path

    def __post_init__(self):
        self._patients = self.get_patients()

    def get_patients(self) -> list[Patient]:
        """Etsi kaikki potilaskansiot (Patient*) data_root-kansiosta ja luo niistä Patient-objektit.

        Returns
        -------
        list[Patient]
            Lista data_root-kansion Patient*-kansioista luotuja Patient-objekteja.

        Raises
        ------
        FileNotFoundError
            Jos data_root-kansiossa ei löydy yhtään Patient*-kansiota.

        """
        patients = [Patient(path=p) for p in self.data_root.glob("Patient*")]
        if not patients:
            raise FileNotFoundError(
                f"Ei löytynyt potilaskansioita (Patient*) polusta: {self.data_root}"
            )
        return patients

    @property
    def patients(self) -> list[Patient]:
        """Palauttaa listan Patient-objekteja, jotka on luotu data_root-kansion Patient*-kansioista."""
        return self._patients

    def sorted_by_number(self) -> list[Patient]:
        """Järjestää potilaat numerojärjestykseen, muuten tulisi aakkosjärjestyksessä."""
        patient_numbers = {p: p.number for p in self.patients}
        return sorted(self.patients, key=lambda p: patient_numbers[p])
