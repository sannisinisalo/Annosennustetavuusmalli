from pathlib import Path

import pytest

from annosennustettavuusmalli.preprocessing.data import AllPatients, Patient


class TestPatient:
    """Tests for the Patient class."""

    def test_patient_name(self, tmp_path):
        """Test that patient name returns the path name."""
        patient_path = tmp_path / "Patient001"
        patient_path.mkdir()
        patient = Patient(path=patient_path)
        assert patient.name == "Patient001"

    def test_patient_dir(self, tmp_path):
        """Test that dir property returns the path."""
        patient_path = tmp_path / "Patient002"
        patient_path.mkdir()
        patient = Patient(path=patient_path)
        assert patient.dir == patient_path

    def test_get_patient_number_single_digit(self, tmp_path):
        """Test extracting a single digit patient number."""
        patient_path = tmp_path / "Patient5"
        patient_path.mkdir()
        patient = Patient(path=patient_path)
        assert patient.number == 5

    def test_get_patient_number_multiple_digits(self, tmp_path):
        """Test extracting multiple digit patient number."""
        patient_path = tmp_path / "Patient123"
        patient_path.mkdir()
        patient = Patient(path=patient_path)
        assert patient.number == 123

    def test_get_patient_number_no_digits(self, tmp_path):
        """Test that patient number returns 0 when no digits found."""
        patient_path = tmp_path / "PatientABC"
        patient_path.mkdir()
        patient = Patient(path=patient_path)
        assert patient.number == 0

    def test_get_patient_number_mixed_format(self, tmp_path):
        """Test extracting patient number from mixed format name."""
        patient_path = tmp_path / "VN042"
        patient_path.mkdir()
        patient = Patient(path=patient_path)
        assert patient.number == 42

    def test_get_patient_number_first_number_only(self, tmp_path):
        """Test that only the first number sequence is extracted."""
        patient_path = tmp_path / "Patient42Code99"
        patient_path.mkdir()
        patient = Patient(path=patient_path)
        assert patient.number == 42

    def test_ct_path(self, tmp_path):
        """Test that ct_path returns correct subdirectory."""
        patient_path = tmp_path / "Patient001"
        patient_path.mkdir()
        patient = Patient(path=patient_path)
        assert patient.ct_path == patient_path / "vanha ct"

    def test_mask_destination(self, tmp_path):
        """Test that mask_destination returns correct subdirectory."""
        patient_path = tmp_path / "Patient001"
        patient_path.mkdir()
        patient = Patient(path=patient_path)
        assert patient.mask_destination == patient_path / "maski"

    def test_struct_path(self, tmp_path):
        """Test that struct_path returns correct subdirectory."""
        patient_path = tmp_path / "Patient001"
        patient_path.mkdir()
        patient = Patient(path=patient_path)
        assert patient.struct_path == patient_path / "struct"

    def test_downsampled_destination(self, tmp_path):
        """Test that downsampled_destination adds 'ds' to parent directory name."""
        # Create structure: tmp_path / VN0 / Patient001
        vn_dir = tmp_path / "VN0"
        vn_dir.mkdir()
        patient_path = vn_dir / "Patient001"
        patient_path.mkdir()

        patient = Patient(path=patient_path)
        expected = tmp_path / "VN0ds" / "Patient001"
        assert patient.downsampled_destination == expected

    def test_downsampled_destination_complex_path(self, tmp_path):
        """Test downsampled_destination with more complex directory structure."""
        # Create structure: tmp_path / Data / VN42 / Patient123
        data_dir = tmp_path / "Data"
        data_dir.mkdir()
        vn_dir = data_dir / "VN42"
        vn_dir.mkdir()
        patient_path = vn_dir / "Patient123"
        patient_path.mkdir()

        patient = Patient(path=patient_path)
        expected = data_dir / "VN42ds" / "Patient123"
        assert patient.downsampled_destination == expected


class TestAllPatients:
    """Tests for the AllPatients class."""

    def test_all_patients_initialization(self, tmp_path):
        """Test that AllPatients initializes with patient directories."""
        # Create multiple patient directories
        (tmp_path / "Patient001").mkdir()
        (tmp_path / "Patient002").mkdir()
        (tmp_path / "Patient003").mkdir()

        all_patients = AllPatients(data_root=tmp_path)
        assert len(all_patients.patients) == 3

    def test_get_patients_finds_all_patient_dirs(self, tmp_path):
        """Test that get_patients finds all Patient* directories."""
        (tmp_path / "Patient100").mkdir()
        (tmp_path / "Patient050").mkdir()
        (tmp_path / "Patient001").mkdir()
        (tmp_path / "NotPatient").mkdir()  # Should not be included

        all_patients = AllPatients(data_root=tmp_path)
        assert len(all_patients.patients) == 3
        patient_names = {p.name for p in all_patients.patients}
        assert patient_names == {"Patient100", "Patient050", "Patient001"}

    def test_get_patients_empty_directory(self, tmp_path):
        """Test that get_patients raises FileNotFoundError when no patients found."""
        (tmp_path / "NotPatient").mkdir()
        (tmp_path / "SomeFile.txt").touch()

        with pytest.raises(FileNotFoundError) as exc_info:
            AllPatients(data_root=tmp_path)
        assert "Ei löytynyt potilaskansioita" in str(exc_info.value)
        assert str(tmp_path) in str(exc_info.value)

    def test_get_patients_nonexistent_directory(self):
        """Test that get_patients raises FileNotFoundError for nonexistent root."""
        nonexistent_path = Path("/nonexistent/path/that/does/not/exist")

        with pytest.raises(FileNotFoundError):
            AllPatients(data_root=nonexistent_path)

    def test_sorted_by_number_ascending_order(self, tmp_path):
        """Test that sorted_by_number returns patients in ascending numeric order."""
        (tmp_path / "Patient100").mkdir()
        (tmp_path / "Patient005").mkdir()
        (tmp_path / "Patient050").mkdir()
        (tmp_path / "Patient001").mkdir()

        all_patients = AllPatients(data_root=tmp_path)
        sorted_patients = all_patients.sorted_by_number()

        patient_numbers = [p.number for p in sorted_patients]
        assert patient_numbers == [1, 5, 50, 100]

    def test_sorted_by_number_mixed_format(self, tmp_path):
        """Test sorted_by_number with mixed patient naming formats."""
        (tmp_path / "Patient042").mkdir()
        (tmp_path / "Patient005").mkdir()
        (tmp_path / "Patient100").mkdir()
        (tmp_path / "Patient01").mkdir()

        all_patients = AllPatients(data_root=tmp_path)
        sorted_patients = all_patients.sorted_by_number()

        patient_numbers = [p.number for p in sorted_patients]
        assert patient_numbers == [1, 5, 42, 100]

    def test_sorted_by_number_with_non_numeric(self, tmp_path):
        """Test sorted_by_number includes patients with no numbers (placed at start)."""
        (tmp_path / "Patient050").mkdir()
        (tmp_path / "PatientABC").mkdir()
        (tmp_path / "Patient005").mkdir()

        all_patients = AllPatients(data_root=tmp_path)
        sorted_patients = all_patients.sorted_by_number()

        patient_numbers = [p.number for p in sorted_patients]
        # Patients with no numbers get 0, so they come first
        assert patient_numbers == [0, 5, 50]

    def test_patients_property_returns_list(self, tmp_path):
        """Test that patients property returns a list."""
        (tmp_path / "Patient001").mkdir()
        (tmp_path / "Patient002").mkdir()

        all_patients = AllPatients(data_root=tmp_path)
        assert isinstance(all_patients.patients, list)
        assert all(isinstance(p, Patient) for p in all_patients.patients)

    def test_single_patient(self, tmp_path):
        """Test AllPatients with a single patient directory."""
        (tmp_path / "Patient001").mkdir()

        all_patients = AllPatients(data_root=tmp_path)
        assert len(all_patients.patients) == 1
        assert all_patients.patients[0].number == 1

    def test_many_patients(self, tmp_path):
        """Test AllPatients with many patient directories."""
        for i in range(1, 101):
            (tmp_path / f"Patient{i:03d}").mkdir()

        all_patients = AllPatients(data_root=tmp_path)
        assert len(all_patients.patients) == 100
        sorted_patients = all_patients.sorted_by_number()
        patient_numbers = [p.number for p in sorted_patients]
        assert patient_numbers == list(range(1, 101))


class TestIntegration:
    """Integration tests combining Patient and AllPatients."""

    def test_patient_properties_in_all_patients(self, tmp_path):
        """Test that Patient properties work correctly within AllPatients."""
        vn_dir = tmp_path / "VN0"
        vn_dir.mkdir()
        patient_path = vn_dir / "Patient001"
        patient_path.mkdir()

        all_patients = AllPatients(data_root=vn_dir)
        patient = all_patients.patients[0]

        assert patient.name == "Patient001"
        assert patient.number == 1
        assert patient.ct_path == patient_path / "vanha ct"
        assert patient.mask_destination == patient_path / "maski"
        assert patient.struct_path == patient_path / "struct"
        assert patient.downsampled_destination == tmp_path / "VN0ds" / "Patient001"

    def test_workflow_with_multiple_patients(self, tmp_path):
        """Test typical workflow with multiple patients."""
        # Setup
        data_dir = tmp_path / "Data"
        data_dir.mkdir()
        vn_dir = data_dir / "VN0"
        vn_dir.mkdir()

        for i in [3, 1, 2]:
            (vn_dir / f"Patient{i:03d}").mkdir()

        # Create AllPatients and sort
        all_patients = AllPatients(data_root=vn_dir)
        sorted_patients = all_patients.sorted_by_number()

        # Verify ordering
        assert len(sorted_patients) == 3
        assert [p.number for p in sorted_patients] == [1, 2, 3]

        # Verify paths
        for patient in sorted_patients:
            assert patient.mask_destination.parent == patient.dir
