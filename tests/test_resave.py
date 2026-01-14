import unittest
from pathlib import Path

from rxnopt.utils.export_data import resave_output_results


class TestResaveOutput(unittest.TestCase):
    """Test cases for resave_output_results function."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        cls.input_file = Path(__file__).parent / "testfile/start_file.csv"
        cls.output_dir = Path(__file__).parent / "test_output"
        cls.output_dir.mkdir(exist_ok=True)

        # Define condition columns and metrics columns
        cls.condition_columns = ["base", "ligand", "solvent", "concentration", "temperature"]
        cls.metrics_columns = ["yield", "cost"]

    def tearDown(self):
        """Clean up test output files after each test."""
        # Clean up output files if they exist
        if self.output_dir.exists():
            for f in self.output_dir.glob("*"):
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass

    def test_csv_to_csv_conversion(self):
        """Test conversion from CSV to CSV format."""
        output_file = self.output_dir / "output.csv"

        resave_output_results(
            str(self.input_file), str(output_file), condition_columns=self.condition_columns, metrics_columns=self.metrics_columns
        )

        # Verify the output file was created
        self.assertTrue(output_file.exists(), "Output CSV file should be created")

    def test_csv_to_excel_conversion(self):
        """Test conversion from CSV to Excel format."""
        output_file = self.output_dir / "output.xlsx"

        resave_output_results(
            str(self.input_file), str(output_file), condition_columns=self.condition_columns, metrics_columns=self.metrics_columns
        )

        # Verify the output file was created
        self.assertTrue(output_file.exists(), "Output Excel file should be created")

    def test_csv_to_json_conversion(self):
        """Test conversion from CSV to JSON format."""
        output_file = self.output_dir / "output.json"

        resave_output_results(
            str(self.input_file), str(output_file), condition_columns=self.condition_columns, metrics_columns=self.metrics_columns
        )

        # Verify the output file was created
        self.assertTrue(output_file.exists(), "Output JSON file should be created")


if __name__ == "__main__":
    unittest.main()
