import unittest
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from synbo import ReactionOptimizer
from synbo.utils.load_data import load_desc_dict


class TestReactionOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reagent_types = ["base", "ligand", "solvent", "concentration", "temperature"]
        cls.opt_direct_info = [{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "min", "opt_range": [0, 0.5]}]

        # Set save directory to testfile under current script
        cls.save_dir = Path(__file__).parent / "testfile"

        # Ensure directory exists
        if not cls.save_dir.exists():
            cls.save_dir.mkdir(parents=True)

        # Pre-load data
        cls.desc_dict, cls.condition_dict = load_desc_dict(
            reagent_types=cls.reagent_types,
            desc_dir=Path(__file__).parent / "dataset/descriptors",
            name_suffix=["_dft", "_dft", "_dft", None, None],
            index_col=cls.reagent_types,
            return_condition_dict=True,
        )

    def _run_optimization_workflow(self, opt_method, normalize, refine, filetype, **opt_kwargs):
        sbo = ReactionOptimizer(opt_metrics=["yield", "cost"], opt_metric_settings=self.opt_direct_info, opt_type="auto", quiet=True)
        sbo.load_rxn_space(condition_dict=self.condition_dict)
        sbo.load_desc(desc_dict=self.desc_dict)

        start_file_path = self.save_dir / "start_file.csv"
        sbo.load_prev_rxn(pd.read_csv(start_file_path, index_col=False))
        sbo.optimize(batch_size=3, desc_normalize=normalize, refine_desc=refine, optimize_method=opt_method, **opt_kwargs)
        sbo.save_results(save_dir=str(self.save_dir), filetype=filetype)

    def test_generate_and_rename_excel(self):
        # Parameter settings
        method = "random_select"
        norm = "minmax"
        refine = "none"
        ftype = "xlsx"
        target_name = "output_excel.xlsx"  # Final desired filename
        opt_kwargs = {"surrogate_model": "RF", "acq_func": "EHVI"}

        files_before = set(self.save_dir.glob("*.xlsx"))

        print(f"\n[Status] Starting optimization to generate {ftype}...")

        try:
            self._run_optimization_workflow(method, norm, refine, ftype, **opt_kwargs)
        except Exception as e:
            self.fail(f"Optimization failed: {e}")

        files_after = set(self.save_dir.glob("*.xlsx"))
        new_files = files_after - files_before

        if not new_files:
            self.fail("No new .xlsx file was detected after optimization!")

        generated_file = list(new_files)[0]
        target_path = self.save_dir / target_name

        print(f"[Status] Detected new file: {generated_file.name}")

        try:
            generated_file.replace(target_path)
            print(f"✅ Success! File renamed to: {target_path}")
            print(f"   You can now open '{target_name}' to check the format.")
        except Exception as e:
            self.fail(f"Failed to rename file: {e}")

    def test_excel_output_with_images(self):
        """Test Excel output with image output"""
        method = "random_select"
        norm = "minmax"
        refine = "none"
        ftype = "xlsx"
        target_name = "output_excel_with_images.xlsx"

        # Set image output parameters
        figure_output = ["base", "ligand"]  # Reagent types to output images for
        figure_path = self.save_dir / "pics"  # Image path

        files_before = set(self.save_dir.glob("*.xlsx"))

        print(f"\n[Status] Starting optimization to generate {ftype} with image integration...")

        try:
            # Run optimization workflow
            sbo = ReactionOptimizer(
                opt_metrics=["yield", "cost"], opt_metric_settings=self.opt_direct_info, opt_type="auto", quiet=True, random_seed=100
            )
            sbo.load_rxn_space(condition_dict=self.condition_dict)
            sbo.load_desc(desc_dict=self.desc_dict)

            start_file_path = self.save_dir / "start_file.csv"
            sbo.load_prev_rxn(pd.read_csv(start_file_path, index_col=False))
            sbo.optimize(batch_size=3, desc_normalize=norm, refine_desc=refine, optimize_method=method)

            # Include image output when saving
            sbo.save_results(save_dir=str(self.save_dir), filetype=ftype, figure_output=figure_output, figure_path=str(figure_path))

        except Exception as e:
            self.fail(f"Optimization with image integration failed: {e}")

        files_after = set(self.save_dir.glob("*.xlsx"))
        new_files = files_after - files_before

        if not new_files:
            self.fail("No new .xlsx file was detected after optimization with image integration!")

        generated_file = list(new_files)[0]
        target_path = self.save_dir / target_name

        print(f"[Status] Detected new file with images: {generated_file.name}")

        try:
            generated_file.replace(target_path)
            print(f"✅ Success! Excel file with images created: {target_path}")
            print(f"   File contains integrated images from: {figure_path}")
            print(f"   Figure types included: {figure_output}")

            # Verify file exists and is not empty
            if target_path.exists() and target_path.stat().st_size > 0:
                print(f"   File size: {target_path.stat().st_size} bytes")
            else:
                self.fail("Generated Excel file is empty or doesn't exist")

        except Exception as e:
            self.fail(f"Failed to rename file with images: {e}")

    def test_excel_output_with_images_transpose(self):
        """Test Excel output with image output"""
        method = "random_select"
        norm = "minmax"
        refine = "none"
        ftype = "xlsx"
        target_name = "output_excel_with_images_transpose.xlsx"

        # Set image output parameters
        figure_output = ["base", "ligand"]  # Reagent types to output images for
        figure_path = self.save_dir / "pics"  # Image path

        files_before = set(self.save_dir.glob("*.xlsx"))

        print(f"\n[Status] Starting optimization to generate {ftype} with image integration...")

        try:
            # Run optimization workflow
            sbo = ReactionOptimizer(
                opt_metrics=["yield", "cost"], opt_metric_settings=self.opt_direct_info, opt_type="auto", quiet=True, random_seed=100
            )
            sbo.load_rxn_space(condition_dict=self.condition_dict)
            sbo.load_desc(desc_dict=self.desc_dict)

            start_file_path = self.save_dir / "start_file.csv"
            sbo.load_prev_rxn(pd.read_csv(start_file_path, index_col=False))
            sbo.optimize(batch_size=5, desc_normalize=norm, refine_desc=refine, optimize_method=method)

            # Include image output when saving
            sbo.save_results(
                save_dir=str(self.save_dir), filetype=ftype, figure_output=figure_output, figure_path=str(figure_path), transpose=True
            )

        except Exception as e:
            self.fail(f"Optimization with image integration failed: {e}")

        files_after = set(self.save_dir.glob("*.xlsx"))
        new_files = files_after - files_before

        if not new_files:
            self.fail("No new .xlsx file was detected after optimization with image integration!")

        generated_file = list(new_files)[0]
        target_path = self.save_dir / target_name

        print(f"[Status] Detected new file with images: {generated_file.name}")

        try:
            generated_file.replace(target_path)
            print(f"✅ Success! Excel file with images created: {target_path}")
            print(f"   File contains integrated images from: {figure_path}")
            print(f"   Figure types included: {figure_output}")

            # Verify file exists and is not empty
            if target_path.exists() and target_path.stat().st_size > 0:
                print(f"   File size: {target_path.stat().st_size} bytes")
            else:
                self.fail("Generated Excel file is empty or doesn't exist")

        except Exception as e:
            self.fail(f"Failed to rename file with images: {e}")


if __name__ == "__main__":
    unittest.main()
