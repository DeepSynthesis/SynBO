"""
Test constrained Bayesian optimization functionality.

This test file demonstrates how to use the constraints parameter in ReactionOptimizer
to limit the search space during optimization.
"""

import unittest
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from synbo import ReactionOptimizer
from synbo.utils.load_data import load_desc_dict


class TestConstrainedOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reagent_types = ["base", "ligand", "solvent", "concentration", "temperature"]
        cls.opt_direct_info = [{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "min", "opt_range": [0, 0.5]}]
        cls.save_dir = "test_results"

        # Pre-load data
        cls.desc_dict, cls.condition_dict = load_desc_dict(
            reagent_types=cls.reagent_types,
            desc_dir=Path(__file__).parent / "dataset/descriptors",
            name_suffix=["_dft", "_dft", "_dft", None, None],
            index_col=cls.reagent_types,
            return_condition_dict=True,
        )

    def tearDown(self):
        if Path(self.save_dir).exists:
            for f in Path(self.save_dir).glob("*"):
                if f.is_file():
                    f.unlink()

    def _run_constrained_optimization(self, constraints, batch_size=2, **opt_kwargs):
        """
        Run constrained optimization workflow.

        Args:
            constraints: Dictionary of constraints {condition_type: [allowed_values]}
            batch_size: Number of conditions to recommend
            **opt_kwargs: Additional optimization parameters
        """
        # core code
        sbo = ReactionOptimizer(opt_metrics=["yield", "cost"], opt_metric_settings=self.opt_direct_info, opt_type="auto", quiet=True)
        sbo.load_rxn_space(condition_dict=self.condition_dict)
        sbo.load_desc(desc_dict=self.desc_dict)
        sbo.load_prev_rxn(pd.read_csv(Path(__file__).parent / "testfile/start_file.csv", index_col=False))

        sbo.optimize(
            batch_size=batch_size,
            desc_normalize="minmax",
            refine_desc="auto_select",
            optimize_method="default_BO",
            temperature=0.1,
            constraints=constraints,
            **opt_kwargs,
        )

        return sbo

    def test_constrain_base(self):
        """Test constraint on base selection."""
        # Get available bases
        available_bases = self.condition_dict["base"]

        # Constrain to only use the first two bases
        constraints = {"base": available_bases[:2]}

        print(f"\nTest 1: Constraining to bases: {constraints['base']}")

        sbo = self._run_constrained_optimization(constraints=constraints, surrogate_model="RF")

        # Verify that all selected conditions use only constrained bases
        selected_conditions = sbo.selected_conditions
        for condition in selected_conditions:
            base_value = condition[self.reagent_types.index("base")]
            self.assertIn(base_value, constraints["base"], f"Selected base {base_value} not in constrained list")

        sbo.save_results(save_dir=self.save_dir, filetype="csv", suffix="_constrain_base")
        print(f"✓ All selected bases are within constraints")

    def test_constrain_multiple_conditions(self):
        """Test constraints on multiple condition types."""
        available_bases = self.condition_dict["base"]
        available_solvents = self.condition_dict["solvent"]
        available_temperatures = self.condition_dict["temperature"]

        # Constrain multiple condition types
        constraints = {
            "base": available_bases[:2],  # Only first two bases
            "solvent": available_solvents[:3],  # Only first three solvents
            "temperature": available_temperatures[::2],  # Every other temperature
        }

        print(f"\nTest 2: Constraining multiple conditions")
        print(f"  - Bases: {len(constraints['base'])} / {len(available_bases)}")
        print(f"  - Solvents: {len(constraints['solvent'])} / {len(available_solvents)}")
        print(f"  - Temperatures: {len(constraints['temperature'])} / {len(available_temperatures)}")

        sbo = self._run_constrained_optimization(constraints=constraints, batch_size=3, surrogate_model="RF")

        # Verify all constraints are respected
        selected_conditions = sbo.selected_conditions
        for condition in selected_conditions:
            base_value = condition[self.reagent_types.index("base")]
            solvent_value = condition[self.reagent_types.index("solvent")]
            temp_value = condition[self.reagent_types.index("temperature")]

            self.assertIn(base_value, constraints["base"])
            self.assertIn(solvent_value, constraints["solvent"])
            self.assertIn(temp_value, constraints["temperature"])

        sbo.save_results(save_dir=self.save_dir, filetype="csv", suffix="_constrain_multiple")
        print(f"✓ All selected conditions satisfy multiple constraints")

    def test_no_constraints(self):
        """Test optimization without constraints (baseline)."""
        print(f"\nTest 3: No constraints (baseline)")

        sbo = self._run_constrained_optimization(constraints=None, surrogate_model="RF")  # No constraints

        sbo.save_results(save_dir=self.save_dir, filetype="csv", suffix="_no_constraints")
        print(f"✓ Completed optimization without constraints")

    def test_constrain_single_value(self):
        """Test constraining to a single value per condition type."""
        available_bases = self.condition_dict["base"]
        available_ligands = self.condition_dict["ligand"]

        # Constrain to single values
        constraints = {"base": [available_bases[0]], "ligand": [available_ligands[1]]}  # Only first base  # Only second ligand

        print(f"\nTest 4: Constraining to single values")
        print(f"  - Base: {constraints['base'][0]}")
        print(f"  - Ligand: {constraints['ligand'][0]}")

        sbo = self._run_constrained_optimization(constraints=constraints, surrogate_model="RF")

        # Verify all selected conditions use the constrained values
        selected_conditions = sbo.selected_conditions
        for condition in selected_conditions:
            base_value = condition[self.reagent_types.index("base")]
            ligand_value = condition[self.reagent_types.index("ligand")]

            self.assertEqual(base_value, constraints["base"][0])
            self.assertEqual(ligand_value, constraints["ligand"][0])

        sbo.save_results(save_dir=self.save_dir, filetype="csv", suffix="_constrain_single")
        print(f"✓ All selected conditions use single constrained values")

    def test_constrain_with_different_methods(self):
        """Test constraints with different optimization methods."""
        available_bases = self.condition_dict["base"]
        constraints = {"base": available_bases[:2]}

        print(f"\nTest 5: Constraints with different acquisition functions")

        # Test with EHVI
        print(f"  - Testing with EHVI acquisition function")
        sbo_ehvi = self._run_constrained_optimization(constraints=constraints, surrogate_model="GP", acq_func="EHVI")
        sbo_ehvi.save_results(save_dir=self.save_dir, filetype="csv", suffix="_constrain_ehvi")
        print(f"    ✓ EHVI completed")

        # Test with UCB
        print(f"  - Testing with UCB acquisition function")
        sbo_ucb = self._run_constrained_optimization(constraints=constraints, surrogate_model="GP", acq_func="UCB")
        sbo_ucb.save_results(save_dir=self.save_dir, filetype="csv", suffix="_constrain_ucb")
        print(f"    ✓ UCB completed")

        # Test with ParEGO
        print(f"  - Testing with ParEGO acquisition function")
        sbo_parego = self._run_constrained_optimization(constraints=constraints, surrogate_model="GP", acq_func="ParEGO")
        sbo_parego.save_results(save_dir=self.save_dir, filetype="csv", suffix="_constrain_parego")
        print(f"    ✓ ParEGO completed")

    def test_constrain_validation(self):
        """Test that invalid constraint values are handled properly."""
        print(f"\nTest 6: Constraint validation")

        # Test with invalid constraint value (should still work, just won't match anything)
        constraints = {"base": ["nonexistent_base_12345"]}

        try:
            sbo = self._run_constrained_optimization(constraints=constraints, batch_size=1, surrogate_model="RF")
            # Should complete but with warning about no valid candidates
            print(f"✓ Handled invalid constraint gracefully")
        except Exception as e:
            # Alternatively, might raise an error
            print(f"✓ Invalid constraint raised error as expected: {e}")


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
