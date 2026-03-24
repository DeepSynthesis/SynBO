"""
Test LLM-based constraint generation functionality.

This test file demonstrates how to use the get_constrains method with LLM
to generate constraints for reaction optimization.
"""

import unittest
from pathlib import Path
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import json

from synbo import ReactionOptimizer
from synbo.utils.load_data import load_desc_dict


class TestConstrainWithLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reagent_types = ["base", "ligand", "solvent", "concentration", "temperature"]
        cls.opt_direct_info = [{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "min", "opt_range": [0, 0.5]}]
        cls.save_dir = "test_results"

        # 预加载数据
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

    def _setup_optimizer(self):
        """Setup basic optimizer configuration."""
        rxn_opt = ReactionOptimizer(opt_metrics=["yield", "cost"], opt_metric_settings=self.opt_direct_info, opt_type="auto", quiet=True)
        rxn_opt.load_rxn_space(condition_dict=self.condition_dict)
        rxn_opt.load_desc(desc_dict=self.desc_dict)
        rxn_opt.load_prev_rxn(pd.read_csv(Path(__file__).parent / "testfile/start_file.csv", index_col=False))
        return rxn_opt

    @patch("rxnopt.analysis.llm_analyzer.OpenAI")
    def test_get_constrains_with_mock_llm(self, mock_openai):
        """Test get_constrains method with mocked LLM response."""
        print("\nTest 1: Mock LLM constraint generation")

        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "base": self.condition_dict["base"][: int(len(self.condition_dict["base"]) * 0.7)],
                "ligand": self.condition_dict["ligand"][: int(len(self.condition_dict["ligand"]) * 0.7)],
                "solvent": self.condition_dict["solvent"][: int(len(self.condition_dict["solvent"]) * 0.7)],
            }
        )

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Setup optimizer
        rxn_opt = self._setup_optimizer()

        # Get constraints
        constraints = rxn_opt.get_constrains(method="llm", reduce_ratio=0.3, api_key="test-api-key", model="gpt-4", temperature=0.7)

        # Verify constraints were generated
        self.assertIsNotNone(constraints, "Constraints should not be None")
        self.assertIsInstance(constraints, dict, "Constraints should be a dictionary")

        # Verify constraint format
        for cond_type, allowed_values in constraints.items():
            self.assertIn(cond_type, self.condition_dict, f"Condition type {cond_type} not in condition_dict")
            self.assertIsInstance(allowed_values, list, f"Allowed values for {cond_type} should be a list")
            self.assertTrue(len(allowed_values) > 0, f"Should have at least one allowed value for {cond_type}")
            self.assertTrue(len(allowed_values) < len(self.condition_dict[cond_type]), f"Should remove some values for {cond_type}")

        # Verify OpenAI client was called with correct parameters
        mock_openai.assert_called_once_with(api_key="test-api-key", base_url=None)
        mock_client.chat.completions.create.assert_called_once()

        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]["model"], "gpt-4")
        self.assertEqual(call_args[1]["temperature"], 0.7)

        print(f"✓ Successfully generated constraints for {len(constraints)} condition types")

    @patch("rxnopt.analysis.llm_analyzer.OpenAI")
    def test_optimization_with_llm_constraints(self, mock_openai):
        """Test optimization workflow with LLM-generated constraints."""
        print("\nTest 2: Optimization with LLM constraints")

        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "base": self.condition_dict["base"][: int(len(self.condition_dict["base"]) * 0.7)],
                "ligand": self.condition_dict["ligand"][: int(len(self.condition_dict["ligand"]) * 0.7)],
            }
        )

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Setup optimizer
        rxn_opt = self._setup_optimizer()

        # Get constraints
        constraints = rxn_opt.get_constrains(method="llm", reduce_ratio=0.3, api_key="test-api-key")

        # Run optimization with constraints
        rxn_opt.optimize(
            batch_size=2,
            desc_normalize="minmax",
            refine_desc="auto_select",
            optimize_method="default_BO",
            temperature=0.1,
            constraints=constraints,
            surrogate_model="RF",
        )

        # Verify optimization completed
        self.assertIsNotNone(rxn_opt.selected_conditions)
        self.assertEqual(len(rxn_opt.selected_conditions), 2)

        # Verify all selected conditions respect constraints
        for condition in rxn_opt.selected_conditions:
            for i, cond_type in enumerate(self.reagent_types):
                if cond_type in constraints:
                    value = condition[i]
                    self.assertNotIn(str(value), constraints[cond_type], f"Selected {cond_type} value {value} not in constraints")

        print(f"✓ Successfully optimized with constraints")

    @patch("rxnopt.analysis.llm_analyzer.OpenAI")
    def test_different_reduce_ratios(self, mock_openai):
        """Test constraint generation with different reduce ratios."""
        print("\nTest 3: Different reduce ratios")

        # Test different reduce ratios
        test_ratios = [0.2, 0.3, 0.5]

        for ratio in test_ratios:
            # Setup mock response
            mock_response = Mock()
            mock_response.choices = [Mock()]

            # Calculate expected number of values to keep
            expected_kept = {cond_type: int(len(values) * (1 - ratio)) for cond_type, values in self.condition_dict.items()}

            mock_response.choices[0].message.content = json.dumps(
                {
                    cond_type: values[: expected_kept[cond_type]]
                    for cond_type, values in self.condition_dict.items()
                    if expected_kept[cond_type] > 0
                }
            )

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            # Setup optimizer
            rxn_opt = self._setup_optimizer()

            # Get constraints
            constraints = rxn_opt.get_constrains(method="llm", reduce_ratio=ratio, api_key="test-api-key")

            # # Verify constraints match expected ratio
            # for cond_type, allowed_values in constraints.items():
            #     total = len(self.condition_dict[cond_type])
            #     kept = len(allowed_values)
            #     actual_ratio = 1 - (kept / total)
            #     self.assertAlmostEqual(
            #         actual_ratio, ratio, delta=0.1,
            #         msg=f"Reduce ratio mismatch for {cond_type}"
            #     )

            print(f"✓ Ratio {ratio}: {len(constraints)} constraints generated")

    @patch("rxnopt.analysis.llm_analyzer.OpenAI")
    def test_custom_base_url(self, mock_openai):
        """Test LLM analysis with custom base URL."""
        print("\nTest 4: Custom base URL")

        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {"base": self.condition_dict["base"][: int(len(self.condition_dict["base"]) * 0.7)]}
        )

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Setup optimizer
        rxn_opt = self._setup_optimizer()

        # Get constraints with custom base URL
        custom_url = "https://custom-endpoint.openai.azure.com/"
        constraints = rxn_opt.get_constrains(method="llm", reduce_ratio=0.3, api_key="test-api-key", base_url=custom_url, model="gpt-4")

        # Verify custom URL was used
        mock_openai.assert_called_once_with(api_key="test-api-key", base_url=custom_url)

        print(f"✓ Custom base URL used: {custom_url}")

    @patch("rxnopt.analysis.llm_analyzer.OpenAI")
    def test_invalid_json_response(self, mock_openai):
        """Test handling of invalid JSON response from LLM."""
        print("\nTest 5: Invalid JSON response handling")

        # Setup mock response with invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is not valid JSON"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Setup optimizer
        rxn_opt = self._setup_optimizer()

        # Get constraints - should return None or handle gracefully
        constraints = rxn_opt.get_constrains(method="llm", reduce_ratio=0.3, api_key="test-api-key")

        # Should handle invalid JSON gracefully
        self.assertIsNone(constraints, "Should return None for invalid JSON")

        print(f"✓ Invalid JSON handled gracefully")

    @patch("rxnopt.analysis.llm_analyzer.OpenAI")
    def test_missing_api_key_error(self, mock_openai):
        """Test that missing API key raises appropriate error."""
        print("\nTest 6: Missing API key error")

        # Setup optimizer
        rxn_opt = self._setup_optimizer()

        # Should raise ValueError without API key
        with self.assertRaises(ValueError) as context:
            rxn_opt.get_constrains(method="llm", reduce_ratio=0.3)

        self.assertIn("api_key", str(context.exception))

        print(f"✓ API key validation works correctly")


if __name__ == "__main__":
    unittest.main(verbosity=2)
