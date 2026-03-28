"""
Simple example demonstrating LLM-based constraint generation and optimization.

This example shows how to use the get_constrains method with LLM
to generate constraints for reaction optimization, followed by optimization.
"""

from pathlib import Path
import pandas as pd
from unittest.mock import Mock, patch
import json

from synbo import ReactionOptimizer
from synbo.utils.load_data import load_desc_dict


def setup_optimizer():
    """Setup basic optimizer configuration."""
    reagent_types = ["base", "ligand", "solvent", "concentration", "temperature"]
    opt_direct_info = [{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "min", "opt_range": [0, 0.5]}]

    # Load data
    desc_dict, condition_dict = load_desc_dict(
        reagent_types=reagent_types,
        desc_dir=Path(__file__).parent / "dataset/descriptors",
        name_suffix=["_dft", "_dft", "_dft", None, None],
        index_col=reagent_types,
        return_condition_dict=True,
    )

    # Create optimizer
    sbo = ReactionOptimizer(opt_metrics=["yield", "cost"], opt_metric_settings=opt_direct_info, opt_type="auto", quiet=True)
    sbo.load_rxn_space(condition_dict=condition_dict)
    sbo.load_desc(desc_dict=desc_dict)
    sbo.load_prev_rxn(pd.read_csv(Path(__file__).parent / "testfile/start_file.csv", index_col=False))

    return sbo, condition_dict, reagent_types


def test_optimization_with_llm_constraints():
    """Test optimization workflow with LLM-generated constraints."""
    print("\n=== Testing LLM Constraint Generation and Optimization ===")

    # Setup optimizer
    sbo, condition_dict, reagent_types = setup_optimizer()

    # Setup mock response manually
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps(
        {
            "base": condition_dict["base"][: int(len(condition_dict["base"]) * 0.7)],
            "ligand": condition_dict["ligand"][: int(len(condition_dict["ligand"]) * 0.7)],
        }
    )

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response

    # Patch OpenAI manually
    with patch("synbo.analysis.llm_analyzer.OpenAI", return_value=mock_client):
        # Step 1: Get constraints from LLM
        print("\nStep 1: Generating constraints using LLM...")
        constraints = sbo.get_constrains(method="llm", reduce_ratio=0.3, api_key="test-api-key", model="gpt-4", temperature=0.7)

        print(f"✓ Successfully generated constraints for {len(constraints)} condition types:")
        for cond_type, allowed_values in constraints.items():
            total = len(condition_dict[cond_type])
            kept = len(allowed_values)
            reduced = total - kept
            print(f"  - {cond_type}: {kept}/{total} values (reduced by {reduced})")

        # Step 2: Run optimization with constraints
        print("\nStep 2: Running optimization with constraints...")
        sbo.optimize(
            batch_size=2,
            desc_normalize="minmax",
            refine_desc="auto_select",
            optimize_method="default_BO",
            temperature=0.1,
            constraints=constraints,
            surrogate_model="RF",
        )

        print(f"✓ Optimization completed")
        print(f"  - Selected {len(sbo.selected_conditions)} conditions")
        print(f"  - Recommendation type: {sbo.recommend_type}")

        # Display selected conditions
        print("\nSelected conditions:")
        for i, condition in enumerate(sbo.selected_conditions, 1):
            print(f"\n  Condition {i}:")
            for j, cond_type in enumerate(reagent_types):
                value = condition[j]
                if cond_type in constraints:
                    is_allowed = str(value) in constraints[cond_type]
                    status = "✓" if is_allowed else "✗"
                    print(f"    {status} {cond_type}: {value}")
                else:
                    print(f"      {cond_type}: {value}")

        print("\n=== Test Completed Successfully ===")
        return sbo, constraints


if __name__ == "__main__":
    test_optimization_with_llm_constraints()
