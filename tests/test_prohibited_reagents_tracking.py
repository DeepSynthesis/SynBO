"""Test script for prohibited reagents tracking functionality.

This script demonstrates the three main functions:
1. Saving prohibited reagents after LLM recommendation
2. Loading prohibited reagents before next LLM recommendation
3. Loading prohibited reagents during optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from synbo.utils.constraints_io import (
    load_prohibited_reagents,
    save_prohibited_reagents,
    merge_constraints
)


def test_save_and_load_prohibited_reagents():
    """Test Function 1 and 2: Saving and loading prohibited reagents."""
    print("\n" + "="*80)
    print("Test 1: Save and Load Prohibited Reagents")
    print("="*80)

    # Create a temporary directory for testing
    test_dir = Path(tempfile.mkdtemp())

    try:
        # Simulate LLM recommending prohibited reagents
        new_prohibited = {
            "base": ["base_A", "base_B"],
            "solvent": ["solvent_X"]
        }

        print(f"\nSimulated LLM recommendation:")
        for cond_type, values in new_prohibited.items():
            print(f"  {cond_type}: {values}")

        # Function 1: Save prohibited reagents
        print(f"\n[Function 1] Saving prohibited reagents to: {test_dir}")
        save_prohibited_reagents(test_dir, new_prohibited)

        # Function 2: Load prohibited reagents
        print(f"\n[Function 2] Loading prohibited reagents from: {test_dir}")
        loaded_prohibited = load_prohibited_reagents(test_dir)

        print(f"\nLoaded prohibited reagents:")
        for cond_type, values in loaded_prohibited.items():
            print(f"  {cond_type}: {values}")

        # Verify the data
        assert loaded_prohibited == new_prohibited, "Loaded data doesn't match saved data!"
        print("\n✅ Test 1 PASSED: Save and load functions work correctly")

        # Test merging with new recommendations
        print("\n" + "-"*80)
        print("Testing merge functionality...")
        new_recommendations = {
            "base": ["base_C"],  # New base
            "additive": ["additive_A"]  # New condition type
        }

        print(f"\nNew LLM recommendations:")
        for cond_type, values in new_recommendations.items():
            print(f"  {cond_type}: {values}")

        merged = merge_constraints(loaded_prohibited, new_recommendations)

        print(f"\nMerged constraints:")
        for cond_type, values in merged.items():
            print(f"  {cond_type}: {values}")

        # Verify merge
        assert "base" in merged and len(merged["base"]) == 3, "Base merge failed!"
        assert "solvent" in merged and len(merged["solvent"]) == 1, "Solvent merge failed!"
        assert "additive" in merged and len(merged["additive"]) == 1, "Additive merge failed!"
        print("\n✅ Merge test PASSED")

        # Save merged results
        save_prohibited_reagents(test_dir, merged, existing_prohibited=loaded_prohibited)
        print(f"\n✅ Merged constraints saved successfully")

    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_load_during_optimization():
    """Test Function 3: Loading prohibited reagents during optimization."""
    print("\n" + "="*80)
    print("Test 2: Load Prohibited Reagents During Optimization")
    print("="*80)

    # Create a temporary directory for testing
    test_dir = Path(tempfile.mkdtemp())

    try:
        # Pre-populate with some prohibited reagents
        existing_prohibited = {
            "base": ["base_old1", "base_old2"],
            "solvent": ["solvent_old"]
        }

        save_prohibited_reagents(test_dir, existing_prohibited)
        print(f"\nPre-saved prohibited reagents:")
        for cond_type, values in existing_prohibited.items():
            print(f"  {cond_type}: {values}")

        # Function 3: Simulate loading during optimization
        print(f"\n[Function 3] Loading prohibited reagents during optimization...")
        file_prohibited = load_prohibited_reagents(test_dir)

        if file_prohibited:
            print(f"  ✓ Loaded {len(file_prohibited)} condition types with prohibited reagents")

        # Simulate merging with user-provided constraints
        user_constraints = {
            "temperature": ["high_temp"]
        }

        print(f"\nUser-provided constraints:")
        for cond_type, values in user_constraints.items():
            print(f"  {cond_type}: {values}")

        # Merge file constraints with user constraints
        final_constraints = merge_constraints(user_constraints, file_prohibited)

        print(f"\nFinal merged constraints for optimization:")
        for cond_type, values in final_constraints.items():
            print(f"  {cond_type}: {values}")

        # Verify all constraints are present
        assert "base" in final_constraints and len(final_constraints["base"]) == 2
        assert "solvent" in final_constraints and len(final_constraints["solvent"]) == 1
        assert "temperature" in final_constraints and len(final_constraints["temperature"]) == 1

        print("\n✅ Test 2 PASSED: Load during optimization works correctly")

    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_nonexistent_file():
    """Test loading from nonexistent file."""
    print("\n" + "="*80)
    print("Test 3: Load from Nonexistent File")
    print("="*80)

    test_dir = Path("/tmp/nonexistent_directory_xyz123")

    print(f"\nAttempting to load from: {test_dir}")
    result = load_prohibited_reagents(test_dir)

    if result is None:
        print("✓ Correctly returned None for nonexistent file")
    else:
        print("✗ Should have returned None!")
        assert False, "Should return None for nonexistent file"

    print("\n✅ Test 3 PASSED: Handles nonexistent file gracefully")


def test_empty_constraints():
    """Test with empty/None constraints."""
    print("\n" + "="*80)
    print("Test 4: Empty/None Constraints")
    print("="*80)

    # Test merging None with None
    result = merge_constraints(None, None)
    assert result is None, "Should return None when both inputs are None"
    print("✓ merge_constraints(None, None) = None")

    # Test merging dict with None
    constraints = {"base": ["base_A"]}
    result = merge_constraints(constraints, None)
    assert result == constraints, "Should return dict when one input is None"
    print("✓ merge_constraints(dict, None) = dict")

    # Test merging None with dict
    result = merge_constraints(None, constraints)
    assert result == constraints, "Should return dict when one input is None"
    print("✓ merge_constraints(None, dict) = dict")

    print("\n✅ Test 4 PASSED: Handles empty/None constraints correctly")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PROHIBITED REAGENTS TRACKING - TEST SUITE")
    print("="*80)

    try:
        test_save_and_load_prohibited_reagents()
        test_load_during_optimization()
        test_nonexistent_file()
        test_empty_constraints()

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✅")
        print("="*80)
        print("\nThe prohibited reagents tracking system is working correctly:")
        print("  1. ✓ Saves prohibited reagents after LLM recommendation")
        print("  2. ✓ Loads prohibited reagents before next LLM recommendation")
        print("  3. ✓ Loads prohibited reagents during optimization")
        print("  4. ✓ Merges constraints from multiple sources")
        print("  5. ✓ Handles edge cases (nonexistent files, None values)")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())