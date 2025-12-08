import sys
import os

# Add the reactionopt directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

try:
    from src.rxnopt.bo_algorithm.hebo_opt import HEBOOptimizerWrapper
    import numpy as np
    import torch
    import pandas as pd

    def test_hebo_variability():
        """Test that HEBO produces different results each time"""
        print("Testing HEBO variability...")

        # Create mock data
        np.random.seed(42)  # Fixed seed for reproducible test data
        training_X = np.random.rand(10, 3) * 2 - 1  # 10 samples, 3 dimensions, normalized to [-1, 1]
        training_y = {
            'yield': np.random.rand(10) * 100,
            'cost': np.random.rand(10) * 50
        }

        candidate_X = np.random.rand(100, 3) * 2 - 1  # 100 candidate points
        name_data = np.array([f"cond_{i}" for i in range(100)])

        opt_direct_info = [
            {"opt_direct": "max", "opt_range": [0, 100]},  # yield
            {"opt_direct": "min", "opt_range": [0, 50]}    # cost
        ]

        device = torch.device("cpu")

        # Run optimization multiple times
        results = []
        for i in range(3):
            print(f"Run {i+1}...")
            # Create HEBO optimizer wrapper
            hebo_wrapper = HEBOOptimizerWrapper(name_data=name_data)

            # Run optimization
            selected_conditions, recommend_type, pred_mean, pred_std = hebo_wrapper.optimize(
                training_X=training_X,
                training_y=training_y,
                candidate_X=candidate_X,
                opt_direct_info=opt_direct_info,
                device=device,
                batch_size=5
            )

            results.append(selected_conditions)
            print(f"  Selected conditions: {selected_conditions}")

        # Check if results are different
        all_same = True
        for i in range(1, len(results)):
            if not np.array_equal(results[0], results[i]):
                all_same = False
                break

        if all_same:
            print("❌ ERROR: All runs produced identical results!")
            return False
        else:
            print("✅ SUCCESS: Different results produced in each run!")
            return True

    if __name__ == "__main__":
        success = test_hebo_variability()
        sys.exit(0 if success else 1)

except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)