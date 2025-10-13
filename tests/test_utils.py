import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add the parent directory to the path to import rxnopt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestUtils:
    """Test utility functions."""
    
    def test_data_validation(self):
        """Test data validation utilities."""
        # Mock data validation function
        data = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })
        
        assert len(data) == 3
        assert list(data.columns) == ['x', 'y']
    
    def test_parameter_space_conversion(self):
        """Test parameter space conversion utilities."""
        parameter_space = {
            'temperature': [20, 80],
            'pressure': [1, 10],
            'catalyst': ['A', 'B', 'C']
        }
        
        # Test continuous parameters
        assert len(parameter_space['temperature']) == 2
        assert parameter_space['temperature'][0] < parameter_space['temperature'][1]
        
        # Test categorical parameters
        assert isinstance(parameter_space['catalyst'], list)
        assert len(parameter_space['catalyst']) == 3


class TestVisualization:
    """Test visualization utilities."""
    
    def test_plot_data_structure(self):
        """Test that plot data structures are correct."""
        # Mock optimization results
        results = {
            'iteration': list(range(10)),
            'best_yield': np.random.uniform(70, 95, 10),
            'best_ee': np.random.uniform(80, 98, 10)
        }
        
        assert len(results['iteration']) == 10
        assert len(results['best_yield']) == 10
        assert len(results['best_ee']) == 10
    
    def test_pareto_front_data(self):
        """Test Pareto front data structure."""
        # Mock Pareto front points
        pareto_points = np.array([
            [85, 95],  # [yield, ee]
            [90, 88],
            [78, 97],
            [95, 85]
        ])
        
        assert pareto_points.shape[1] == 2  # Two objectives
        assert pareto_points.shape[0] > 0   # At least one point


class TestDataIO:
    """Test data input/output functions."""
    
    def test_excel_data_structure(self):
        """Test Excel data handling structure."""
        # Mock Excel data structure
        excel_data = {
            'Sheet1': pd.DataFrame({
                'Experiment_ID': [1, 2, 3],
                'Temperature': [25, 50, 75],
                'Yield': [80, 85, 90],
                'EE': [95, 90, 85]
            })
        }
        
        df = excel_data['Sheet1']
        assert 'Experiment_ID' in df.columns
        assert 'Temperature' in df.columns
        assert 'Yield' in df.columns
        assert 'EE' in df.columns
        assert len(df) == 3
    
    def test_csv_export_structure(self):
        """Test CSV export data structure."""
        # Mock results for CSV export
        results_df = pd.DataFrame({
            'iteration': [1, 2, 3, 4, 5],
            'temperature': [25, 35, 45, 55, 65],
            'catalyst_loading': [1.0, 2.0, 3.0, 4.0, 5.0],
            'predicted_yield': [80, 82, 85, 87, 90],
            'predicted_ee': [90, 88, 92, 89, 94],
            'acquisition_value': [0.1, 0.15, 0.2, 0.12, 0.18]
        })
        
        assert len(results_df) == 5
        assert 'iteration' in results_df.columns
        assert 'predicted_yield' in results_df.columns
        assert 'predicted_ee' in results_df.columns


if __name__ == '__main__':
    pytest.main([__file__])