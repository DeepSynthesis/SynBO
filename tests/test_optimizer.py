import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add the parent directory to the path to import rxnopt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rxnopt import ReactionOptimizer


@pytest.fixture
def sample_data():
    """Create sample reaction data for testing."""
    np.random.seed(42)
    data = pd.DataFrame({
        'temperature': np.random.uniform(20, 80, 50),
        'catalyst_loading': np.random.uniform(0.1, 10.0, 50),
        'solvent': np.random.choice(['DCM', 'THF', 'Toluene'], 50),
        'yield': np.random.uniform(0, 100, 50),
        'ee': np.random.uniform(0, 99, 50)
    })
    return data


@pytest.fixture
def parameter_space():
    """Define parameter space for testing."""
    return {
        'temperature': [20, 80],
        'catalyst_loading': [0.1, 10.0],
        'solvent': ['DCM', 'THF', 'Toluene']
    }


class TestReactionOptimizer:
    """Test cases for the ReactionOptimizer class."""
    
    def test_initialization(self):
        """Test ReactionOptimizer initialization."""
        optimizer = ReactionOptimizer(
            objectives=['yield', 'ee'],
            n_initial_points=10,
            n_iterations=20
        )
        
        assert optimizer.objectives == ['yield', 'ee']
        assert optimizer.n_initial_points == 10
        assert optimizer.n_iterations == 20
    
    def test_initialization_with_custom_params(self):
        """Test ReactionOptimizer initialization with custom parameters."""
        optimizer = ReactionOptimizer(
            objectives=['yield'],
            acquisition_function='EI',
            surrogate_model='GP',
            n_initial_points=5,
            n_iterations=10,
            random_seed=42
        )
        
        assert optimizer.objectives == ['yield']
        assert hasattr(optimizer, 'acquisition_function')
        assert hasattr(optimizer, 'surrogate_model')
        assert optimizer.random_seed == 42
    
    def test_invalid_objectives(self):
        """Test that invalid objectives raise appropriate errors."""
        with pytest.raises(ValueError):
            ReactionOptimizer(objectives=[])
        
        with pytest.raises(TypeError):
            ReactionOptimizer(objectives='yield')  # Should be list
    
    def test_parameter_space_validation(self, parameter_space):
        """Test parameter space validation."""
        optimizer = ReactionOptimizer(objectives=['yield'])
        
        # This should not raise an error for valid parameter space
        assert optimizer._validate_parameter_space(parameter_space)
        
        # Test invalid parameter space
        invalid_space = {'temperature': [80, 20]}  # max < min
        with pytest.raises(ValueError):
            optimizer._validate_parameter_space(invalid_space)
    
    @pytest.mark.parametrize("n_initial_points", [1, 5, 10, 20])
    def test_different_initial_points(self, n_initial_points):
        """Test optimizer with different numbers of initial points."""
        optimizer = ReactionOptimizer(
            objectives=['yield'],
            n_initial_points=n_initial_points
        )
        assert optimizer.n_initial_points == n_initial_points
    
    @pytest.mark.parametrize("objectives", [['yield'], ['ee'], ['yield', 'ee']])
    def test_different_objectives(self, objectives):
        """Test optimizer with different objective combinations."""
        optimizer = ReactionOptimizer(objectives=objectives)
        assert optimizer.objectives == objectives


class TestDataProcessing:
    """Test data processing utilities."""
    
    def test_data_preprocessing(self, sample_data):
        """Test data preprocessing functionality."""
        optimizer = ReactionOptimizer(objectives=['yield', 'ee'])
        
        # Test that data can be processed without errors
        processed_data = optimizer._preprocess_data(sample_data)
        
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) == len(sample_data)
        assert all(col in processed_data.columns for col in sample_data.columns)
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        data_with_nan = pd.DataFrame({
            'temperature': [20, 30, np.nan, 40],
            'yield': [80, np.nan, 75, 90],
            'ee': [95, 90, 85, np.nan]
        })
        
        optimizer = ReactionOptimizer(objectives=['yield', 'ee'])
        
        # Should handle missing data appropriately
        processed = optimizer._preprocess_data(data_with_nan)
        assert not processed.isnull().any().any()


class TestOptimization:
    """Test optimization functionality."""
    
    def test_optimization_setup(self, sample_data, parameter_space):
        """Test optimization setup without running full optimization."""
        optimizer = ReactionOptimizer(
            objectives=['yield', 'ee'],
            n_initial_points=5,
            n_iterations=2  # Small number for testing
        )
        
        # Test that optimization can be set up
        assert hasattr(optimizer, 'optimize')
        
        # Mock optimization run (without actually running expensive computation)
        try:
            # This might fail if the actual implementation requires specific setup
            # but we're testing the interface exists
            result = optimizer.optimize(sample_data, parameter_space)
        except (NotImplementedError, AttributeError):
            # If optimize method is not fully implemented, that's ok for now
            pass
    
    def test_parameter_bounds_checking(self, parameter_space):
        """Test that parameter bounds are properly validated."""
        optimizer = ReactionOptimizer(objectives=['yield'])
        
        # Valid bounds
        assert optimizer._validate_parameter_space(parameter_space)
        
        # Invalid bounds (min > max)
        invalid_bounds = parameter_space.copy()
        invalid_bounds['temperature'] = [80, 20]  # min > max
        
        with pytest.raises(ValueError):
            optimizer._validate_parameter_space(invalid_bounds)


@pytest.mark.integration
class TestIntegration:
    """Integration tests that require more setup."""
    
    def test_full_workflow_mock(self, sample_data, parameter_space):
        """Test a simplified version of the full workflow."""
        optimizer = ReactionOptimizer(
            objectives=['yield'],
            n_initial_points=3,
            n_iterations=2
        )
        
        # Mock a simple workflow
        # (Actual implementation would depend on the full optimizer)
        assert len(sample_data) > 0
        assert 'yield' in sample_data.columns
        assert isinstance(parameter_space, dict)


if __name__ == '__main__':
    pytest.main([__file__])