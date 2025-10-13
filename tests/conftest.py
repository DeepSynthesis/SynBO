"""Configuration file for pytest"""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Test configuration
import pytest

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests (may be slow)"
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests (should be fast)"
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require GPU (may be skipped)"
    )