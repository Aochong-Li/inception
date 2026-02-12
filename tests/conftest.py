"""
Pytest configuration for inception-eval tests.

Defines markers for test categorization:
- integration: Tests that make real API calls
- unit: Tests that run entirely in-memory without external dependencies
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests that make real API calls (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests that run without external dependencies"
    )
