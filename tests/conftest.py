"""Test configuration and fixtures for ShardGuard tests."""

import os

import pytest
from rich.console import Console


@pytest.fixture
def mock_console():
    """Provide a mock console that doesn't output to stdout/stderr."""
    # Create a console that writes to /dev/null to avoid test output pollution

    null_file = open(os.devnull, "w")
    return Console(file=null_file, stderr=False)


@pytest.fixture
def sample_json_response():
    """Provide a sample valid JSON response for testing."""
    return """
    {
        "original_prompt": "Test prompt with [[P1]]",
        "sub_prompts": [
            {
                "id": 1,
                "content": "First task",
                "opaque_values": {}
            },
            {
                "id": 2,
                "content": "Second task with [[P1]]",
                "opaque_values": {
                    "[[P1]]": "sensitive_data"
                }
            }
        ]
    }
    """


@pytest.fixture
def sample_plan_data():
    """Provide sample plan data for testing."""
    return {
        "original_prompt": "Process sensitive information",
        "sub_prompts": [
            {"id": 1, "content": "Validate input", "opaque_values": {}},
            {
                "id": 2,
                "content": "Process [[P1]] data",
                "opaque_values": {"[[P1]]": "medical_record"},
            },
        ],
    }


# Configure pytest to handle async tests if needed
pytest_plugins = ["pytest_asyncio"]


# Configure test discovery
def pytest_configure(config):
    """Configure pytest settings."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
