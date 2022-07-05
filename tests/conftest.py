import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sagemaker_shim.app import app


def pytest_sessionstart(session):
    """https://docs.pytest.org/en/latest/reference/reference.html#_pytest.hookspec.pytest_sessionstart"""
    subprocess.check_call(
        ["make", "-C", Path(__file__).parent.parent / "dist", "all"]
    )


@pytest.fixture
def client():
    return TestClient(app=app)
