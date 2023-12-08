import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from time import sleep
from typing import NamedTuple

import docker
import pytest
from docker.models.containers import Container
from fastapi.testclient import TestClient

from sagemaker_shim.app import app


def pytest_sessionstart(session):
    """https://docs.pytest.org/en/latest/reference/reference.html#_pytest.hookspec.pytest_sessionstart"""
    if sys.platform == "linux":
        subprocess.check_call(
            ["make", "-C", Path(__file__).parent.parent / "dist", "all"]
        )


@pytest.fixture
def client():
    return TestClient(app=app)


class Minio(NamedTuple):
    input_bucket_name: str
    output_bucket_name: str
    container: Container
    port: int


@contextmanager
def minio_container():
    input_bucket_name = "test-inputs"
    output_bucket_name = "test-outputs"

    client = docker.from_env()
    minio = client.containers.run(
        image="minio/minio:latest",
        entrypoint="/bin/sh",
        command=[
            "-c",
            f"mkdir -p /data/{input_bucket_name} /data/{output_bucket_name} "
            "&& minio --compat server /data",
        ],
        ports={9000: None},
        auto_remove=True,
        detach=True,
        init=True,
    )

    # Wait for startup
    sleep(1)

    mpatch = pytest.MonkeyPatch()

    try:
        minio.reload()  # required to get ports
        port = minio.ports["9000/tcp"][0]["HostPort"]

        mpatch.setenv("AWS_ACCESS_KEY_ID", "minioadmin")
        mpatch.setenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
        mpatch.setenv("AWS_S3_ENDPOINT_URL", f"http://localhost:{port}")

        yield Minio(
            input_bucket_name=input_bucket_name,
            output_bucket_name=output_bucket_name,
            container=minio,
            port=port,
        )
    finally:
        mpatch.undo()
        minio.stop(timeout=0)


@pytest.fixture(scope="session")
def minio():
    with minio_container() as m:
        yield m
