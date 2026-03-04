import logging
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

import sagemaker_shim.app
from sagemaker_shim.models import UserProcess


def pytest_sessionstart(session):
    # Reduce the level of the aiobotocore logger as this
    # causes issues with writing to closed streams
    # when used in tests that use capsys and raise exceptions.
    logging.getLogger("aiobotocore").setLevel(logging.WARNING)

    """https://docs.pytest.org/en/latest/reference/reference.html#_pytest.hookspec.pytest_sessionstart"""
    if sys.platform == "linux":
        subprocess.check_call(
            ["make", "-C", Path(__file__).parent.parent / "dist", "all"]
        )


@pytest.fixture
def client():
    sagemaker_shim.app.USER_PROCESS = UserProcess()

    return TestClient(app=sagemaker_shim.app.app)


class LocalS3(NamedTuple):
    input_bucket_name: str
    output_bucket_name: str
    container: Container
    port: int
    env: dict[str, str]


@contextmanager
def local_s3_container():
    input_bucket_name = "test-inputs"
    output_bucket_name = "test-outputs"

    environment = {
        "AWS_ACCESS_KEY_ID": "s3admin",
        "AWS_SECRET_ACCESS_KEY": "s3admin",
    }

    client = docker.from_env()
    local_s3 = client.containers.run(
        image="chrislusf/seaweedfs",
        command="mini",
        ports={8333: None},
        auto_remove=True,
        detach=True,
        init=True,
        environment=environment,
    )

    # Wait for startup
    sleep(1)

    mpatch = pytest.MonkeyPatch()

    try:
        local_s3.reload()  # required to get ports
        port = local_s3.ports["8333/tcp"][0]["HostPort"]

        environment["AWS_S3_ENDPOINT_URL"] = f"http://localhost:{port}"

        for key, value in environment.items():
            mpatch.setenv(key, value)

        yield LocalS3(
            input_bucket_name=input_bucket_name,
            output_bucket_name=output_bucket_name,
            container=local_s3,
            port=port,
            env=environment,
        )
    finally:
        mpatch.undo()
        local_s3.stop(timeout=0)


@pytest.fixture(scope="session")
def local_s3():
    with local_s3_container() as m:
        yield m
