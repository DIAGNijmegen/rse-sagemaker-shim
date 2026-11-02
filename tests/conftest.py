import asyncio
import logging
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from time import sleep
from typing import NamedTuple

import aioboto3
import docker
import pytest
from botocore.exceptions import EndpointConnectionError
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
    sagemaker_shim.app.USER_PROCESS._healthy = True

    return TestClient(app=sagemaker_shim.app.app)


class LocalS3(NamedTuple):
    input_bucket_name: str
    output_bucket_name: str
    container: Container
    port: int
    env: dict[str, str]


def container_endpoint(*, container: Container, container_port: int) -> str:
    """Return a URL that reaches a port inside the container.

    The GitHub-hosted runners stopped reliably forwarding published
    container ports (host connections are accepted then closed without a
    response), so on Linux we talk to the container directly on its bridge
    IP, which is routable from the host. Elsewhere (e.g. macOS with Docker
    Desktop) the bridge IP is not routable, so we use the published port on
    localhost.
    """
    if sys.platform == "linux":
        networks = container.attrs["NetworkSettings"]["Networks"]
        container_ip = next(
            (n["IPAddress"] for n in networks.values() if n["IPAddress"]),
            "",
        )

        if not container_ip:
            raise RuntimeError("Container has no bridge IP address")

        endpoint = f"http://{container_ip}:{container_port}"
    else:
        host_port = container.ports[f"{container_port}/tcp"][0]["HostPort"]
        endpoint = f"http://localhost:{host_port}"

    return endpoint


def _resolve_endpoint_when_ready(
    *, container: Container, container_port: int, attempts: int = 30
) -> str:
    """Reload until Docker has assigned networking, then return the URL."""
    for _ in range(attempts):
        container.reload()
        try:
            return container_endpoint(
                container=container, container_port=container_port
            )
        except (KeyError, IndexError, RuntimeError):
            sleep(1)
    raise RuntimeError("Container networking did not become ready")


def _create_buckets_when_ready(
    *, endpoint_url: str, bucket_names: set[str], attempts: int = 60
) -> None:
    """Create the buckets, retrying until the S3 endpoint is serving.

    Creating the buckets doubles as the readiness check, so no fixed sleep
    is needed.
    """

    async def create_buckets() -> None:
        session = aioboto3.Session()
        async with session.client(
            "s3", endpoint_url=endpoint_url
        ) as s3_client:
            for bucket_name in bucket_names:
                await s3_client.create_bucket(Bucket=bucket_name)

    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            asyncio.run(create_buckets())
            return
        except EndpointConnectionError as error:
            last_error = error
            sleep(1)
    raise RuntimeError(
        f"S3 endpoint {endpoint_url} did not start in time"
    ) from last_error


@contextmanager
def local_s3_container():
    input_bucket_name = "test-inputs"
    output_bucket_name = "test-outputs"

    environment = {
        "AWS_ACCESS_KEY_ID": "s3admin",
        "AWS_SECRET_ACCESS_KEY": "s3admin",
    }

    docker_client = docker.from_env()
    local_s3 = docker_client.containers.run(
        image="chrislusf/seaweedfs",
        command="mini",
        ports={8333: None},
        auto_remove=True,
        detach=True,
        init=True,
        environment=environment,
    )

    mpatch = pytest.MonkeyPatch()
    try:
        s3_endpoint_url = _resolve_endpoint_when_ready(
            container=local_s3, container_port=8333
        )
        port = local_s3.ports["8333/tcp"][0]["HostPort"]

        environment["AWS_S3_ENDPOINT_URL"] = s3_endpoint_url

        for key, value in environment.items():
            mpatch.setenv(key, value)

        _create_buckets_when_ready(
            endpoint_url=s3_endpoint_url,
            bucket_names={input_bucket_name, output_bucket_name},
        )

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
