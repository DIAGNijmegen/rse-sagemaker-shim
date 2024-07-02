import copy
import sys
from contextlib import contextmanager
from importlib.metadata import version
from time import sleep
from uuid import uuid4

import docker
import httpx
import pytest

from tests import __version__
from tests.conftest import minio_container
from tests.utils import (
    encode_b64j,
    get_image_config,
    get_new_env_vars,
    mutate_image,
    pull_image,
    push_image,
)

# Tests for compatability with
# https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-batch-code.html


@pytest.fixture(autouse=True)
def _container_helper(request) -> None:
    marker = request.node.get_closest_marker("container")
    if marker:
        request.getfixturevalue("container")  # pragma: no cover


@contextmanager
def _container(*, base_image="ubuntu:latest", host_port=8080, cmd=None):
    client = docker.from_env()
    registry = client.containers.run(
        image="registry:2",
        ports={5000: None},
        auto_remove=True,
        detach=True,
        init=True,
    )
    # Wait for startup
    sleep(1)

    try:
        registry.reload()  # required to get ports
        port = registry.ports["5000/tcp"][0]["HostPort"]
        repo = f"localhost:{port}"
        repo_tag = f"{repo}/{base_image}"

        # Pull the base image
        container_image = client.images.pull(base_image)

        # Tag and push it to the local registry
        container_image.tag(repo_tag)
        push_image(client=client, repo_tag=repo_tag)

        config = get_image_config(repo_tag=repo_tag)
        env_vars = get_new_env_vars(existing_config=config)

        if cmd is not None:
            env_vars["GRAND_CHALLENGE_COMPONENT_CMD_B64J"] = encode_b64j(
                val=cmd
            )

        new_tag = mutate_image(
            repo_tag=repo_tag,
            env_vars=env_vars,
            version=__version__,
        )
        pull_image(client=client, repo_tag=new_tag)

        with minio_container() as minio:
            container_env = copy.deepcopy(minio.env)

            container_env["AWS_S3_ENDPOINT_URL"] = "http://minio:9000"

            container = client.containers.run(
                image=new_tag,
                # For batch transforms, SageMaker runs the container as
                # docker run image serve
                command="serve",
                # Containers must implement a web server that responds
                # to invocations and ping requests on port 8080.
                ports={8080: host_port},
                auto_remove=True,
                detach=True,
                # You can't use the init initializer as your entry point
                # in SageMaker containers because it gets confused by the
                # train and serve arguments
                init=False,
                environment=container_env,
                links={minio.container.name: "minio"},
                user=0,
            )

            # Wait for startup
            sleep(5)

            try:
                yield container
            finally:
                container.stop(timeout=0)
    finally:
        registry.stop(timeout=0)


@pytest.fixture(scope="session")
def container():
    with _container() as c:
        yield c


@pytest.mark.container
@pytest.mark.skipif(
    sys.platform != "linux", reason="does not run outside linux"
)
def test_container_responds_to_ping():
    response = httpx.get("http://localhost:8080/ping", timeout=30)

    # SageMaker waits for an HTTP 200 status code and an empty body
    # for a successful ping request before sending an invocations request.
    assert response.status_code == 200
    assert response.content == b""


@pytest.mark.container
@pytest.mark.skipif(
    sys.platform != "linux", reason="does not run outside linux"
)
def test_container_responds_to_execution_parameters():
    response = httpx.get(
        "http://localhost:8080/execution-parameters", timeout=30
    )

    assert response.json() == {
        "MaxConcurrentTransforms": 1,
        "BatchStrategy": "SINGLE_RECORD",
        "MaxPayloadInMB": 6,
    }


@pytest.mark.container
@pytest.mark.skipif(
    sys.platform != "linux", reason="does not run outside linux"
)
def test_invocations_endpoint(minio):
    # To receive inference requests, the container must have a web server
    # listening on port 8080 and must accept POST requests to the
    # /invocations endpoint.
    pk = str(uuid4())
    data = {
        "pk": str(uuid4()),
        "inputs": [],
        "output_bucket_name": minio.output_bucket_name,
        "output_prefix": f"test/{pk}",
    }
    response = httpx.post(
        "http://localhost:8080/invocations", json=data, timeout=30
    )

    # To obtain inferences, Amazon SageMaker sends a POST request to the
    # inference container. The POST request body contains data from
    # Amazon S3. Amazon SageMaker passes the request to the container,
    # and returns the inference result from the container, saving the
    # data from the response to Amazon S3.
    response = response.json()

    assert response["return_code"] == 0
    assert response["outputs"] == []
    assert response["sagemaker_shim_version"] == version("sagemaker-shim")


@pytest.mark.container
@pytest.mark.skipif(
    sys.platform != "linux", reason="does not run outside linux"
)
def test_alpine_image(minio):
    # https://github.com/JonathonReinhart/staticx/issues/143
    host_port = 8081
    with _container(
        base_image="python:3.12-alpine",
        host_port=host_port,
        cmd=["python", "-c", "print('hello_world')"],
    ):
        pk = str(uuid4())
        data = {
            "pk": pk,
            "inputs": [],
            "output_bucket_name": minio.output_bucket_name,
            "output_prefix": f"test/{pk}",
        }
        response = httpx.post(
            f"http://localhost:{host_port}/invocations", json=data, timeout=30
        )

        response = response.json()

        assert response["return_code"] == 0
        assert response["outputs"] == []
        assert response["sagemaker_shim_version"] == version("sagemaker-shim")
