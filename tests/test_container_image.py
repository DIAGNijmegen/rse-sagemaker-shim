import subprocess
from importlib.metadata import version
from pathlib import Path
from time import sleep
from uuid import uuid4

import docker
import httpx
import pytest

from patch_image import get_image_config, get_new_env_vars, mutate_image
from tests import __version__

# Tests for compatability with
# https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-batch-code.html


@pytest.fixture(autouse=True)
def _container_helper(request) -> None:
    marker = request.node.get_closest_marker("container")
    if marker:
        request.getfixturevalue("container")  # pragma: no cover


@pytest.fixture(scope="session")
def container():
    subprocess.check_call(
        ["make", "-C", Path(__file__).parent.parent / "dist", "all"]
    )

    client = docker.from_env()
    registry = client.containers.run(
        image="registry:2.7",
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
        base_image = "hello-world:latest"
        repo_tag = f"{repo}/{base_image}"

        # Pull the base image
        container_image = client.images.pull(base_image)

        # Tag and push it to the local registry
        container_image.tag(repo_tag)
        client.images.push(repo_tag)

        config = get_image_config(repo_tag=repo_tag)
        env_vars = get_new_env_vars(existing_config=config)
        new_tag = mutate_image(
            repo_tag=repo_tag,
            env_vars=env_vars,
            version=__version__,
        )
        client.images.pull(new_tag)

        container = client.containers.run(
            image=new_tag,
            # For batch transforms, SageMaker runs the container as
            # docker run image serve
            command="serve",
            # Containers must implement a web server that responds
            # to invocations and ping requests on port 8080.
            ports={8080: 8080},
            auto_remove=True,
            detach=True,
            # You can't use the init initializer as your entry point
            # in SageMaker containers because it gets confused by the
            # train and serve arguments
            init=False,
        )

        # Wait for startup
        sleep(3)

        try:
            yield container
        finally:
            container.stop(timeout=0)
    finally:
        registry.stop(timeout=0)


@pytest.mark.container
def test_container_responds_to_ping():
    response = httpx.get("http://localhost:8080/ping")

    # SageMaker waits for an HTTP 200 status code and an empty body
    # for a successful ping request before sending an invocations request.
    assert response.status_code == 200
    assert response.content == b""


@pytest.mark.container
def test_container_responds_to_execution_parameters():
    response = httpx.get("http://localhost:8080/execution-parameters")

    assert response.json() == {
        "MaxConcurrentTransforms": 1,
        "BatchStrategy": "SINGLE_RECORD",
        "MaxPayloadInMB": 6,
    }


@pytest.mark.container
def test_invocations_endpoint():
    # To receive inference requests, the container must have a web server
    # listening on port 8080 and must accept POST requests to the
    # /invocations endpoint.
    data = {
        "pk": str(uuid4()),
        "inputs": [],
        "output_bucket_name": "test",
        "output_prefix": "test",
    }
    response = httpx.post("http://localhost:8080/invocations", json=data)

    # To obtain inferences, Amazon SageMaker sends a POST request to the
    # inference container. The POST request body contains data from
    # Amazon S3. Amazon SageMaker passes the request to the container,
    # and returns the inference result from the container, saving the
    # data from the response to Amazon S3.
    response = response.json()

    assert response["return_code"] == 0
    assert response["outputs"] == []
    assert response["sagemaker_shim_version"] == version("sagemaker-shim")
