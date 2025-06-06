import io
import sys
from time import sleep

import docker
import pytest

from tests import __version__
from tests.utils import (
    get_image_config,
    get_new_env_vars,
    mutate_image,
    push_image,
)


@pytest.fixture(autouse=True)
def _registry_helper(request) -> None:
    marker = request.node.get_closest_marker("registry")
    if marker:
        request.getfixturevalue("registry")  # pragma: no cover


@pytest.fixture(scope="session")
def registry():
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
        yield f"localhost:{port}", client
    finally:
        registry.stop(timeout=0)


@pytest.mark.registry
@pytest.mark.skipif(
    sys.platform != "linux", reason="does not run outside linux"
)
def test_patch_image(registry):
    repo = registry[0]
    client = registry[1]

    dockerfile = io.BytesIO(
        b"""
        FROM busybox:latest
        """
    )
    repo_tag = f"{repo}/busybox:latest"

    client.images.build(fileobj=dockerfile, tag=repo_tag)
    push_image(client=client, repo_tag=repo_tag)

    config = get_image_config(repo_tag=repo_tag)
    env_vars = get_new_env_vars(existing_config=config)

    assert set(config["config"]["Env"]) == {
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    }
    assert env_vars == {
        "GRAND_CHALLENGE_COMPONENT_CMD_B64J": "WyJzaCJd",
        "GRAND_CHALLENGE_COMPONENT_ENTRYPOINT_B64J": "bnVsbA==",
        "GRAND_CHALLENGE_COMPONENT_USER": "0:0",
    }

    new_tag = mutate_image(
        repo_tag=repo_tag, env_vars=env_vars, version=__version__
    )
    new_config = get_image_config(repo_tag=new_tag)

    assert new_config["config"]["Entrypoint"] == ["/sagemaker-shim"]
    assert "Cmd" not in new_config["config"]
    assert set(new_config["config"]["Env"]) == {
        "GRAND_CHALLENGE_COMPONENT_CMD_B64J=WyJzaCJd",
        "GRAND_CHALLENGE_COMPONENT_ENTRYPOINT_B64J=bnVsbA==",
        "GRAND_CHALLENGE_COMPONENT_USER=0:0",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    }
