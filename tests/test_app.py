from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from patch_image import encode_b64j
from sagemaker_shim.app import app
from sagemaker_shim.models import InferenceTask


@pytest.fixture
def client():
    return TestClient(app=app)


def test_container_responds_to_ping(client):
    response = client.get("/ping")

    # SageMaker waits for an HTTP 200 status code and an empty body for
    # a successful ping request before sending an invocations request.
    assert response.status_code == 200
    assert response.content == b""


def test_container_responds_to_execution_parameters(client):
    response = client.get("/execution-parameters")

    assert response.json() == {
        "MaxConcurrentTransforms": 1,
        "BatchStrategy": "SINGLE_RECORD",
        "MaxPayloadInMB": 6,
    }


def test_invocations_endpoint(client, tmp_path, monkeypatch):
    # To receive inference requests, the container must have a web server
    # listening on port 8080 and must accept POST requests to the
    # /invocations endpoint.
    # TODO use FactoryBoy for tasks
    data = {
        "pk": str(uuid4()),
        "inputs": [],
        "output_bucket_name": "test",
        "output_prefix": "test",
    }

    input_path = tmp_path / "input"
    output_path = tmp_path / "output"

    input_path.mkdir()
    output_path.mkdir()

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_ENTRYPOINT_B64J",
        encode_b64j(val=["echo", "hello world"]),
    )
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_INPUT_PATH",
        str(input_path),
    )
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_OUTPUT_PATH",
        str(output_path),
    )

    response = client.post("/invocations", json=data)

    # To obtain inferences, Amazon SageMaker sends a POST request to the
    # inference container. The POST request body contains data from
    # Amazon S3. Amazon SageMaker passes the request to the container,
    # and returns the inference result from the container, saving the
    # data from the response to Amazon S3.
    response = response.json()
    assert response["return_code"] == 0


@pytest.mark.parametrize(
    "cmd,entrypoint,expected",
    (
        (
            None,
            "exec_entry p1_entry",
            ["/bin/sh", "-c", "exec_entry p1_entry"],
        ),
        (None, ["exec_entry", "p1_entry"], ["exec_entry", "p1_entry"]),
        (["exec_cmd", "p1_cmd"], None, ["exec_cmd", "p1_cmd"]),
        (
            ["exec_cmd", "p1_cmd"],
            "exec_entry p1_entry",
            ["/bin/sh", "-c", "exec_entry p1_entry"],
        ),
        (
            ["exec_cmd", "p1_cmd"],
            ["exec_entry", "p1_entry"],
            ["exec_entry", "p1_entry", "exec_cmd", "p1_cmd"],
        ),
        (["p1_cmd", "p2_cmd"], None, ["p1_cmd", "p2_cmd"]),
        (
            ["p1_cmd", "p2_cmd"],
            "exec_entry p1_entry",
            ["/bin/sh", "-c", "exec_entry p1_entry"],
        ),
        (
            ["p1_cmd", "p2_cmd"],
            ["exec_entry", "p1_entry"],
            ["exec_entry", "p1_entry", "p1_cmd", "p2_cmd"],
        ),
        ("exec_cmd p1_cmd", None, ["/bin/sh", "-c", "exec_cmd p1_cmd"]),
        (
            "exec_cmd p1_cmd",
            "exec_entry p1_entry",
            ["/bin/sh", "-c", "exec_entry p1_entry"],
        ),
        (
            "exec_cmd p1_cmd",
            ["exec_entry", "p1_entry"],
            ["exec_entry", "p1_entry", "/bin/sh", "-c", "exec_cmd p1_cmd"],
        ),
    ),
)
def test_proc_args(cmd, entrypoint, expected, monkeypatch):
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_ENTRYPOINT_B64J",
        encode_b64j(val=entrypoint),
    )
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_CMD_B64J",
        encode_b64j(val=cmd),
    )
    j = InferenceTask(
        pk=uuid4(),
        inputs=[],
        output_bucket_name="test",
        output_prefix="test",
    )

    assert j.proc_args == expected


@pytest.mark.parametrize(
    "envvars",
    (
        (),
        ("GRAND_CHALLENGE_COMPONENT_ENTRYPOINT_B64J",),
        (
            "GRAND_CHALLENGE_COMPONENT_ENTRYPOINT_B64J",
            "GRAND_CHALLENGE_COMPONENT_CMD_B64J",
        ),
        ("GRAND_CHALLENGE_COMPONENT_CMD_B64J"),
    ),
)
def test_unset_cmd_and_entrypoint(envvars, monkeypatch):
    for var in envvars:
        monkeypatch.setenv(var, encode_b64j(val=None))

    j = InferenceTask(
        pk=uuid4(),
        inputs=[],
        output_bucket_name="test",
        output_prefix="test",
    )

    with pytest.raises(ValueError) as e:
        j.proc_args

    assert "Either cmd or entrypoint must be set" in str(e)


@pytest.mark.parametrize(
    "val",
    (
        (None),
        (["exec_cmd", "p1_cmd"]),
        ("exec_cmd p1_cmd"),
        ("c\xf7>"),
        ("👍"),
        ("null"),
    ),
)
def test_decode_b64j(val):
    encoded = encode_b64j(val=val)
    assert InferenceTask.decode_b64j(encoded=encoded) == val


def test_decode_returns_none():
    assert InferenceTask.decode_b64j(encoded=None) is None
