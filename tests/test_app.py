import logging.config
from asyncio.streams import _DEFAULT_LIMIT
from copy import deepcopy
from uuid import uuid4

import pytest

from sagemaker_shim.logging import LOGGING_CONFIG
from sagemaker_shim.models import InferenceTask
from tests.utils import encode_b64j


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


def test_invocations_endpoint(client, tmp_path, monkeypatch, capsys, minio):
    # To receive inference requests, the container must have a web server
    # listening on port 8080 and must accept POST requests to the
    # /invocations endpoint.

    pk = str(uuid4())
    data = {
        "pk": pk,
        "inputs": [],
        "output_bucket_name": minio.output_bucket_name,
        "output_prefix": f"test/{pk}",
    }

    input_path = tmp_path / "input"
    output_path = tmp_path / "output"

    input_path.mkdir()
    output_path.mkdir()

    bigfile = tmp_path / "test.txt"
    with open(bigfile, "w") as f:
        # asyncio has a limit to the size of log lines,
        # Anything above this will not be logged
        f.write("a" * (_DEFAULT_LIMIT + 1))
        f.write("\n")

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_ENTRYPOINT_B64J",
        encode_b64j(
            val=[
                "sh",
                "-c",
                f"cat {bigfile} && echo hellostdout && echo hellostderr 1>&2",
            ]
        ),
    )
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_INPUT_PATH",
        str(input_path),
    )
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_OUTPUT_PATH",
        str(output_path),
    )
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_SET_EXTRA_GROUPS", "False")

    debug_log = deepcopy(LOGGING_CONFIG)
    debug_log["root"]["level"] = "DEBUG"
    logging.config.dictConfig(debug_log)

    response = client.post("/invocations", json=data)

    # The logs need to be interprable by grand challenge
    captured = capsys.readouterr()
    assert (
        '{"log": "hellostdout", "level": "INFO", "source": "stdout", '
        f'"internal": false, "task": "{pk}"}}\n'
    ) in captured.out
    assert (
        '{"log": "return_code=0", "level": "INFO", "source": "stdout", '
        '"internal": true, "task": null}'
    ) in captured.out
    assert captured.err == (
        '{"log": "WARNING: A log line was skipped as it was too long", '
        '"level": "WARNING", "source": "stderr", "internal": false, '
        f'"task": "{pk}"}}\n'
        '{"log": "hellostderr", "level": "WARNING", "source": "stderr", '
        f'"internal": false, "task": "{pk}"}}\n'
    )

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
        pk=str(uuid4()),
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
        pk=str(uuid4()),
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
