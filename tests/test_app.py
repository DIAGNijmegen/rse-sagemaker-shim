import logging.config
from asyncio.streams import _DEFAULT_LIMIT
from copy import deepcopy
from unittest.mock import patch
from uuid import uuid4

import pytest

import sagemaker_shim.app
from sagemaker_shim.logging import LOGGING_CONFIG
from sagemaker_shim.models import UserProcess
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


def test_invocations_endpoint(client, tmp_path, monkeypatch, capsys, local_s3):
    # To receive inference requests, the container must have a web server
    # listening on port 8080 and must accept POST requests to the
    # /invocations endpoint.
    input_path = tmp_path / "input"
    linked_input_parent = tmp_path / "linked-input"

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_INPUT_PATH", str(input_path.absolute())
    )
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_LINKED_INPUT_PARENT",
        str(linked_input_parent),
    )

    pk = str(uuid4())
    data = {
        "pk": pk,
        "inputs": [],
        "output_bucket_name": local_s3.output_bucket_name,
        "output_prefix": f"test/{pk}",
        "timeout": "PT10S",
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

    # The logs need to be interpretable by grand challenge
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
    p = UserProcess()

    assert p.proc_args == expected


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

    p = UserProcess()

    with pytest.raises(ValueError) as e:
        p.proc_args

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
    assert UserProcess.decode_b64j(encoded=encoded) == val


def test_decode_returns_none():
    assert UserProcess.decode_b64j(encoded=None) is None


class TestGracefulShutdown:
    def test_ping_returns_503_when_unhealthy(self, client):
        sagemaker_shim.app.USER_PROCESS._healthy = False

        response = client.get("/ping")

        assert response.status_code == 503

    def test_ping_initiates_shutdown_when_unhealthy(self, client):
        sagemaker_shim.app.USER_PROCESS._healthy = False

        with patch("sagemaker_shim.app._terminate") as mock_terminate:
            client.get("/ping")

            assert sagemaker_shim.app._SHUTTING_DOWN is True
            # _terminate is scheduled via call_later, not called directly
            mock_terminate.assert_not_called()

    def test_ping_returns_503_during_shutdown(self, client):
        """Once shutdown is initiated, /ping returns 503 even if the
        process hasn't terminated yet."""
        sagemaker_shim.app._SHUTTING_DOWN = True

        response = client.get("/ping")

        assert response.status_code == 503

    def test_invocations_returns_503_during_shutdown(self, client):
        """Invocations that arrive during the grace period get a clean 503."""
        sagemaker_shim.app._SHUTTING_DOWN = True

        data = {
            "pk": str(uuid4()),
            "inputs": [],
            "output_bucket_name": "test-bucket",
            "output_prefix": "test/",
            "timeout": "PT10S",
        }
        response = client.post("/invocations", json=data)

        assert response.status_code == 503

    def test_shutdown_only_initiated_once(self, client):
        """Multiple unhealthy pings don't stack up shutdown calls."""
        sagemaker_shim.app.USER_PROCESS._healthy = False

        with patch("sagemaker_shim.app.asyncio.get_event_loop") as mock_loop:
            client.get("/ping")
            client.get("/ping")
            client.get("/ping")

            # call_later should only be called once
            mock_loop.return_value.call_later.assert_called_once()

    def test_terminate_sends_sigterm(self):
        with patch("sagemaker_shim.app.os.kill") as mock_kill:
            sagemaker_shim.app._terminate()

            import os
            import signal

            mock_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)
