import io
import json
import resource
from unittest.mock import patch
from uuid import uuid4

import pytest
from click.testing import CliRunner

from sagemaker_shim.cli import cli
from sagemaker_shim.models import get_s3_client, get_s3_file_content
from tests.utils import encode_b64j


def test_invoke_neither_set():
    runner = CliRunner()
    result = runner.invoke(cli, ["invoke"])
    assert result.exit_code == 2
    assert "Error: One of tasks or file should be set" in result.output


def test_invoke_both_set():
    runner = CliRunner()
    result = runner.invoke(cli, ["invoke", "-f", "fasd", "-t", "[]"])
    assert result.exit_code == 2
    assert "Only one of tasks or file should be set" in result.output


def test_invoke_empty_task_list():
    runner = CliRunner()
    result = runner.invoke(cli, ["invoke", "-t", "[]"])
    assert result.exit_code == 2
    assert "Empty task list provided" in result.output


@pytest.mark.parametrize(
    "task_list",
    (
        "gdfsa",
        "[{}]",
        "1",
        1,
        "{}",
    ),
)
def test_invoke_invalid_task_list(task_list):
    runner = CliRunner()
    result = runner.invoke(cli, ["invoke", "-t", task_list])
    assert result.exit_code == 2
    assert "The tasks definition is invalid" in result.output


def test_invoke_missing_s3_file(minio):
    runner = CliRunner()
    result = runner.invoke(
        cli, ["invoke", "-f", f"s3://{minio.output_bucket_name}/missing.json"]
    )
    assert result.exit_code == 2
    assert (
        "An error occurred (404) when calling the HeadObject operation: Not Found"
        in result.output
    )


def test_invoke_bad_bucket(minio):
    runner = CliRunner()
    result = runner.invoke(cli, ["invoke", "-f", "s3://fasd/missing.json"])
    assert result.exit_code == 2
    assert (
        "An error occurred (404) when calling the HeadObject operation: Not Found"
        in result.output
    )


@pytest.mark.parametrize(
    "cmd,expected_return_code",
    (
        (["echo", "hello"], 0),
        (["bash", "-c", "exit 1"], 1),
    ),
)
def test_inference_from_task_list(
    minio, monkeypatch, cmd, expected_return_code
):
    pk1, pk2 = str(uuid4()), str(uuid4())
    tasks = [
        {
            "pk": pk1,
            "inputs": [],
            "output_bucket_name": minio.output_bucket_name,
            "output_prefix": f"tasks/{pk1}",
        },
        {
            "pk": pk2,
            "inputs": [],
            "output_bucket_name": minio.output_bucket_name,
            "output_prefix": f"tasks/{pk2}",
        },
    ]

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_CMD_B64J",
        encode_b64j(val=cmd),
    )
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_SET_EXTRA_GROUPS", "False")
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USE_LINKED_INPUT", "False")

    runner = CliRunner()
    runner.invoke(cli, ["invoke", "-t", json.dumps(tasks)])

    for pk in pk1, pk2:
        serialised_invocation = get_s3_file_content(
            s3_uri=f"s3://{minio.output_bucket_name}/tasks/{pk}/.sagemaker_shim/inference_result.json"
        )
        parsed_result = json.loads(serialised_invocation)

        assert parsed_result["return_code"] == expected_return_code
        assert parsed_result["pk"] == pk


@pytest.mark.parametrize(
    "cmd,expected_return_code",
    (
        (["echo", "hello"], 0),
        (["bash", "-c", "exit 1"], 1),
    ),
)
def test_inference_from_s3_uri(minio, monkeypatch, cmd, expected_return_code):
    pk1, pk2 = str(uuid4()), str(uuid4())
    tasks = [
        {
            "pk": pk1,
            "inputs": [],
            "output_bucket_name": minio.output_bucket_name,
            "output_prefix": f"tasks/{pk1}",
        },
        {
            "pk": pk2,
            "inputs": [],
            "output_bucket_name": minio.output_bucket_name,
            "output_prefix": f"tasks/{pk2}",
        },
    ]

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_CMD_B64J",
        encode_b64j(val=cmd),
    )
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_SET_EXTRA_GROUPS", "False")
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USE_LINKED_INPUT", "False")

    definition_key = f"{uuid4()}/invocations.json"

    s3_client = get_s3_client()
    s3_client.upload_fileobj(
        Fileobj=io.BytesIO(json.dumps(tasks).encode("utf-8")),
        Bucket=minio.input_bucket_name,
        Key=definition_key,
    )

    runner = CliRunner()
    runner.invoke(
        cli,
        ["invoke", "-f", f"s3://{minio.input_bucket_name}/{definition_key}"],
    )

    for pk in pk1, pk2:
        serialised_invocation = get_s3_file_content(
            s3_uri=f"s3://{minio.output_bucket_name}/tasks/{pk}/.sagemaker_shim/inference_result.json"
        )
        parsed_result = json.loads(serialised_invocation)

        assert parsed_result["return_code"] == expected_return_code
        assert parsed_result["pk"] == pk


def test_logging_setup(minio, monkeypatch):
    pk = str(uuid4())
    tasks = [
        {
            "pk": pk,
            "inputs": [],
            "output_bucket_name": minio.output_bucket_name,
            "output_prefix": f"tasks/{pk}",
        }
    ]

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_CMD_B64J",
        encode_b64j(val=["echo", "hello"]),
    )
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_SET_EXTRA_GROUPS", "False")
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USE_LINKED_INPUT", "False")

    runner = CliRunner()
    result = runner.invoke(cli, ["invoke", "-t", json.dumps(tasks)])

    assert (
        '{"log": "hello", "level": "INFO", '
        f'"source": "stdout", "internal": false, "task": "{pk}"}}'
    ) in result.output
    assert "Setting up Auxiliary Data" in result.output
    assert "Cleaning up Auxiliary Data" in result.output


def test_logging_stderr_setup(minio, monkeypatch):
    pk = str(uuid4())
    tasks = [
        {
            "pk": pk,
            "inputs": [],
            "output_bucket_name": minio.output_bucket_name,
            "output_prefix": f"tasks/{pk}",
        }
    ]

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_CMD_B64J",
        encode_b64j(
            val=["bash", "-c", "echo 'hello' >> /dev/stderr && exit 1"]
        ),
    )
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_SET_EXTRA_GROUPS", "False")
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USE_LINKED_INPUT", "False")

    runner = CliRunner()
    result = runner.invoke(cli, ["invoke", "-t", json.dumps(tasks)])

    assert (
        '{"log": "hello", "level": "WARNING", '
        f'"source": "stderr", "internal": false, "task": "{pk}"}}'
    ) in result.output


def test_memory_limit_undefined(minio, monkeypatch):
    pk = str(uuid4())
    tasks = [
        {
            "pk": pk,
            "inputs": [],
            "output_bucket_name": minio.output_bucket_name,
            "output_prefix": f"tasks/{pk}",
        }
    ]

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_CMD_B64J",
        encode_b64j(val=["echo", "hello"]),
    )
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_SET_EXTRA_GROUPS", "False")

    runner = CliRunner()
    result = runner.invoke(cli, ["invoke", "-t", json.dumps(tasks)])

    assert (
        '{"log": "Not setting a memory limit", "level": "INFO", '
        '"source": "stdout", "internal": true, "task": null}'
    ) in result.output


def test_memory_limit_defined(minio, monkeypatch):
    pk = str(uuid4())
    tasks = [
        {
            "pk": pk,
            "inputs": [],
            "output_bucket_name": minio.output_bucket_name,
            "output_prefix": f"tasks/{pk}",
        }
    ]

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_CMD_B64J",
        encode_b64j(val=["echo", "hello"]),
    )
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_SET_EXTRA_GROUPS", "False")
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_MAX_MEMORY_MB", "1337")

    expected_limit = 1337 * 1024 * 1024

    with patch("resource.setrlimit") as mock_setrlimit:
        runner = CliRunner()
        result = runner.invoke(cli, ["invoke", "-t", json.dumps(tasks)])

        mock_setrlimit.assert_called_once_with(
            resource.RLIMIT_DATA, (expected_limit, expected_limit)
        )

    assert (
        '{"log": "Setting memory limit to 1337 MB", "level": "INFO", '
        '"source": "stdout", "internal": true, "task": null}'
    ) in result.output
