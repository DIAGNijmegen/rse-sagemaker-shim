import io
import json
import logging.config
import os
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

import pytest

from sagemaker_shim.logging import LOGGING_CONFIG
from sagemaker_shim.models import (
    InferenceIO,
    InferenceResult,
    InferenceTask,
    clean_path,
    get_s3_client,
)
from tests.utils import encode_b64j


def test_input_download(minio, tmp_path, monkeypatch):
    pk = str(uuid4())
    prefix = f"tasks/{pk}"
    task = InferenceTask(
        pk=pk,
        inputs=[
            InferenceIO(
                bucket_name=minio.input_bucket_name,
                bucket_key=f"{prefix}/root.bin",
                relative_path=Path("root.bin"),
            ),
            InferenceIO(
                bucket_name=minio.input_bucket_name,
                bucket_key=f"{prefix}/sub/dir.bin",
                relative_path=Path("sub/dir.bin"),
            ),
        ],
        output_bucket_name=minio.output_bucket_name,
        output_prefix=str(prefix),
    )

    # Prep input bucket
    root_data = os.urandom(8)
    root_f = io.BytesIO(root_data)
    task._s3_client.upload_fileobj(
        root_f, minio.input_bucket_name, f"{prefix}/root.bin"
    )

    sub_data = os.urandom(8)
    sub_f = io.BytesIO(sub_data)
    task._s3_client.upload_fileobj(
        sub_f, minio.input_bucket_name, f"{prefix}/sub/dir.bin"
    )

    response = task._s3_client.list_objects_v2(
        Bucket=minio.input_bucket_name,
        Prefix=prefix,
    )
    assert {f["Key"] for f in response["Contents"]} == {
        f"tasks/{task.pk}/root.bin",
        f"tasks/{task.pk}/sub/dir.bin",
    }

    # Download the data
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_INPUT_PATH",
        str(tmp_path),
    )
    task.download_input()

    # Check
    with open(tmp_path / "root.bin", "rb") as f:
        created_root = f.read()

    with open(tmp_path / "sub" / "dir.bin", "rb") as f:
        created_sub = f.read()

    assert created_root == root_data
    assert created_sub == sub_data


def test_input_decompress(minio, tmp_path, monkeypatch):
    pk = str(uuid4())
    prefix = f"tasks/{pk}"
    task = InferenceTask(
        pk=pk,
        inputs=[
            InferenceIO(
                bucket_name=minio.input_bucket_name,
                bucket_key=f"{prefix}/sub/predictions.zip",
                relative_path=Path("sub/predictions.zip"),
                decompress=True,
            ),
        ],
        output_bucket_name=minio.output_bucket_name,
        output_prefix=str(prefix),
    )

    # Prep input bucket
    sub_f = io.BytesIO()
    with ZipFile(file=sub_f, mode="w") as zip:
        zip.writestr("sdsdaf/test.txt", str(pk))
    sub_f.seek(0)
    task._s3_client.upload_fileobj(
        sub_f, minio.input_bucket_name, f"{prefix}/sub/predictions.zip"
    )

    response = task._s3_client.list_objects_v2(
        Bucket=minio.input_bucket_name,
        Prefix=prefix,
    )
    assert {f["Key"] for f in response["Contents"]} == {
        f"tasks/{task.pk}/sub/predictions.zip",
    }

    # Download the data
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_INPUT_PATH",
        str(tmp_path),
    )
    task.download_input()

    # Check
    with open(tmp_path / "sub" / "test.txt") as f:
        created_sub = f.read()

    assert created_sub == str(pk)


def test_invoke_with_dodgy_file(client, minio, tmp_path, monkeypatch, capsys):
    s3_client = get_s3_client()
    pk = str(uuid4())
    prefix = f"tasks/{pk}"
    data = {
        "pk": pk,
        "inputs": [
            {
                "bucket_name": minio.input_bucket_name,
                "bucket_key": f"{prefix}/sub/dodgy.zip",
                "relative_path": "sub/dodgy.zip",
                "decompress": True,
            }
        ],
        "output_bucket_name": minio.output_bucket_name,
        "output_prefix": prefix,
    }

    input_path = tmp_path / "input"
    output_path = tmp_path / "output"

    input_path.mkdir()
    output_path.mkdir()

    # Prep input bucket
    sub_f = io.BytesIO()
    with ZipFile(file=sub_f, mode="w") as zip:
        zip.writestr("../foo.txt", "hello!")
        zip.writestr("../../foo.txt", "hello!")
    sub_f.seek(0)
    s3_client.upload_fileobj(
        sub_f, minio.input_bucket_name, f"{prefix}/sub/dodgy.zip"
    )

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_INPUT_PATH",
        str(input_path),
    )

    logging.config.dictConfig(LOGGING_CONFIG)

    response = client.post("/invocations", json=data)

    response = response.json()
    assert response["return_code"] == 1

    captured = capsys.readouterr()
    assert captured.err == (
        '{"log": "Zip file contains invalid paths", "level": "ERROR", '
        f'"source": "stderr", "internal": false, "task": "{pk}"}}\n'
    )


def test_invoke_with_non_zip(client, minio, tmp_path, monkeypatch, capsys):
    s3_client = get_s3_client()
    pk = str(uuid4())
    prefix = f"tasks/{pk}"
    data = {
        "pk": pk,
        "inputs": [
            {
                "bucket_name": minio.input_bucket_name,
                "bucket_key": f"{prefix}/sub/dodgy.zip",
                "relative_path": "sub/dodgy.zip",
                "decompress": True,
            }
        ],
        "output_bucket_name": minio.output_bucket_name,
        "output_prefix": prefix,
    }

    input_path = tmp_path / "input"
    output_path = tmp_path / "output"

    input_path.mkdir()
    output_path.mkdir()

    # Prep input bucket
    sub_data = os.urandom(8)
    sub_f = io.BytesIO(sub_data)
    sub_f.seek(0)
    s3_client.upload_fileobj(
        sub_f, minio.input_bucket_name, f"{prefix}/sub/dodgy.zip"
    )

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_INPUT_PATH",
        str(input_path),
    )

    logging.config.dictConfig(LOGGING_CONFIG)

    response = client.post("/invocations", json=data)

    response = response.json()
    assert response["return_code"] == 1

    captured = capsys.readouterr()
    assert captured.err == (
        '{"log": "Input zip file could not be extracted", "level": "ERROR", '
        f'"source": "stderr", "internal": false, "task": "{pk}"}}\n'
    )


def test_output_upload(minio, tmp_path, monkeypatch):
    s3_client = get_s3_client()
    pk = str(uuid4())
    prefix = f"tasks/{pk}"
    task = InferenceTask(
        pk=pk,
        inputs=[],
        output_bucket_name=minio.output_bucket_name,
        output_prefix=str(prefix),
    )

    # Prep output bucket
    root_data = os.urandom(8)
    with open(tmp_path / "root.bin", "wb") as f:
        f.write(root_data)

    sub_data = os.urandom(8)
    (tmp_path / "sub").mkdir()
    with open(tmp_path / "sub" / "dir.bin", "wb") as f:
        f.write(sub_data)

    # Upload the data
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_OUTPUT_PATH",
        str(tmp_path),
    )
    task.upload_output()

    response = task._s3_client.list_objects_v2(
        Bucket=minio.output_bucket_name,
        Prefix=prefix,
    )
    assert {f["Key"] for f in response["Contents"]} == {
        f"tasks/{task.pk}/root.bin",
        f"tasks/{task.pk}/sub/dir.bin",
    }

    root_f = io.BytesIO()
    s3_client.download_fileobj(
        Fileobj=root_f,
        Bucket=minio.output_bucket_name,
        Key=f"tasks/{task.pk}/root.bin",
    )
    root_f.seek(0)

    sub_f = io.BytesIO()
    s3_client.download_fileobj(
        Fileobj=sub_f,
        Bucket=minio.output_bucket_name,
        Key=f"tasks/{task.pk}/sub/dir.bin",
    )
    sub_f.seek(0)

    assert root_f.read() == root_data
    assert sub_f.read() == sub_data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cmd,expected_return_code",
    (
        (["echo", "hello"], 0),
        (["bash", "-c", "exit 1"], 1),
    ),
)
async def test_inference_result_upload(
    minio, tmp_path, monkeypatch, cmd, expected_return_code
):
    pk = str(uuid4())
    prefix = f"tasks/{pk}"
    task = InferenceTask(
        pk=pk,
        inputs=[],
        output_bucket_name=minio.output_bucket_name,
        output_prefix=str(prefix),
    )

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_CMD_B64J",
        encode_b64j(val=cmd),
    )
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_SET_EXTRA_GROUPS", "False")
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USE_LINKED_INPUT", "False")

    direct_invocation = await task.invoke()

    assert direct_invocation.return_code == expected_return_code
    assert direct_invocation.pk == pk

    serialised_invocation = io.BytesIO()
    task._s3_client.download_fileobj(
        Fileobj=serialised_invocation,
        Bucket=minio.output_bucket_name,
        Key=f"tasks/{pk}/.sagemaker_shim/inference_result.json",
    )
    serialised_invocation.seek(0)

    assert direct_invocation == InferenceResult(
        **json.loads(serialised_invocation.read().decode("utf-8"))
    )


def test_folder_cleanup(tmp_path):
    for dir in ["test", "nested/test"]:
        (tmp_path / dir).mkdir(parents=True)

    for f in ["test", ".test", "test/test", "test/.test", "nested/test/.test"]:
        (tmp_path / f).touch()

    clean_path(path=tmp_path)

    assert [*tmp_path.rglob("**/*")] == []
