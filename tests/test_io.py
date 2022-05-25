import io
import os
from contextlib import contextmanager
from pathlib import Path
from time import sleep
from typing import NamedTuple
from uuid import uuid4

import boto3
import docker
import pytest
from docker.models.containers import Container

from sagemaker_shim.models import InferenceIO, InferenceTask


class Minio(NamedTuple):
    input_bucket_name: str
    output_bucket_name: str
    container: Container
    port: int


@contextmanager
def minio_container():
    input_bucket_name = "test-inputs"
    output_bucket_name = "test-outputs"

    client = docker.from_env()
    minio = client.containers.run(
        image="minio/minio:latest",
        entrypoint="/bin/sh",
        command=[
            "-c",
            f"mkdir -p /data/{input_bucket_name} /data/{output_bucket_name} "
            "&& minio --compat server /data",
        ],
        ports={9000: None},
        auto_remove=True,
        detach=True,
        init=True,
    )

    # Wait for startup
    sleep(1)

    mpatch = pytest.MonkeyPatch()

    try:
        minio.reload()  # required to get ports
        port = minio.ports["9000/tcp"][0]["HostPort"]

        mpatch.setenv("AWS_ACCESS_KEY_ID", "minioadmin")
        mpatch.setenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
        mpatch.setenv("AWS_S3_ENDPOINT_URL", f"http://localhost:{port}")

        yield Minio(
            input_bucket_name=input_bucket_name,
            output_bucket_name=output_bucket_name,
            container=minio,
            port=port,
        )
    finally:
        mpatch.undo()
        minio.stop(timeout=0)


@pytest.fixture(scope="session")
def minio():
    with minio_container() as m:
        yield m


def test_input_download(minio, tmp_path, monkeypatch):
    s3_client = boto3.client(
        "s3", endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL")
    )
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
    s3_client.upload_fileobj(
        root_f, minio.input_bucket_name, f"{prefix}/root.bin"
    )

    sub_data = os.urandom(8)
    sub_f = io.BytesIO(sub_data)
    s3_client.upload_fileobj(
        sub_f, minio.input_bucket_name, f"{prefix}/sub/dir.bin"
    )

    response = s3_client.list_objects_v2(
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


def test_output_upload(minio, tmp_path, monkeypatch):
    s3_client = boto3.client(
        "s3", endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL")
    )
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

    response = s3_client.list_objects_v2(
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
