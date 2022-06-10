import os

import pytest

from sagemaker_shim.models import InferenceTask, validate_bucket_name


def test_invalid_bucket_name():
    with pytest.raises(ValueError):
        validate_bucket_name("$!#")


def test_blank_prefix():
    with pytest.raises(ValueError) as error:
        InferenceTask(
            pk="test", inputs=[], output_bucket_name="test", output_prefix=""
        )

    assert str(error) == (
        "<ExceptionInfo ValidationError(model='InferenceTask', "
        "errors=[{'loc': ('output_prefix',), "
        "'msg': 'Prefix cannot be blank', 'type': 'value_error'}]) tblen=2>"
    )


def test_prefix_slash_appended():
    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )
    assert t.output_prefix == "test/"


def test_patching_ld_library_path(monkeypatch):
    """Subprocess must run with the original LD_LIBRARY_PATH set"""
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "special")

    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )

    env = os.environ.copy()

    assert env["LD_LIBRARY_PATH_ORIG"] == "special"
    assert "LD_LIBRARY_PATH" not in env
    assert t.proc_env["LD_LIBRARY_PATH"] == "special"


def test_removing_ld_library_path(monkeypatch):
    """Subprocess must have LD_LIBRARY_PATH removed if set"""
    monkeypatch.setenv("LD_LIBRARY_PATH", "present")

    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )

    env = os.environ.copy()

    assert "LD_LIBRARY_PATH" not in t.proc_env
    assert env["LD_LIBRARY_PATH"] == "present"
