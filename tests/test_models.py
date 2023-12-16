import grp
import os
import pwd

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

    assert str(error).startswith(
        "<ExceptionInfo 1 validation error for InferenceTask\n"
        "output_prefix\n  Value error, Prefix cannot be blank "
        "[type=value_error, input_value='', input_type=str]\n"
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


@pytest.mark.parametrize(
    "user,expected_user,expected_group",
    (
        ("0", 0, 0),
        ("0:0", 0, 0),
        (":0", None, 0),
        ("", None, None),
        ("root", 0, 0),
        (f"root:{grp.getgrgid(0).gr_name}", 0, 0),
        (f":{grp.getgrgid(0).gr_name}", None, 0),
        ("", None, None),
        ("🙈:🙉", None, None),
        ("root:0", 0, 0),
        (f"0:{grp.getgrgid(0).gr_name}", 0, 0),
        (f":{os.getgid()}", None, os.getgid()),
        (f"root:{os.getgid()}", 0, os.getgid()),
    ),
)
def test_proc_user(monkeypatch, user, expected_user, expected_group):
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USER", user)

    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )

    assert t.user == user
    assert t.proc_user.uid == expected_user
    assert t.proc_user.gid == expected_group


def test_proc_user_unset():
    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )

    assert t.user == ""
    assert t.proc_user.uid is None
    assert t.proc_user.gid is None


def test_home_is_set(monkeypatch):
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USER", "root")

    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )

    assert t.proc_env["HOME"] == pwd.getpwnam("root").pw_dir
