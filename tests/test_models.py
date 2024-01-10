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
    "user,expected_user,expected_group,expected_home",
    (
        ("0", 0, 0, pwd.getpwnam("root").pw_dir),
        ("0:0", 0, 0, pwd.getpwnam("root").pw_dir),
        (":0", None, 0, None),
        ("", None, None, None),
        ("root", 0, 0, pwd.getpwnam("root").pw_dir),
        (f"root:{grp.getgrgid(0).gr_name}", 0, 0, pwd.getpwnam("root").pw_dir),
        (f":{grp.getgrgid(0).gr_name}", None, 0, None),
        ("🙈:🙉", None, None, None),
        ("root:0", 0, 0, pwd.getpwnam("root").pw_dir),
        (f"0:{grp.getgrgid(0).gr_name}", 0, 0, pwd.getpwnam("root").pw_dir),
        (f":{os.getgid()}", None, os.getgid(), None),
        (f"root:{os.getgid()}", 0, os.getgid(), pwd.getpwnam("root").pw_dir),
        # User exists
        (f"{os.getuid()}", os.getuid(), os.getgid(), os.path.expanduser("~")),
        (
            f"{os.getlogin()}",
            os.getuid(),
            os.getgid(),
            os.path.expanduser("~"),
        ),
        # Group does not exist, but is an int
        (f"{os.getuid()}:23746", os.getuid(), 23746, os.path.expanduser("~")),
        (
            f"{os.getlogin()}:23746",
            os.getuid(),
            23746,
            os.path.expanduser("~"),
        ),
        # User does not exist, but is an int
        ("23746", 23746, None, None),
        (f"23746:{grp.getgrgid(0).gr_name}", 23746, 0, None),
        (f"23746:{os.getgid()}", 23746, os.getgid(), None),
        # User and group do not exist, but are ints
        ("23746:23746", 23746, 23746, None),
    ),
)
def test_proc_user(
    monkeypatch, user, expected_user, expected_group, expected_home
):
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USER", user)

    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )

    assert t.user == user
    assert t.proc_user.uid == expected_user
    assert t.proc_user.gid == expected_group
    assert t.proc_user.home == expected_home


# Should error
@pytest.mark.parametrize(
    "user,expected_error",
    (
        (
            f"{os.getuid()}:nonExistantGroup",
            "Group 'nonExistantGroup' not found",
        ),
        (
            f"{os.getlogin()}:nonExistantGroup",
            "Group 'nonExistantGroup' not found",
        ),
        ("nonExistantUser", "User 'nonExistantUser' not found"),
        (
            "nonExistantUser:nonExistantGroup",
            "User 'nonExistantUser' not found",
        ),
        (":nonExistantGroup", "Group 'nonExistantGroup' not found"),
        (
            f"nonExistantUser:{grp.getgrgid(0).gr_name}",
            "User 'nonExistantUser' not found",
        ),
        (f"nonExistantUser:{os.getgid()}", "User 'nonExistantUser' not found"),
    ),
)
def test_proc_user_errors(monkeypatch, user, expected_error):
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USER", user)

    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )

    assert t.user == user

    with pytest.raises(RuntimeError) as error:
        _ = t.proc_user

    assert str(error.value) == expected_error


def test_proc_user_unset():
    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )

    assert t.user == ""
    assert t.proc_user.uid is None
    assert t.proc_user.gid is None
    assert t.proc_user.home is None


def test_home_is_set(monkeypatch):
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USER", "root")

    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )

    assert t.proc_env["HOME"] == pwd.getpwnam("root").pw_dir
