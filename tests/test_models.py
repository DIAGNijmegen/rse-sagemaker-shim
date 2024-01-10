import getpass
import grp
import os
import pwd

import pytest

from sagemaker_shim.models import (
    InferenceTask,
    _get_users_groups,
    _put_gid_first,
    validate_bucket_name,
)


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


ROOT_HOME = pwd.getpwnam("root").pw_dir
ROOT_GROUPS = _get_users_groups(user=pwd.getpwnam("root"))
USER_HOME = os.path.expanduser("~")
USER_GROUPS = _get_users_groups(user=pwd.getpwnam(getpass.getuser()))


@pytest.mark.parametrize(
    "user,expected_user,expected_group,expected_home,expected_extra_groups",
    (
        ("0", 0, 0, ROOT_HOME, ROOT_GROUPS),
        ("0:0", 0, 0, ROOT_HOME, ROOT_GROUPS),
        (":0", None, 0, None, [0]),
        ("", None, None, None, []),
        ("root", 0, 0, ROOT_HOME, ROOT_GROUPS),
        (f"root:{grp.getgrgid(0).gr_name}", 0, 0, ROOT_HOME, ROOT_GROUPS),
        (f":{grp.getgrgid(0).gr_name}", None, 0, None, [0]),
        ("root:0", 0, 0, ROOT_HOME, ROOT_GROUPS),
        (f"0:{grp.getgrgid(0).gr_name}", 0, 0, ROOT_HOME, ROOT_GROUPS),
        (f":{os.getgid()}", None, os.getgid(), None, [os.getgid()]),
        (
            f"root:{os.getgid()}",
            0,
            os.getgid(),
            ROOT_HOME,
            _put_gid_first(gid=os.getgid(), groups=ROOT_GROUPS),
        ),
        # User exists
        (f"{os.getuid()}", os.getuid(), os.getgid(), USER_HOME, USER_GROUPS),
        (
            f"{getpass.getuser()}",
            os.getuid(),
            os.getgid(),
            USER_HOME,
            USER_GROUPS,
        ),
        # Group does not exist, but is an int
        (
            f"{os.getuid()}:23746",
            os.getuid(),
            23746,
            USER_HOME,
            _put_gid_first(gid=23746, groups=USER_GROUPS),
        ),
        (
            f"{getpass.getuser()}:23746",
            os.getuid(),
            23746,
            USER_HOME,
            _put_gid_first(gid=23746, groups=USER_GROUPS),
        ),
        # User does not exist, but is an int
        ("23746", 23746, None, None, []),
        (f"23746:{grp.getgrgid(0).gr_name}", 23746, 0, None, [0]),
        (f"23746:{os.getgid()}", 23746, os.getgid(), None, [os.getgid()]),
        # User and group do not exist, but are ints
        ("23746:23746", 23746, 23746, None, [23746]),
    ),
)
def test_proc_user(
    monkeypatch,
    user,
    expected_user,
    expected_group,
    expected_home,
    expected_extra_groups,
):
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USER", user)

    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )

    assert t.user == user
    assert t.proc_user.uid == expected_user
    assert t.proc_user.gid == expected_group
    assert t.proc_user.home == expected_home
    assert t.extra_groups == expected_extra_groups
    assert None not in t.extra_groups


def test_put_gid_first():
    # Setting None leaves the groups unmodified
    assert _put_gid_first(gid=None, groups=[2, 1, 3, 4]) == [2, 1, 3, 4]
    # Setting an existing group puts it first and orders the rest
    assert _put_gid_first(gid=3, groups=[2, 1, 3, 4]) == [3, 1, 2, 4]
    # Adding a group puts it first and orders the rest
    assert _put_gid_first(gid=5, groups=[2, 1, 3, 4]) == [5, 1, 2, 3, 4]


# Should error
@pytest.mark.parametrize(
    "user,expected_error",
    (
        (
            f"{os.getuid()}:nonExistentGroup",
            "Group 'nonExistentGroup' not found",
        ),
        (
            f"{getpass.getuser()}:nonExistentGroup",
            "Group 'nonExistentGroup' not found",
        ),
        ("nonExistentUser", "User 'nonExistentUser' not found"),
        (
            "nonExistentUser:nonExistentGroup",
            "User 'nonExistentUser' not found",
        ),
        (":nonExistentGroup", "Group 'nonExistentGroup' not found"),
        (
            f"nonExistentUser:{grp.getgrgid(0).gr_name}",
            "User 'nonExistentUser' not found",
        ),
        (f"nonExistentUser:{os.getgid()}", "User 'nonExistentUser' not found"),
        ("🙈:🙉", "Invalid user '🙈:🙉'"),
        ("🙈", "Invalid user '🙈'"),
        (":🙉", "Invalid user ':🙉'"),
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
