import getpass
import grp
import io
import os
import pwd
import tarfile
from pathlib import Path
from uuid import uuid4

import pytest

from sagemaker_shim.models import (
    AuxiliaryData,
    InferenceTask,
    ProcUserMixin,
    ProcUserTarfile,
    get_s3_client,
    validate_bucket_name,
)


@pytest.fixture
def algorithm_model():
    model_f = io.BytesIO()
    with tarfile.open(fileobj=model_f, mode="w:gz") as tar:
        content = b"Hello, World!"
        file_info = tarfile.TarInfo("model-file1.txt")
        file_info.size = len(content)
        tar.addfile(file_info, io.BytesIO(content))

        file_info = tarfile.TarInfo("model-sub/model-file2.txt")
        file_info.size = len(content)
        tar.addfile(file_info, io.BytesIO(content))

    model_f.seek(0)

    return model_f


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
ROOT_GROUPS = ProcUserMixin._get_users_groups(user=pwd.getpwnam("root"))
USER_HOME = os.path.expanduser("~")
USER_GROUPS = ProcUserMixin._get_users_groups(
    user=pwd.getpwnam(getpass.getuser())
)


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
            ProcUserMixin._put_gid_first(gid=os.getgid(), groups=ROOT_GROUPS),
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
            ProcUserMixin._put_gid_first(gid=23746, groups=USER_GROUPS),
        ),
        (
            f"{getpass.getuser()}:23746",
            os.getuid(),
            23746,
            USER_HOME,
            ProcUserMixin._put_gid_first(gid=23746, groups=USER_GROUPS),
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
    algorithm_model,
):
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USER", user)

    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )

    assert t._user == user
    assert t.proc_user.uid == expected_user
    assert t.proc_user.gid == expected_group
    assert t.proc_user.home == expected_home
    assert t.extra_groups == expected_extra_groups
    assert None not in t.extra_groups

    with ProcUserTarfile.open(fileobj=algorithm_model, mode="r") as tar:
        assert tar.proc_user == t.proc_user


def test_put_gid_first():
    # Setting None leaves the groups unmodified
    assert ProcUserMixin._put_gid_first(gid=None, groups=[2, 1, 3, 4]) == [
        2,
        1,
        3,
        4,
    ]
    # Setting an existing group puts it first and orders the rest
    assert ProcUserMixin._put_gid_first(gid=3, groups=[2, 1, 3, 4]) == [
        3,
        1,
        2,
        4,
    ]
    # Adding a group puts it first and orders the rest
    assert ProcUserMixin._put_gid_first(gid=5, groups=[2, 1, 3, 4]) == [
        5,
        1,
        2,
        3,
        4,
    ]


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

    assert t._user == user

    with pytest.raises(RuntimeError) as error:
        _ = t.proc_user

    assert str(error.value) == expected_error


def test_proc_user_unset(algorithm_model):
    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )

    assert t._user == ""
    assert t.proc_user.uid is None
    assert t.proc_user.gid is None
    assert t.proc_user.home is None

    with ProcUserTarfile.open(fileobj=algorithm_model, mode="r") as tar:
        assert tar.proc_user == t.proc_user


def test_home_is_set(monkeypatch):
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USER", "root")

    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )

    assert t.proc_env["HOME"] == pwd.getpwnam("root").pw_dir


def test_model_and_ground_truth_extraction(
    minio, monkeypatch, tmp_path, mocker, algorithm_model
):
    s3_client = get_s3_client()

    model_pk = str(uuid4())

    s3_client.upload_fileobj(
        algorithm_model, minio.input_bucket_name, f"{model_pk}/model.tar.gz"
    )

    ground_truth_pk = str(uuid4())

    ground_truth_f = io.BytesIO()
    with tarfile.open(fileobj=ground_truth_f, mode="w:gz") as tar:
        content = b"Hello, World!"
        file_info = tarfile.TarInfo("gt-file1.txt")
        file_info.size = len(content)
        tar.addfile(file_info, io.BytesIO(content))

        file_info = tarfile.TarInfo("gt-sub/gt-file2.txt")
        file_info.size = len(content)
        tar.addfile(file_info, io.BytesIO(content))

    ground_truth_f.seek(0)

    s3_client.upload_fileobj(
        ground_truth_f,
        minio.input_bucket_name,
        f"{ground_truth_pk}/ground_truth.tar.gz",
    )

    model_destination = tmp_path / "model"
    ground_truth_destination = tmp_path / "ground_truth"

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_MODEL",
        f"s3://{minio.input_bucket_name}/{model_pk}/model.tar.gz",
    )
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_MODEL_DEST", str(model_destination)
    )
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_GROUND_TRUTH",
        f"s3://{minio.input_bucket_name}/{ground_truth_pk}/ground_truth.tar.gz",
    )
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_GROUND_TRUTH_DEST",
        str(ground_truth_destination),
    )
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_POST_CLEAN_DIRECTORIES",
        f"{model_destination}:{ground_truth_destination}",
    )

    spy = mocker.spy(ProcUserTarfile, "chown")

    with AuxiliaryData():
        downloaded_files = {
            str(f.relative_to(tmp_path))
            for f in tmp_path.rglob("**/*")
            if f.is_file()
        }

    assert downloaded_files == {
        "model/model-file1.txt",
        "model/model-sub/model-file2.txt",
        "ground_truth/gt-file1.txt",
        "ground_truth/gt-sub/gt-file2.txt",
    }

    # Files should be cleaned up
    assert {str(f.relative_to(tmp_path)) for f in tmp_path.rglob("**/*")} == {
        "model",
        "ground_truth",
    }

    # We cannot test chown as you need to be root, but we can test that it was called
    assert spy.call_count == 4


def test_ensure_directories_are_writable_unset():
    with AuxiliaryData() as d:
        assert d.writable_directories == []
        assert d.post_clean_directories == []
        assert d.model_source is None
        assert d.model_dest == Path("/opt/ml/model")
        assert not d.model_dest.exists()
        assert d.ground_truth_source is None
        assert d.ground_truth_dest == Path("/opt/ml/input/data/ground_truth")
        assert not d.ground_truth_dest.exists()


@pytest.mark.parametrize(
    "directories,expected",
    (
        ("", []),
        (":", []),
        ("::", []),
    ),
)
def test_ensure_directories_are_writable_set(
    monkeypatch, directories, expected
):
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_WRITABLE_DIRECTORIES",
        directories,
    )

    d = AuxiliaryData()
    assert d.writable_directories == expected


def test_ensure_directories_are_writable(tmp_path, monkeypatch):
    data = tmp_path / "opt" / "ml" / "output" / "data"
    data.mkdir(mode=0o755, parents=True)

    model = tmp_path / "opt" / "ml" / "model"
    model.mkdir(mode=0o755, parents=True)

    # Do not create the checkpoints dir in the test
    checkpoints = tmp_path / "opt" / "ml" / "checkpoints"

    tmp = tmp_path / "tmp"
    tmp.mkdir(mode=0o755)

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_WRITABLE_DIRECTORIES",
        f"{data.absolute()}:{model.absolute()}:{checkpoints.absolute()}:{tmp.absolute()}",
    )

    with AuxiliaryData():
        pass

    assert data.stat().st_mode == 0o40777
    assert model.stat().st_mode == 0o40777
    assert checkpoints.stat().st_mode == 0o40777
    assert tmp.stat().st_mode == 0o40777


def test_linked_input_path_default():
    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )

    assert t.linked_input_path == Path("/opt/ml/input/data/test-input")


def test_linked_input_path_setting(monkeypatch):
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_LINKED_INPUT_PARENT", "/foo")

    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )

    assert t.linked_input_path == Path("/foo/test-input")


def test_reset_linked_input(tmp_path, monkeypatch):
    input_path = tmp_path / "input"
    linked_input_parent = tmp_path / "linked-input"

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_INPUT_PATH", input_path.absolute()
    )
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_LINKED_INPUT_PARENT", linked_input_parent
    )

    t = InferenceTask(
        pk="test", inputs=[], output_bucket_name="test", output_prefix="test"
    )
    t.reset_io()

    expected_input_directory = linked_input_parent / "test-input"

    assert input_path.exists()
    assert input_path.is_symlink()
    assert expected_input_directory.exists()
    assert expected_input_directory.is_dir()
    assert input_path.resolve(strict=True) == expected_input_directory

    # Ensure 0o755 permissions
    assert os.stat(input_path).st_mode & 0o777 == 0o755
    assert os.stat(expected_input_directory).st_mode & 0o777 == 0o755
