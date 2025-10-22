import getpass
import grp
import io
import logging.config
import os
import pwd
import tarfile
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from sagemaker_shim.exceptions import UserSafeError
from sagemaker_shim.logging import LOGGING_CONFIG
from sagemaker_shim.models import (
    AuxiliaryData,
    InferenceTask,
    ProcUserMixin,
    ProcUserTarfile,
    clean_path,
    get_s3_resources,
    validate_bucket_name,
)
from tests.utils import encode_b64j


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
            pk="test",
            inputs=[],
            output_bucket_name="test",
            output_prefix="",
            timeout=timedelta(),
        )

    assert str(error).startswith(
        "<ExceptionInfo 1 validation error for InferenceTask\n"
        "output_prefix\n  Value error, Prefix cannot be blank "
        "[type=value_error, input_value='', input_type=str]\n"
    )


def test_prefix_slash_appended():
    t = InferenceTask(
        pk="test",
        inputs=[],
        output_bucket_name="test",
        output_prefix="test",
        timeout=timedelta(),
    )
    assert t.output_prefix == "test/"


def test_patching_ld_library_path(monkeypatch):
    """Subprocess must run with the original LD_LIBRARY_PATH set"""
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "special")

    t = InferenceTask(
        pk="test",
        inputs=[],
        output_bucket_name="test",
        output_prefix="test",
        timeout=timedelta(),
    )

    env = os.environ.copy()

    assert env["LD_LIBRARY_PATH_ORIG"] == "special"
    assert "LD_LIBRARY_PATH" not in env
    assert t.proc_env["LD_LIBRARY_PATH"] == "special"


def test_removing_ld_library_path(monkeypatch):
    """Subprocess must have LD_LIBRARY_PATH removed if set"""
    monkeypatch.setenv("LD_LIBRARY_PATH", "present")

    t = InferenceTask(
        pk="test",
        inputs=[],
        output_bucket_name="test",
        output_prefix="test",
        timeout=timedelta(),
    )

    env = os.environ.copy()

    assert "LD_LIBRARY_PATH" not in t.proc_env
    assert env["LD_LIBRARY_PATH"] == "present"


def test_all_grand_challenge_env_vars_removed(monkeypatch):
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_SIGNING_KEY_HEX", "somekey")
    monkeypatch.setenv("grand_challenge_foo", "bar")

    t = InferenceTask(
        pk="test",
        inputs=[],
        output_bucket_name="test",
        output_prefix="test",
        timeout=timedelta(),
    )

    env = os.environ.copy()

    assert env["GRAND_CHALLENGE_COMPONENT_SIGNING_KEY_HEX"] == "somekey"
    assert env["grand_challenge_foo"] == "bar"
    assert "GRAND_CHALLENGE_COMPONENT_SIGNING_KEY_HEX" not in t.proc_env
    assert "grand_challenge_foo" not in t.proc_env


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
        pk="test",
        inputs=[],
        output_bucket_name="test",
        output_prefix="test",
        timeout=timedelta(),
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
            "The group defined in the containers USER instruction does not exist",
        ),
        (
            f"{getpass.getuser()}:nonExistentGroup",
            "The group defined in the containers USER instruction does not exist",
        ),
        (
            "nonExistentUser",
            "The user defined in the containers USER instruction does not exist",
        ),
        (
            "nonExistentUser:nonExistentGroup",
            "The user defined in the containers USER instruction does not exist",
        ),
        (
            ":nonExistentGroup",
            "The group defined in the containers USER instruction does not exist",
        ),
        (
            f"nonExistentUser:{grp.getgrgid(0).gr_name}",
            "The user defined in the containers USER instruction does not exist",
        ),
        (
            f"nonExistentUser:{os.getgid()}",
            "The user defined in the containers USER instruction does not exist",
        ),
        ("🙈:🙉", "Invalid argument for the containers USER instruction"),
        ("🙈", "Invalid argument for the containers USER instruction"),
        (":🙉", "Invalid argument for the containers USER instruction"),
    ),
)
def test_proc_user_errors(monkeypatch, user, expected_error):
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USER", user)

    t = InferenceTask(
        pk="test",
        inputs=[],
        output_bucket_name="test",
        output_prefix="test",
        timeout=timedelta(),
    )

    assert t._user == user

    with pytest.raises(UserSafeError) as error:
        _ = t.proc_user

    assert str(error.value) == expected_error


def test_proc_user_unset(algorithm_model):
    t = InferenceTask(
        pk="test",
        inputs=[],
        output_bucket_name="test",
        output_prefix="test",
        timeout=timedelta(),
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
        pk="test",
        inputs=[],
        output_bucket_name="test",
        output_prefix="test",
        timeout=timedelta(),
    )

    assert t.proc_env["HOME"] == pwd.getpwnam("root").pw_dir


@pytest.mark.asyncio
async def test_model_and_ground_truth_extraction(
    minio, monkeypatch, tmp_path, mocker, algorithm_model
):
    model_pk = str(uuid4())
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
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_WRITABLE_DIRECTORIES", "")

    spy = mocker.spy(ProcUserTarfile, "chown")

    async with get_s3_resources() as s3_resources:
        async with s3_resources.semaphore:
            await s3_resources.client.upload_fileobj(
                algorithm_model,
                minio.input_bucket_name,
                f"{model_pk}/model.tar.gz",
            )
            await s3_resources.client.upload_fileobj(
                ground_truth_f,
                minio.input_bucket_name,
                f"{ground_truth_pk}/ground_truth.tar.gz",
            )

        async with AuxiliaryData(s3_resources=s3_resources):
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


@pytest.mark.asyncio
async def test_ensure_directories_are_writable_unset(monkeypatch):
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_WRITABLE_DIRECTORIES", "")
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_POST_CLEAN_DIRECTORIES", "")

    async with get_s3_resources() as s3_resources:
        async with AuxiliaryData(s3_resources=s3_resources) as d:
            assert d.writable_directories == []
            assert d.post_clean_directories == []
            assert d.model_source is None
            assert d.model_dest == Path("/opt/ml/model")
            assert not d.model_dest.exists()
            assert d.ground_truth_source is None
            assert d.ground_truth_dest == Path(
                "/opt/ml/input/data/ground_truth"
            )
            assert not d.ground_truth_dest.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "directories,expected",
    (
        ("", []),
        (":", []),
        ("::", []),
    ),
)
async def test_ensure_directories_are_writable_set(
    monkeypatch, directories, expected
):
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_WRITABLE_DIRECTORIES",
        directories,
    )

    async with get_s3_resources() as s3_resources:
        async with AuxiliaryData(s3_resources=s3_resources) as d:
            assert d.writable_directories == expected


@pytest.mark.asyncio
async def test_ensure_directories_are_writable(tmp_path, monkeypatch):
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

    async with get_s3_resources() as s3_resources:
        async with AuxiliaryData(s3_resources=s3_resources):
            pass

    assert data.stat().st_mode == 0o40777
    assert model.stat().st_mode == 0o40777
    assert checkpoints.stat().st_mode == 0o40777
    assert tmp.stat().st_mode == 0o40777


def test_linked_input_path_default():
    t = InferenceTask(
        pk="test",
        inputs=[],
        output_bucket_name="test",
        output_prefix="test",
        timeout=timedelta(),
    )

    assert t.linked_input_path == Path("/opt/ml/input/data/test-input")


def test_linked_input_path_setting(monkeypatch):
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_LINKED_INPUT_PARENT", "/foo")

    t = InferenceTask(
        pk="test",
        inputs=[],
        output_bucket_name="test",
        output_prefix="test",
        timeout=timedelta(),
    )

    assert t.linked_input_path == Path("/foo/test-input")


def test_reset_linked_input(tmp_path, monkeypatch):
    input_path = tmp_path / "input"
    linked_input_parent = tmp_path / "linked-input"

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_INPUT_PATH", str(input_path.absolute())
    )
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_LINKED_INPUT_PARENT",
        str(linked_input_parent),
    )

    t = InferenceTask(
        pk="test",
        inputs=[],
        output_bucket_name="test",
        output_prefix="test",
        timeout=timedelta(),
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


@pytest.mark.asyncio
async def test_timeout(minio, monkeypatch, capsys):
    cmd = ["sleep", "10"]
    pk = str(uuid4())
    prefix = f"tasks/{pk}"
    task = InferenceTask(
        pk=pk,
        inputs=[],
        output_bucket_name=minio.output_bucket_name,
        output_prefix=str(prefix),
        timeout=timedelta(seconds=1),
    )

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_CMD_B64J",
        encode_b64j(val=cmd),
    )
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_SET_EXTRA_GROUPS", "False")
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USE_LINKED_INPUT", "False")

    logging.config.dictConfig(LOGGING_CONFIG)

    async with get_s3_resources() as s3_resources:
        result = await task.invoke(s3_resources=s3_resources)

    assert result.return_code == 1
    assert int(result.exec_duration.total_seconds()) == 1
    assert result.invoke_duration is None  # should only be set for invocation

    captured = capsys.readouterr()
    # "Time limit exceeded" must be the last log for the user error
    assert captured.err == (
        '{"log": "Time limit exceeded", "level": "ERROR", "source": "stderr", '
        f'"internal": false, "task": "{pk}"}}\n'
    )
    assert (
        '{"log": "Execution was cancelled", "level": "INFO", "source": "stdout", '
        '"internal": true, "task": null}' in captured.out
    )
    assert (
        '{"log": "Process group terminated", "level": "INFO", "source": "stdout", '
        '"internal": true, "task": null}' in captured.out
    )


@pytest.mark.asyncio
async def test_non_existent_user(minio, monkeypatch, capsys):
    cmd = ["sleep", "10"]
    pk = str(uuid4())
    prefix = f"tasks/{pk}"
    task = InferenceTask(
        pk=pk,
        inputs=[],
        output_bucket_name=minio.output_bucket_name,
        output_prefix=str(prefix),
        timeout=timedelta(seconds=1),
    )

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_CMD_B64J",
        encode_b64j(val=cmd),
    )
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_SET_EXTRA_GROUPS", "False")
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USE_LINKED_INPUT", "False")
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USER", "-gdfs")

    logging.config.dictConfig(LOGGING_CONFIG)

    async with get_s3_resources() as s3_resources:
        result = await task.invoke(s3_resources=s3_resources)

    assert result.return_code == 1
    assert int(result.exec_duration.total_seconds()) == 0
    assert result.invoke_duration is None  # should only be set for invocation

    captured = capsys.readouterr()
    # Invalid argument must be the last log for the user error
    assert captured.err == (
        '{"log": "Invalid argument for the containers USER instruction", '
        f'"level": "ERROR", "source": "stderr", "internal": false, "task": "{pk}"}}\n'
    )


@pytest.mark.asyncio
async def test_user_cmd_permission_denied(
    minio, monkeypatch, capsys, tmp_path
):
    test_file = tmp_path / "no_perms.sh"
    test_file.touch()

    cmd = [str(tmp_path / "no_perms.sh")]
    pk = str(uuid4())
    prefix = f"tasks/{pk}"
    task = InferenceTask(
        pk=pk,
        inputs=[],
        output_bucket_name=minio.output_bucket_name,
        output_prefix=str(prefix),
        timeout=timedelta(seconds=1),
    )

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_CMD_B64J",
        encode_b64j(val=cmd),
    )
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_SET_EXTRA_GROUPS", "False")
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USE_LINKED_INPUT", "False")

    logging.config.dictConfig(LOGGING_CONFIG)

    async with get_s3_resources() as s3_resources:
        result = await task.invoke(s3_resources=s3_resources)

    assert result.return_code == 1
    assert int(result.exec_duration.total_seconds()) == 0
    assert result.invoke_duration is None  # should only be set for invocation

    captured = capsys.readouterr()
    # No permission must be the last log for the user error
    assert captured.err == (
        '{"log": "The user defined in the containers USER instruction '
        "does not have permission to execute the command defined by "
        'the containers ENTRYPOINT and CMD instructions", "level": "ERROR", '
        f'"source": "stderr", "internal": false, "task": "{pk}"}}\n'
    )


@pytest.mark.asyncio
async def test_user_cmd_missing(minio, monkeypatch, capsys):
    cmd = ["doesnt_exist.sh"]
    pk = str(uuid4())
    prefix = f"tasks/{pk}"
    task = InferenceTask(
        pk=pk,
        inputs=[],
        output_bucket_name=minio.output_bucket_name,
        output_prefix=str(prefix),
        timeout=timedelta(seconds=1),
    )

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_CMD_B64J",
        encode_b64j(val=cmd),
    )
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_SET_EXTRA_GROUPS", "False")
    monkeypatch.setenv("GRAND_CHALLENGE_COMPONENT_USE_LINKED_INPUT", "False")

    logging.config.dictConfig(LOGGING_CONFIG)

    async with get_s3_resources() as s3_resources:
        result = await task.invoke(s3_resources=s3_resources)

    assert result.return_code == 1
    assert int(result.exec_duration.total_seconds()) == 0
    assert result.invoke_duration is None  # should only be set for invocation

    captured = capsys.readouterr()
    # Command not found must be the last log for the user error
    assert captured.err == (
        '{"log": "The command defined by the containers ENTRYPOINT '
        'and CMD instructions does not exist", "level": "ERROR", '
        f'"source": "stderr", "internal": false, "task": "{pk}"}}\n'
    )


def test_folder_cleanup_with_deleted_symlink_targets(tmp_path):
    dirty_path = tmp_path / "test"
    dirty_path.mkdir()

    # Create a target directory
    target_dir = tmp_path / "target-dir"
    target_dir.mkdir()

    # Create a symlink to the target directory
    dir_symlink = dirty_path / "to-target-dir"
    dir_symlink.symlink_to(target=target_dir, target_is_directory=True)

    # Create a target file
    target_file = tmp_path / "target-file"
    target_file.touch()

    # Create a symlink to the target file
    file_symlink = dirty_path / "to-target-file"
    file_symlink.symlink_to(target=target_file, target_is_directory=False)

    # Remove the targets so the symlinks do not point at anything
    target_dir.rmdir()
    target_file.unlink()

    clean_path(dirty_path)

    assert {*dirty_path.iterdir()} == set()
    assert dirty_path.exists()

    assert not target_dir.exists()
    assert not dir_symlink.exists()
    assert not target_file.exists()
    assert not file_symlink.exists()


def test_symlink_cleanup(tmp_path):
    dirty_path = tmp_path / "test"
    dirty_path.mkdir()

    # Create a target directory
    target_dir = tmp_path / "target-dir"
    target_dir.mkdir()

    # Create a symlink to the target directory
    dir_symlink = dirty_path / "to-target-dir"
    dir_symlink.symlink_to(target=target_dir, target_is_directory=True)

    # Create a target file
    target_file = tmp_path / "target-file"
    target_file.touch()

    # Create a symlink to the target file
    file_symlink = dirty_path / "to-target-file"
    file_symlink.symlink_to(target=target_file, target_is_directory=False)

    clean_path(dirty_path)

    assert {*dirty_path.iterdir()} == set()
    assert dirty_path.exists()

    assert target_dir.exists()
    assert not dir_symlink.exists()
    assert target_file.exists()
    assert not file_symlink.exists()


def test_circular_cleanup(tmp_path):
    dirty_path = tmp_path / "clean-me"
    dirty_path.mkdir()

    dirty_path_dir = dirty_path / "dirty-dir"
    dirty_path_dir.mkdir()

    dirty_path_file = dirty_path / "dirty-file"
    dirty_path_file.touch()

    # Create a target directory
    target_dir = tmp_path / "target-dir"
    target_dir.mkdir()

    target_dir_symlink_back = target_dir / "dirty-dir"
    target_dir_symlink_back.symlink_to(
        dirty_path_dir, target_is_directory=True
    )

    target_dir_symlink_file_back = target_dir / "dirty-file"
    target_dir_symlink_file_back.symlink_to(
        dirty_path_file, target_is_directory=False
    )

    # Create a symlink to the target directory
    dir_symlink = dirty_path / "to-target-dir"
    dir_symlink.symlink_to(target=target_dir, target_is_directory=True)

    # Create a target file
    target_file = tmp_path / "target-file"
    target_file.touch()

    # Create a symlink to the target file
    file_symlink = dirty_path / "to-target-file"
    file_symlink.symlink_to(target=target_file, target_is_directory=False)

    clean_path(dirty_path)

    assert {*dirty_path.iterdir()} == set()
    assert dirty_path.exists()

    assert target_dir.exists()
    assert not dir_symlink.exists()
    assert target_file.exists()
    assert not file_symlink.exists()
    assert not dirty_path_dir.exists()
    assert target_dir_symlink_back.exists(follow_symlinks=False)
    assert target_dir_symlink_file_back.exists(follow_symlinks=False)
