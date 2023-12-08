from uuid import uuid4
from zipfile import ZipFile

import pytest

from sagemaker_shim.exceptions import ZipExtractionError
from sagemaker_shim.extract import safe_extract


def test_extract_with_dodgy_file(tmp_path):
    file = tmp_path / "test.zip"
    dest = tmp_path / "dest"

    with ZipFile(file=file, mode="w") as zip:
        zip.writestr("../foo.txt", "hello!")
        zip.writestr("../../foo.txt", "hello!")

    dest.mkdir()

    with pytest.raises(ZipExtractionError) as error:
        safe_extract(src=file, dest=dest)

    assert str(error) == (
        "<ExceptionInfo "
        "ZipExtractionError('Zip file contains invalid paths') tblen=2>"
    )


def test_directory_must_exist(tmp_path):
    with pytest.raises(RuntimeError) as error:
        safe_extract(src=tmp_path, dest=tmp_path / "dfg")

    assert (
        str(error)
        == "<ExceptionInfo RuntimeError('The destination must exist') tblen=2>"
    )


def test_extraction(tmp_path):
    file = tmp_path / "test.zip"
    dest = tmp_path / "dest"
    dest.mkdir()

    with ZipFile(file=file, mode="w") as zip:
        for arcname in [
            "common/base.txt",
            "common/sub/sub.txt",
            "__MACOSX",
            ".DS_Store",
            "desktop.ini",
        ]:
            zip.writestr(arcname, "hello")

        zip.writestr("common/bigfile.bin", "a" * int(8192 * 1.5))

    safe_extract(src=file, dest=dest)

    assert {*dest.rglob("**/*")} == {
        dest / "base.txt",
        dest / "bigfile.bin",
        dest / "sub",
        dest / "sub/sub.txt",
    }

    with open(dest / "bigfile.bin") as f:
        contents = f.readlines()

    assert contents == ["a" * int(8192 * 1.5)]


def test_single_file(tmp_path):
    file = tmp_path / "test.zip"
    dest = tmp_path / "dest"
    dest.mkdir()
    pk = uuid4()

    with ZipFile(file=file, mode="w") as zip:
        zip.writestr("test.txt", str(pk))

    safe_extract(src=file, dest=dest)

    with open(dest / "test.txt") as f:
        content = f.readlines()

    assert content == [str(pk)]


def test_single_file_nested(tmp_path):
    file = tmp_path / "test.zip"
    dest = tmp_path / "dest"
    dest.mkdir()
    pk = uuid4()

    with ZipFile(file=file, mode="w") as zip:
        zip.writestr("foo/test.txt", str(pk))

    safe_extract(src=file, dest=dest)

    with open(dest / "test.txt") as f:
        content = f.readlines()

    assert content == [str(pk)]


def test_single_directory(tmp_path):
    file = tmp_path / "test.zip"
    dest = tmp_path / "dest"
    dest.mkdir()

    with ZipFile(file=file, mode="w") as zip:
        zip.mkdir("just-a-directory")

    safe_extract(src=file, dest=dest)

    assert {*dest.rglob("**/*")} == set()
