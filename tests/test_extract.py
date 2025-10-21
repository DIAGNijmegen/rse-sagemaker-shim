import struct
import zlib
from uuid import uuid4
from zipfile import ZipFile

import pytest

from sagemaker_shim.exceptions import UserSafeError
from sagemaker_shim.extract import safe_extract


def test_extract_with_dodgy_file(tmp_path):
    file = tmp_path / "test.zip"
    dest = tmp_path / "dest"

    with ZipFile(file=file, mode="w") as zip:
        zip.writestr("../foo.txt", "hello!")
        zip.writestr("../../foo.txt", "hello!")

    dest.mkdir()

    with pytest.raises(UserSafeError) as error:
        safe_extract(src=file, dest=dest)

    assert str(error) == (
        "<ExceptionInfo "
        "UserSafeError('Zip file contains invalid paths') tblen=2>"
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


def build_zip_with_unsupported_compression(
    filename: bytes, data: bytes
) -> bytes:
    """
    Build a tiny zip file bytes with compression method 99 for the file entry.
    """
    crc = zlib.crc32(data) & 0xFFFFFFFF
    comp_size = len(data)
    uncomp_size = len(data)
    method = 99  # unsupported compression method

    # Local file header
    # signature, version_needed, flag, compression, mtime, mdate,
    # crc, comp_size, uncomp_size, filename length, extra length
    lfh = struct.pack(
        "<I5H3I2H",
        0x04034B50,  # local file header signature
        20,  # version needed to extract
        0,  # general purpose bit flag
        method,  # compression method (unsupported)
        0,  # last mod file time
        0,  # last mod file date
        crc,
        comp_size,
        uncomp_size,
        len(filename),
        0,  # extra length
    )
    lfh_part = lfh + filename + data

    # Central directory header
    cd = struct.pack(
        "<I6H3I5H2I",
        0x02014B50,  # central file header signature
        0,  # version made by
        20,  # version needed to extract
        0,  # flag
        method,  # compression method
        0,  # mtime
        0,  # mdate
        crc,
        comp_size,
        uncomp_size,
        len(filename),
        0,  # extra len
        0,  # comment len
        0,  # disk start
        0,  # internal attrs
        0,  # external attrs
        0,  # relative offset of local header (we set LFH at offset 0)
    )
    cd_part = cd + filename

    central_dir_size = len(cd_part)
    central_dir_offset = len(
        lfh_part
    )  # central dir immediately after local header + data

    # End of central directory
    eocd = struct.pack(
        "<IHHHHIIH",
        0x06054B50,  # end of central dir signature
        0,  # number of this disk
        0,  # disk where central directory starts
        1,  # number of central directory records on this disk
        1,  # total number of central directory records
        central_dir_size,
        central_dir_offset,
        0,  # comment length
    )

    return lfh_part + cd_part + eocd


def test_safe_extract_raises_for_unsupported_compression(tmp_path):
    filename = b"file.txt"
    data = b"hello unsupported compression\n"
    zip_bytes = build_zip_with_unsupported_compression(filename, data)

    zip_path = tmp_path / "bad.zip"
    zip_path.write_bytes(zip_bytes)

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    with pytest.raises(UserSafeError) as excinfo:
        safe_extract(src=zip_path, dest=dest_dir)

    assert (
        str(excinfo.value)
        == "Zip file contains an unsupported compression method"
    )
