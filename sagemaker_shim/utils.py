import logging
import re
import zipfile
from os.path import commonpath
from pathlib import Path

from sagemaker_shim.exceptions import ZipExtractionError
from sagemaker_shim.vendor.werkzeug.security import safe_join

logger = logging.getLogger(__name__)


def _filter_members(members: list[zipfile.ZipInfo]) -> list[dict[str, str]]:
    """Filter common prefixes and uninteresting files from a zip archive"""
    members = [
        m
        for m in members
        if not m.is_dir()
        and re.search(r"(__MACOSX|\.DS_Store|desktop.ini)", m.filename) is None
    ]

    # Remove any common parent directories
    if len(members) == 1:
        path = str(Path(members[0].filename).parent)
        path = "" if path == "." else path
    else:
        path = commonpath([m.filename for m in members])

    if path:
        sliced_path = slice(len(path) + 1, None, None)
    else:
        sliced_path = slice(None, None, None)

    return [
        {"src": m.filename, "dest": m.filename[sliced_path]} for m in members
    ]


def safe_extract(*, src: Path, dest: Path) -> None:
    """
    Safely extracts a zip file into a directory

    Any common prefixes and system files are removed.
    """

    if not dest.exists():
        raise RuntimeError("The destination must exist")

    with src.open("rb") as f:
        with zipfile.ZipFile(f) as zf:
            members = _filter_members(zf.infolist())

            for member in members:
                file_dest = safe_join(str(dest), member["dest"])

                if file_dest is None:
                    raise ZipExtractionError("Zip file contains invalid paths")

                # We know that the dest is within the prefix as
                # safe_join is used, and the destination is already
                # created, so ok to create the parents here
                file_dest.parent.mkdir(exist_ok=True, parents=True)

                logger.info(
                    f"Extracting {member['src']=} from {src} to {file_dest}"
                )

                with zf.open(member["src"], "r") as fs, open(
                    file_dest, "wb"
                ) as fd:
                    while chunk := fs.read(8192):
                        fd.write(chunk)
