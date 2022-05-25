"""From Werkzeug 2.1"""
import os
import posixpath
from pathlib import Path

_os_alt_seps: list[str] = list(
    sep
    for sep in [os.path.sep, os.path.altsep]
    if sep is not None and sep != "/"
)


def safe_join(directory: str, *pathnames: str) -> Path | None:
    """Safely join zero or more untrusted path components to a base
    directory to avoid escaping the base directory.
    :param directory: The trusted base directory.
    :param pathnames: The untrusted path components relative to the
        base directory.
    :return: A safe path, otherwise ``None``.
    """
    if not directory:
        # Ensure we end up with ./path if directory="" is given,
        # otherwise the first untrusted part could become trusted.
        directory = "."

    parts = [directory]

    for filename in pathnames:
        if filename != "":
            filename = posixpath.normpath(filename)

        if (
            any(sep in filename for sep in _os_alt_seps)
            or os.path.isabs(filename)
            or filename == ".."
            or filename.startswith("../")
        ):
            return None

        parts.append(filename)

    return Path(posixpath.join(*parts))
