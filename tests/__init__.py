import hashlib
from pathlib import Path


def get_version():
    h = hashlib.sha256()

    root_dir = Path(__file__).parent.parent

    files = [
        *(root_dir / "sagemaker_shim").rglob("**/*.py"),
        root_dir / "uv.lock",
    ]

    for file in files:
        with open(file, "rb") as f:
            h.update(f.read())

    return h.hexdigest()[:8]


__version__ = get_version()
