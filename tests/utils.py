import itertools
import json
import subprocess
import tarfile
from base64 import b64encode
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

CRANE_COMMAND = "dist/crane"


def encode_b64j(*, val: Any) -> str:
    return b64encode(json.dumps(val).encode("utf-8")).decode("utf-8")


def push_image(*, client, repo_tag):
    with TemporaryDirectory() as tempdir:
        image = client.images.get(name=repo_tag)
        image_filename = Path(tempdir) / "image.tar"

        f = open(image_filename, "wb")

        for chunk in image.save():
            f.write(chunk)

        f.close()

        subprocess.run(
            [CRANE_COMMAND, "push", image_filename, repo_tag],
            capture_output=True,
        )


def pull_image(*, client, repo_tag):
    with TemporaryDirectory() as tempdir:
        image_filename = Path(tempdir) / "image.tar"

        subprocess.run(
            [CRANE_COMMAND, "pull", repo_tag, image_filename],
            capture_output=True,
        )

        with open(image_filename, "rb") as f:
            client.images.load(data=f)


def get_image_config(*, repo_tag: str) -> Any:
    output = subprocess.run(
        [CRANE_COMMAND, "config", repo_tag],
        capture_output=True,
    )
    return json.loads(output.stdout.decode("utf-8"))


def get_new_env_vars(*, existing_config: dict[str, Any]) -> dict[str, str]:
    cmd = existing_config["config"].get("Cmd")
    entrypoint = existing_config["config"].get("Entrypoint")

    return {
        "GRAND_CHALLENGE_COMPONENT_CMD_B64J": encode_b64j(val=cmd),
        "GRAND_CHALLENGE_COMPONENT_ENTRYPOINT_B64J": encode_b64j(
            val=entrypoint
        ),
        "GRAND_CHALLENGE_COMPONENT_USER": "0",
    }


def mutate_image(
    *, repo_tag: str, env_vars: dict[str, str], version: str
) -> str:
    new_tag = f"{repo_tag}-sagemaker-shim-{version}"

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        new_layer = tmp_path / "sagemaker-shim.tar"

        with tarfile.open(new_layer, "w") as f:

            def _set_root_555_perms(
                tarinfo: tarfile.TarInfo,
            ) -> tarfile.TarInfo:
                tarinfo.uid = 0
                tarinfo.gid = 0
                tarinfo.mode = 0o555
                return tarinfo

            f.add(
                name=f"dist/sagemaker-shim-static-{version}",
                arcname="/sagemaker-shim",
                filter=_set_root_555_perms,
            )

            for dir in ["/input", "/output", "/tmp"]:
                # /tmp is required by staticx
                tarinfo = tarfile.TarInfo(dir)
                tarinfo.type = tarfile.DIRTYPE
                tarinfo.uid = 0
                tarinfo.gid = 0
                tarinfo.mode = 0o777
                f.addfile(tarinfo=tarinfo)

        subprocess.run(
            [
                CRANE_COMMAND,
                "mutate",
                repo_tag,
                "--cmd",
                "",
                "--entrypoint",
                "/sagemaker-shim",
                "--tag",
                new_tag,
                "--append",
                str(new_layer),
                *itertools.chain(
                    *[["--env", f"{k}={v}"] for k, v in env_vars.items()]
                ),
            ],
            capture_output=True,
        )

    return new_tag
