import asyncio
import json
import logging
import os
import re
import subprocess
from base64 import b64decode
from functools import cached_property
from importlib.metadata import version
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
from zipfile import BadZipFile

import boto3
from pydantic import BaseModel, validator

from sagemaker_shim.exceptions import ZipExtractionError
from sagemaker_shim.logging import STDOUT_LEVEL
from sagemaker_shim.utils import safe_extract

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client  # pragma: no cover
else:
    S3Client = object


logger = logging.getLogger(__name__)
lock = asyncio.Lock()

BUCKET_NAME_REGEX = re.compile(r"^[a-zA-Z0-9.\-_]{1,255}$")
BUCKET_ARN_REGEX = re.compile(
    r"^arn:(aws).*:(s3|s3-object-lambda):[a-z\-0-9]*:[0-9]{12}:accesspoint[/:][a-zA-Z0-9\-.]{1,63}$|^arn:(aws).*:s3-outposts:[a-z\-0-9]+:[0-9]{12}:outpost[/:][a-zA-Z0-9\-]{1,63}[/:]accesspoint[/:][a-zA-Z0-9\-]{1,63}$"
)


def validate_bucket_name(v: str) -> str:
    if BUCKET_NAME_REGEX.match(v) or BUCKET_ARN_REGEX.match(v):
        return v
    else:
        raise ValueError("Invalid bucket name")


class InferenceIO(BaseModel):
    """A single input or output file for an inference job"""

    relative_path: Path
    bucket_name: str
    bucket_key: str
    decompress: bool = False

    class Config:
        frozen = True

    @validator("bucket_name")
    def validate_bucket_name(cls, v: str) -> str:  # noqa:B902
        return validate_bucket_name(v)

    def local_file(self, path: Path) -> Path:
        """The local location of the file"""
        return path / self.relative_path

    def download(self, *, input_path: Path, s3_client: S3Client) -> None:
        """Download this file from s3"""
        dest_file = self.local_file(path=input_path)

        logger.info(
            f"Downloading {self.bucket_key=} from {self.bucket_name=} to {dest_file=}"
        )

        dest_file.parent.mkdir(exist_ok=True, parents=True)

        if self.decompress:
            with TemporaryDirectory() as tmp_dir:
                zipfile = Path(tmp_dir) / "src.zip"
                with zipfile.open("wb") as f:
                    s3_client.download_fileobj(
                        Bucket=self.bucket_name,
                        Key=self.bucket_key,
                        Fileobj=f,
                    )
                try:
                    safe_extract(src=zipfile, dest=dest_file.parent)
                except BadZipFile as error:
                    raise ZipExtractionError(
                        "Input zip file could not be extracted"
                    ) from error
        else:
            with dest_file.open("wb") as f:
                s3_client.download_fileobj(
                    Bucket=self.bucket_name,
                    Key=self.bucket_key,
                    Fileobj=f,
                )

    def upload(self, *, output_path: Path, s3_client: S3Client) -> None:
        """Upload this file to s3"""
        src_file = str(self.local_file(path=output_path))

        logger.info(
            f"Uploading {src_file=} to {self.bucket_name=} with {self.bucket_key=}"
        )

        s3_client.upload_file(
            Filename=src_file,
            Bucket=self.bucket_name,
            Key=self.bucket_key,
        )


class InferenceResult(BaseModel):
    return_code: int
    outputs: list[InferenceIO]
    sagemaker_shim_version: str = version("sagemaker-shim")

    class Config:
        frozen = True


class InferenceTask(BaseModel):
    pk: str
    inputs: list[InferenceIO]
    output_bucket_name: str
    output_prefix: str

    class Config:
        frozen = True

    @validator("output_prefix")
    def validate_prefix(cls, v: str) -> str:  # noqa:B902
        if not v:
            raise ValueError("Prefix cannot be blank")

        if v[-1] != "/":
            v += "/"

        return v

    @validator("output_bucket_name")
    def validate_bucket_name(cls, v: str) -> str:  # noqa:B902
        return validate_bucket_name(v)

    @staticmethod
    def decode_b64j(*, encoded: str | None) -> Any:
        """JSON decode a base64 string"""
        if encoded is None:
            return None
        else:
            return json.loads(
                b64decode(encoded.encode("utf-8")).decode("utf-8")
            )

    @classmethod
    @property
    def cmd(cls) -> Any:
        """The original command for the subprocess"""
        cmd = cls.decode_b64j(
            encoded=os.environ.get("GRAND_CHALLENGE_COMPONENT_CMD_B64J")
        )
        logger.debug(f"{cmd=}")
        return cmd

    @classmethod
    @property
    def entrypoint(cls) -> Any:
        """The original entrypoint for the subprocess"""
        entrypoint = cls.decode_b64j(
            encoded=os.environ.get("GRAND_CHALLENGE_COMPONENT_ENTRYPOINT_B64J")
        )
        logger.debug(f"{entrypoint=}")
        return entrypoint

    @classmethod
    @property
    def input_path(cls) -> Path:
        """Local path where the subprocess is expected to read its input files"""
        input_path = Path(
            os.environ.get("GRAND_CHALLENGE_COMPONENT_INPUT_PATH", "/input")
        )
        logger.debug(f"{input_path=}")
        return input_path

    @classmethod
    @property
    def output_path(cls) -> Path:
        """Local path where the subprocess is expected to write its files"""
        output_path = Path(
            os.environ.get("GRAND_CHALLENGE_COMPONENT_OUTPUT_PATH", "/output")
        )
        logger.debug(f"{output_path=}")
        return output_path

    @cached_property
    def _s3_client(self) -> S3Client:
        return boto3.client(
            "s3", endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL")
        )

    @property
    def proc_args(self) -> list[str]:
        """
        Implementation of CMD and ENTRYPOINT parsing from
        https://docs.docker.com/engine/reference/builder/#understand-how-cmd-and-entrypoint-interact
                                    | No ENTRYPOINT              | ENTRYPOINT exec_entry p1_entry | ENTRYPOINT ["exec_entry", "p1_entry"]
        No CMD                      | error, not allowed         | /bin/sh -c exec_entry p1_entry | exec_entry p1_entry
        CMD ["exec_cmd", "p1_cmd"]  | exec_cmd p1_cmd            | /bin/sh -c exec_entry p1_entry | exec_entry p1_entry exec_cmd p1_cmd
        CMD ["p1_cmd", "p2_cmd"]    | p1_cmd p2_cmd              | /bin/sh -c exec_entry p1_entry | exec_entry p1_entry p1_cmd p2_cmd
        CMD exec_cmd p1_cmd         | /bin/sh -c exec_cmd p1_cmd | /bin/sh -c exec_entry p1_entry | exec_entry p1_entry /bin/sh -c exec_cmd p1_cmd
        """  # noqa: B950
        entrypoint = self.entrypoint
        cmd = self.cmd

        if entrypoint is None and cmd is None:
            raise ValueError("Either cmd or entrypoint must be set")
        elif isinstance(entrypoint, str):
            proc_args = ["/bin/sh", "-c", entrypoint]
        else:
            proc_args = entrypoint if entrypoint is not None else []

            if cmd is None:
                pass
            elif isinstance(cmd, str):
                proc_args += ["/bin/sh", "-c", cmd]
            else:
                proc_args += cmd

        return proc_args

    @property
    def proc_env(self) -> dict[str, str]:
        """The environment for the subprocess"""
        env = os.environ.copy()

        # Set LD_LIBRARY_PATH correctly, see
        # https://pyinstaller.org/en/stable/runtime-information.html#ld-library-path-libpath-considerations
        lp_key = "LD_LIBRARY_PATH"
        lp_orig = env.get(lp_key + "_ORIG")

        if lp_orig is not None:
            env[lp_key] = lp_orig
        else:
            env.pop(lp_key, None)

        return env

    async def invoke(self) -> InferenceResult:
        """Run the inference on a single case"""

        await asyncio.wait_for(lock.acquire(), timeout=1.0)

        ignore_clean_errors = False

        try:
            try:
                self.clean_io(ignore_errors=ignore_clean_errors)
            except PermissionError as error:
                self.log_external(
                    level=logging.ERROR,
                    msg=(
                        f"Could not clean '{self.input_path}' and/or "
                        f"'{self.output_path}' directories: {error}"
                    ),
                )
                ignore_clean_errors = True
                return InferenceResult(return_code=1, outputs=[])

            try:
                self.download_input()
            except ZipExtractionError as error:
                self.log_external(level=logging.ERROR, msg=str(error))
                return InferenceResult(return_code=1, outputs=[])

            return_code = await self.execute()

            if return_code == 0:
                outputs = self.upload_output()
            else:
                outputs = set()

            return InferenceResult(return_code=return_code, outputs=outputs)
        finally:
            try:
                self.clean_io(ignore_errors=ignore_clean_errors)
            finally:
                lock.release()

    def clean_io(self, *, ignore_errors: bool = False) -> None:
        """Clean all contents of input and output folders"""
        self._clean_path(path=self.input_path, ignore_errors=ignore_errors)
        self._clean_path(path=self.output_path, ignore_errors=ignore_errors)

    @staticmethod
    def _clean_path(*, path: Path, ignore_errors: bool) -> None:
        """Removes contents of a directory, keeping the parent"""
        for f in path.glob("**/*"):
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                rmtree(path=f, ignore_errors=ignore_errors)

    def download_input(self) -> None:
        """Download all the inputs to the input path"""
        for input_file in self.inputs:
            input_file.download(
                input_path=self.input_path, s3_client=self._s3_client
            )

    def upload_output(self) -> set[InferenceIO]:
        """Upload all the outputs from the output path to s3"""
        output_path = self.output_path
        outputs: set[InferenceIO] = set()

        for f in output_path.rglob("**/*"):
            if not f.is_file() and not f.is_symlink():
                logger.warning(f"Skipping {f=}")
            else:
                relative_path = f.relative_to(output_path)
                output = InferenceIO(
                    relative_path=relative_path,
                    bucket_key=f"{self.output_prefix}{relative_path}",
                    bucket_name=self.output_bucket_name,
                )
                output.upload(
                    output_path=self.output_path, s3_client=self._s3_client
                )
                outputs.add(output)

        return outputs

    async def execute(self) -> int:
        """Run the original entrypoint and command in a subprocess"""
        logger.debug(f"Calling {self.proc_args=}")

        process = await asyncio.create_subprocess_exec(
            *self.proc_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.proc_env,
        )

        try:
            await asyncio.gather(
                self._stream_to_external(
                    stream=process.stdout, level=STDOUT_LEVEL
                ),
                self._stream_to_external(
                    stream=process.stderr, level=STDOUT_LEVEL + 10
                ),
            )
        except Exception:
            process.kill()
            raise
        finally:
            return_code = await process.wait()

        logger.debug(f"{return_code=}")

        return return_code

    async def _stream_to_external(
        self, *, stream: asyncio.StreamReader | None, level: int
    ) -> None:
        """Send the contents of an io stream to the external logs"""
        if stream is None:
            return

        while True:
            try:
                line = await stream.readline()
            except ValueError:
                self.log_external(
                    level=logging.WARNING,
                    msg="WARNING: A log line was skipped as it was too long",
                )
                continue

            if not line:
                break

            self.log_external(
                level=level,
                msg=line.replace(b"\x00", b"").decode("utf-8"),
            )

    def log_external(self, *, level: int, msg: str) -> None:
        """Send a message to the external logger"""
        logger.log(
            level=level, msg=msg, extra={"internal": False, "task": self}
        )
