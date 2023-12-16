import asyncio
import errno
import grp
import io
import json
import logging
import os
import pwd
import re
import subprocess
from base64 import b64decode
from functools import cached_property
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, NamedTuple
from zipfile import BadZipFile

import boto3
from pydantic import BaseModel, ConfigDict, RootModel, field_validator

from sagemaker_shim.exceptions import ZipExtractionError
from sagemaker_shim.extract import safe_extract
from sagemaker_shim.logging import STDOUT_LEVEL

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


def get_s3_client() -> S3Client:
    return boto3.client(
        "s3", endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL")
    )


def get_s3_file_content(*, s3_uri: str) -> bytes:
    pattern = r"^(https|s3)://(?P<bucket>[^/]+)/?(?P<key>.*)$"
    match = re.fullmatch(pattern, s3_uri)

    if not match:
        raise ValueError(f"Not a valid S3 uri, must match pattern {pattern}")

    s3_client = get_s3_client()

    content = io.BytesIO()
    s3_client.download_fileobj(
        Fileobj=content,
        Bucket=match.group("bucket"),
        Key=match.group("key"),
    )
    content.seek(0)

    return content.read()


def validate_bucket_name(v: str) -> str:
    if BUCKET_NAME_REGEX.match(v) or BUCKET_ARN_REGEX.match(v):
        return v
    else:
        raise ValueError("Invalid bucket name")


class InferenceIO(BaseModel):
    """A single input or output file for an inference job"""

    model_config = ConfigDict(frozen=True)

    relative_path: Path
    bucket_name: str
    bucket_key: str
    decompress: bool = False

    @field_validator("bucket_name")
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
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
                except OSError as error:
                    if error.errno == errno.ENOSPC:
                        raise ZipExtractionError(
                            "Contents of zip file too large"
                        ) from error
                    else:
                        raise error
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
    model_config = ConfigDict(frozen=True)

    pk: str
    return_code: int
    outputs: list[InferenceIO]
    sagemaker_shim_version: str = version("sagemaker-shim")


class UserGroup(NamedTuple):
    uid: int | None
    gid: int | None
    home: str | None


class InferenceTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    pk: str
    inputs: list[InferenceIO]
    output_bucket_name: str
    output_prefix: str

    @field_validator("output_prefix")
    @classmethod
    def validate_prefix(cls, v: str) -> str:
        if not v:
            raise ValueError("Prefix cannot be blank")

        if v[-1] != "/":
            v += "/"

        return v

    @field_validator("output_bucket_name")
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
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

    @property
    def cmd(self) -> Any:
        """The original command for the subprocess"""
        cmd = self.decode_b64j(
            encoded=os.environ.get("GRAND_CHALLENGE_COMPONENT_CMD_B64J")
        )
        logger.debug(f"{cmd=}")
        return cmd

    @property
    def entrypoint(self) -> Any:
        """The original entrypoint for the subprocess"""
        entrypoint = self.decode_b64j(
            encoded=os.environ.get("GRAND_CHALLENGE_COMPONENT_ENTRYPOINT_B64J")
        )
        logger.debug(f"{entrypoint=}")
        return entrypoint

    @property
    def user(self) -> str:
        return os.environ.get("GRAND_CHALLENGE_COMPONENT_USER", "")

    @property
    def input_path(self) -> Path:
        """Local path where the subprocess is expected to read its input files"""
        input_path = Path(
            os.environ.get("GRAND_CHALLENGE_COMPONENT_INPUT_PATH", "/input")
        )
        logger.debug(f"{input_path=}")
        return input_path

    @property
    def output_path(self) -> Path:
        """Local path where the subprocess is expected to write its files"""
        output_path = Path(
            os.environ.get("GRAND_CHALLENGE_COMPONENT_OUTPUT_PATH", "/output")
        )
        logger.debug(f"{output_path=}")
        return output_path

    @cached_property
    def _s3_client(self) -> S3Client:
        return get_s3_client()

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

        if self.proc_user.home is not None:
            env["HOME"] = self.proc_user.home

        return env

    @staticmethod
    def _get_user_info(id_or_name: str) -> pwd.struct_passwd | None:
        if id_or_name == "":
            return None

        try:
            return pwd.getpwnam(id_or_name)
        except (KeyError, AttributeError):
            try:
                return pwd.getpwuid(int(id_or_name))
            except (KeyError, ValueError, AttributeError) as error:
                raise RuntimeError(f"User {id_or_name} not found") from error

    @staticmethod
    def _get_group_info(id_or_name: str) -> grp.struct_group | None:
        if id_or_name == "":
            return None

        try:
            return grp.getgrnam(id_or_name)
        except (KeyError, AttributeError):
            try:
                return grp.getgrgid(int(id_or_name))
            except (KeyError, ValueError, AttributeError) as error:
                raise RuntimeError(f"Group {id_or_name} not found") from error

    @cached_property
    def proc_user(self) -> UserGroup:
        match = re.fullmatch(
            r"^(?P<user>[0-9a-zA-Z]*):?(?P<group>[0-9a-zA-Z]*)$", self.user
        )

        if match:
            user = self._get_user_info(id_or_name=match.group("user"))
            group = self._get_group_info(id_or_name=match.group("group"))

            if user is None:
                uid = None
                home = None
            else:
                uid = user.pw_uid
                home = user.pw_dir

            if group is None:
                if user is None:
                    gid = None
                else:
                    # Switch to the users primary group
                    gid = user.pw_gid
            else:
                gid = group.gr_gid

            return UserGroup(uid=uid, gid=gid, home=home)
        else:
            return UserGroup(uid=None, gid=None, home=None)

    async def invoke(self) -> InferenceResult:
        """Run the inference on a single case"""

        await asyncio.wait_for(lock.acquire(), timeout=1.0)

        try:
            inference_result = await self._invoke()

            logger.info(
                f"Inference for {self.pk=} complete, {inference_result=}"
            )

            self.upload_inference_result(inference_result=inference_result)

            return inference_result
        finally:
            lock.release()

    async def _invoke(self) -> InferenceResult:
        logger.info(f"Invoking {self.pk=}")

        try:
            self.clean_io()

            try:
                self.download_input()
            except ZipExtractionError as error:
                self.log_external(level=logging.ERROR, msg=str(error))
                return InferenceResult(pk=self.pk, return_code=1, outputs=[])

            return_code = await self.execute()

            if return_code == 0:
                outputs = self.upload_output()
            else:
                outputs = set()

            return InferenceResult(
                pk=self.pk, return_code=return_code, outputs=outputs
            )
        finally:
            self.clean_io()

    def clean_io(self) -> None:
        """Clean all contents of input and output folders"""
        self._clean_path(path=self.input_path)
        self._clean_path(path=self.output_path)

    def _clean_path(self, *, path: Path) -> None:
        """Removes contents of a directory, keeping the parent"""
        for f in path.glob("*"):
            if f.is_file():
                f.chmod(0o700)
                f.unlink()
            elif f.is_dir():
                f.chmod(0o700)
                self._clean_path(path=f)
                f.rmdir()

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

    def upload_inference_result(
        self, *, inference_result: InferenceResult
    ) -> None:
        fileobj = io.BytesIO(
            inference_result.model_dump_json().encode("utf-8")
        )
        bucket_key = (
            f"{self.output_prefix}.sagemaker_shim/inference_result.json"
        )

        logger.info(
            f"Uploading Inference Result to "
            f"{self.output_bucket_name=} with {bucket_key=}"
        )

        self._s3_client.upload_fileobj(
            Fileobj=fileobj,
            Bucket=self.output_bucket_name,
            Key=bucket_key,
        )

    async def execute(self) -> int:
        """Run the original entrypoint and command in a subprocess"""
        logger.info(f"Calling {self.proc_args=}")

        process = await asyncio.create_subprocess_exec(
            *self.proc_args,
            user=self.proc_user.uid,
            group=self.proc_user.gid,
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

        logger.info(f"{return_code=}")

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


class InferenceTaskList(RootModel[list[InferenceTask]]):
    pass
