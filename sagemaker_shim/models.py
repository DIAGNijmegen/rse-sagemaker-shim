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
import tarfile
from base64 import b64decode
from functools import cached_property
from importlib.metadata import version
from pathlib import Path
from tempfile import SpooledTemporaryFile, TemporaryDirectory
from types import TracebackType
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


class S3File(NamedTuple):
    bucket: str
    key: str


def parse_s3_uri(*, s3_uri: str) -> S3File:
    pattern = r"^(https|s3)://(?P<bucket>[^/]+)/?(?P<key>.*)$"
    match = re.fullmatch(pattern, s3_uri)

    if not match:
        raise ValueError(f"Not a valid S3 uri, must match pattern {pattern}")

    return S3File(bucket=match.group("bucket"), key=match.group("key"))


def get_s3_file_content(*, s3_uri: str) -> bytes:
    s3_file = parse_s3_uri(s3_uri=s3_uri)

    s3_client = get_s3_client()

    content = io.BytesIO()
    s3_client.download_fileobj(
        Fileobj=content,
        Bucket=s3_file.bucket,
        Key=s3_file.key,
    )
    content.seek(0)

    return content.read()


def download_and_extract_tarball(*, s3_uri: str, dest: Path) -> None:
    s3_file = parse_s3_uri(s3_uri=s3_uri)
    s3_client = get_s3_client()

    with SpooledTemporaryFile(max_size=4 * 1024 * 1024 * 1024) as f:
        s3_client.download_fileobj(
            Bucket=s3_file.bucket,
            Key=s3_file.key,
            Fileobj=f,
        )

        f.seek(0)

        with tarfile.open(fileobj=f, mode="r") as tar:
            tar.extractall(path=dest, filter="data")


def validate_bucket_name(v: str) -> str:
    if BUCKET_NAME_REGEX.match(v) or BUCKET_ARN_REGEX.match(v):
        return v
    else:
        raise ValueError("Invalid bucket name")


def clean_path(path: Path) -> None:
    for f in path.glob("*"):
        if f.is_file():
            f.chmod(0o700)
            f.unlink()
        elif f.is_dir():
            f.chmod(0o700)
            clean_path(f)
            f.rmdir()


class DependentData:
    @property
    def model_source(self) -> str | None:
        """s3 URI to a .tar.gz file that is extracted to model_dest"""
        model = os.environ.get("GRAND_CHALLENGE_COMPONENT_MODEL")
        logger.debug(f"{model=}")
        return model

    @property
    def model_dest(self) -> Path:
        model_dest = Path(
            os.environ.get(
                "GRAND_CHALLENGE_COMPONENT_MODEL_DEST", "/opt/ml/model/"
            )
        )
        logger.debug(f"{model_dest=}")
        return model_dest

    @property
    def ground_truth_source(self) -> str | None:
        """s3 URI to a .tar.gz file that is extracted to ground_truth_dest"""
        ground_truth = os.environ.get("GRAND_CHALLENGE_COMPONENT_GROUND_TRUTH")
        logger.debug(f"{ground_truth=}")
        return ground_truth

    @property
    def ground_truth_dest(self) -> Path:
        ground_truth_dest = Path(
            os.environ.get(
                "GRAND_CHALLENGE_COMPONENT_GROUND_TRUTH_DEST",
                "/opt/ml/input/data/ground_truth/",
            )
        )
        logger.debug(f"{ground_truth_dest=}")
        return ground_truth_dest

    @property
    def writable_directories(self) -> list[Path]:
        writable_directories = [
            Path(d)
            for d in os.environ.get(
                "GRAND_CHALLENGE_COMPONENT_WRITABLE_DIRECTORIES", ""
            ).split(":")
            if d
        ]
        logger.debug(f"{writable_directories=}")
        return writable_directories

    @property
    def post_clean_directories(self) -> list[Path]:
        post_clean_directories = [
            Path(d)
            for d in os.environ.get(
                "GRAND_CHALLENGE_COMPONENT_POST_CLEAN_DIRECTORIES", ""
            ).split(":")
            if d
        ]
        logger.debug(f"{post_clean_directories=}")
        return post_clean_directories

    def __enter__(self) -> "DependentData":
        logger.info("Setting up Dependent Data")
        self.ensure_directories_are_writable()
        self.download_model()
        self.download_ground_truth()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        logger.info("Cleaning up Dependent Data")
        for p in self.post_clean_directories:
            logger.info(f"Cleaning {p=}")
            clean_path(p)

    def ensure_directories_are_writable(self) -> None:
        for directory in self.writable_directories:
            path = Path(directory)
            path.mkdir(exist_ok=True, parents=True)
            path.chmod(mode=0o777)

    def download_model(self) -> None:
        if self.model_source is not None:
            logger.info(
                f"Downloading model from {self.model_source=} to {self.model_dest=}"
            )
            self.model_dest.mkdir(parents=True, exist_ok=True)
            download_and_extract_tarball(
                s3_uri=self.model_source, dest=self.model_dest
            )

    def download_ground_truth(self) -> None:
        if self.ground_truth_source is not None:
            logger.info(
                f"Downloading ground truth from {self.ground_truth_source=} "
                f"to {self.ground_truth_dest=}"
            )
            self.ground_truth_dest.mkdir(parents=True, exist_ok=True)
            download_and_extract_tarball(
                s3_uri=self.ground_truth_source, dest=self.ground_truth_dest
            )


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


class UserInfo(NamedTuple):
    uid: int | None
    gid: int | None
    home: str | None
    groups: list[int]


def _get_user_info(id_or_name: str) -> UserInfo:
    if id_or_name == "":
        return UserInfo(uid=None, gid=None, home=None, groups=[])

    try:
        user = pwd.getpwnam(id_or_name)
    except (KeyError, AttributeError):
        try:
            uid = int(id_or_name)
        except ValueError as error:
            raise RuntimeError(f"User '{id_or_name}' not found") from error

        try:
            user = pwd.getpwuid(uid)
        except (KeyError, AttributeError):
            return UserInfo(uid=uid, gid=None, home=None, groups=[])

    return UserInfo(
        uid=user.pw_uid,
        gid=user.pw_gid,
        home=user.pw_dir,
        groups=_get_users_groups(user=user),
    )


def _get_users_groups(*, user: pwd.struct_passwd) -> list[int]:
    users_groups = [
        g.gr_gid for g in grp.getgrall() if user.pw_name in g.gr_mem
    ]
    return _put_gid_first(gid=user.pw_gid, groups=users_groups)


def _put_gid_first(*, gid: int | None, groups: list[int]) -> list[int]:
    if gid is None:
        return groups
    else:
        user_groups = set(groups)

        try:
            user_groups.remove(gid)
        except KeyError:
            pass

        return [gid, *sorted(user_groups)]


def _get_group_id(id_or_name: str) -> int | None:
    if id_or_name == "":
        return None

    try:
        return grp.getgrnam(id_or_name).gr_gid
    except (KeyError, AttributeError):
        try:
            return int(id_or_name)
        except ValueError as error:
            raise RuntimeError(f"Group '{id_or_name}' not found") from error


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

    @property
    def extra_groups(self) -> list[int] | None:
        if (
            os.environ.get(
                "GRAND_CHALLENGE_COMPONENT_SET_EXTRA_GROUPS", "True"
            ).lower()
            == "true"
        ):
            return self.proc_user.groups
        else:
            return None

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

    @cached_property
    def proc_user(self) -> UserInfo:
        if self.user == "":
            return UserInfo(uid=None, gid=None, home=None, groups=[])

        match = re.fullmatch(
            r"^(?P<user>[0-9a-zA-Z]*):?(?P<group>[0-9a-zA-Z]*)$", self.user
        )

        if match:
            info = _get_user_info(id_or_name=match.group("user"))
            group_id = _get_group_id(id_or_name=match.group("group"))

            gid = info.gid if group_id is None else group_id

            return UserInfo(
                uid=info.uid,
                gid=gid,
                home=info.home,
                groups=_put_gid_first(gid=gid, groups=info.groups),
            )
        else:
            raise RuntimeError(f"Invalid user '{self.user}'")

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
        clean_path(path=self.input_path)
        clean_path(path=self.output_path)

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
            extra_groups=self.extra_groups,
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
