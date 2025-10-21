import asyncio
import errno
import grp
import hashlib
import hmac
import io
import json
import logging
import os
import pwd
import re
import signal
import subprocess
import tarfile
import time
from asyncio import Semaphore
from base64 import b64decode
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from datetime import timedelta
from functools import cached_property
from importlib.metadata import version
from pathlib import Path
from tempfile import SpooledTemporaryFile, TemporaryDirectory
from types import TracebackType
from typing import TYPE_CHECKING, Any, NamedTuple
from zipfile import BadZipFile

import aioboto3
from botocore.config import Config
from pydantic import BaseModel, ConfigDict, RootModel, field_validator

from sagemaker_shim.exceptions import UserSafeError
from sagemaker_shim.extract import safe_extract
from sagemaker_shim.logging import STDOUT_LEVEL

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath  # pragma: no cover
    from types_aiobotocore_s3.client import S3Client  # pragma: no cover
else:
    StrOrBytesPath = object
    S3Client = object


logger = logging.getLogger(__name__)
lock = asyncio.Lock()

BUCKET_NAME_REGEX = re.compile(r"^[a-zA-Z0-9.\-_]{1,255}$")
BUCKET_ARN_REGEX = re.compile(
    r"^arn:(aws).*:(s3|s3-object-lambda):[a-z\-0-9]*:[0-9]{12}:accesspoint[/:][a-zA-Z0-9\-.]{1,63}$|^arn:(aws).*:s3-outposts:[a-z\-0-9]+:[0-9]{12}:outpost[/:][a-zA-Z0-9\-]{1,63}[/:]accesspoint[/:][a-zA-Z0-9\-]{1,63}$"
)


class UserInfo(NamedTuple):
    uid: int | None
    gid: int | None
    home: str | None
    groups: list[int]


class ProcUserMixin:
    @property
    def _user(self) -> str:
        user = os.environ.get("GRAND_CHALLENGE_COMPONENT_USER", "")
        logger.info(f"{user=}")
        return user

    @cached_property
    def proc_user(self) -> UserInfo:
        if self._user == "":
            return UserInfo(uid=None, gid=None, home=None, groups=[])

        match = re.fullmatch(
            r"^(?P<user>[A-Za-z][A-Za-z0-9._-]*|[0-9]+)?(:(?P<group>[A-Za-z][A-Za-z0-9._-]*|[0-9]+))?$",
            self._user,
        )

        if match:
            user = match.group("user")
            group = match.group("group")

            logger.info(f"Looking up {user=} {group=}")

            info = self._get_user_info(id_or_name=user)
            group_id = self._get_group_id(id_or_name=group)

            gid = info.gid if group_id is None else group_id

            return UserInfo(
                uid=info.uid,
                gid=gid,
                home=info.home,
                groups=self._put_gid_first(gid=gid, groups=info.groups),
            )
        else:
            raise UserSafeError(
                "Invalid argument for the containers USER instruction"
            )

    @classmethod
    def _get_user_info(cls, id_or_name: str | None) -> UserInfo:
        if id_or_name is None:
            return UserInfo(uid=None, gid=None, home=None, groups=[])

        try:
            user = pwd.getpwnam(id_or_name)
        except (KeyError, AttributeError):
            try:
                uid = int(id_or_name)
            except ValueError as error:
                raise UserSafeError(
                    "The user defined in the containers USER instruction "
                    "does not exist"
                ) from error

            try:
                user = pwd.getpwuid(uid)
            except (KeyError, AttributeError):
                return UserInfo(uid=uid, gid=None, home=None, groups=[])

        return UserInfo(
            uid=user.pw_uid,
            gid=user.pw_gid,
            home=user.pw_dir,
            groups=cls._get_users_groups(user=user),
        )

    @classmethod
    def _get_users_groups(cls, *, user: pwd.struct_passwd) -> list[int]:
        users_groups = [
            g.gr_gid for g in grp.getgrall() if user.pw_name in g.gr_mem
        ]
        return cls._put_gid_first(gid=user.pw_gid, groups=users_groups)

    @staticmethod
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

    @staticmethod
    def _get_group_id(id_or_name: str | None) -> int | None:
        if id_or_name is None:
            return None

        try:
            return grp.getgrnam(id_or_name).gr_gid
        except (KeyError, AttributeError):
            try:
                return int(id_or_name)
            except ValueError as error:
                raise UserSafeError(
                    "The group defined in the containers USER "
                    "instruction does not exist"
                ) from error


def clean_path(path: Path) -> None:
    if not path.exists():
        return

    for entry in path.iterdir():
        full_path = entry.resolve()

        if entry.is_symlink():
            try:
                entry.chmod(0o700, follow_symlinks=False)
            except NotImplementedError as error:
                if (
                    str(error)
                    == "chmod: follow_symlinks unavailable on this platform"
                ):
                    pass
                else:
                    raise
            entry.unlink()

        if full_path.is_file():
            full_path.chmod(0o700)
            full_path.unlink()
        elif full_path.is_dir():
            full_path.chmod(0o700)
            clean_path(path=full_path)
            full_path.rmdir()


class S3File(NamedTuple):
    bucket: str
    key: str


class S3Resources(NamedTuple):
    semaphore: Semaphore
    client: S3Client


@asynccontextmanager
async def get_s3_resources() -> AsyncIterator[S3Resources]:
    semaphore = asyncio.Semaphore(
        int(
            os.environ.get("GRAND_CHALLENGE_COMPONENT_ASYNC_CONCURRENCY", "50")
        )
    )
    session = aioboto3.Session()
    boto_config = Config(
        max_pool_connections=int(
            os.environ.get(
                "GRAND_CHALLENGE_COMPONENT_BOTO_MAX_POOL_CONNECTIONS", "120"
            )
        )
    )

    async with session.client(
        "s3",
        endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL"),
        config=boto_config,
    ) as client:
        yield S3Resources(semaphore=semaphore, client=client)


def parse_s3_uri(*, s3_uri: str) -> S3File:
    pattern = r"^(https|s3)://(?P<bucket>[^/]+)/?(?P<key>.*)$"
    match = re.fullmatch(pattern, s3_uri)

    if not match:
        raise ValueError(f"Not a valid S3 uri, must match pattern {pattern}")

    return S3File(bucket=match.group("bucket"), key=match.group("key"))


async def get_s3_file_content(
    *, s3_uri: str, s3_resources: S3Resources
) -> bytes:
    s3_file = parse_s3_uri(s3_uri=s3_uri)

    content = io.BytesIO()

    async with s3_resources.semaphore:
        await s3_resources.client.download_fileobj(
            Fileobj=content,
            Bucket=s3_file.bucket,
            Key=s3_file.key,
        )

    content.seek(0)

    return content.read()


def validate_bucket_name(v: str) -> str:
    if BUCKET_NAME_REGEX.match(v) or BUCKET_ARN_REGEX.match(v):
        return v
    else:
        raise ValueError("Invalid bucket name")


class ProcUserTarfile(ProcUserMixin, tarfile.TarFile):
    """
    A tarfile that sets the owner of the extracted files to the user and group
    specified in the GRAND_CHALLENGE_COMPONENT_USER environment variable.
    """

    def chown(
        self,
        tarinfo: tarfile.TarInfo,
        targetpath: StrOrBytesPath,
        numeric_owner: bool,
    ) -> None:
        logger.info(f"chown of extracted {targetpath=}")

        if self.proc_user.uid is None and self.proc_user.gid is None:
            # No user or group specified, use the default
            return super().chown(
                tarinfo=tarinfo,
                targetpath=targetpath,
                numeric_owner=numeric_owner,
            )
        else:
            # Do not change owner if the user or group is not set
            uid = -1 if self.proc_user.uid is None else self.proc_user.uid
            gid = -1 if self.proc_user.gid is None else self.proc_user.gid

            logger.debug(f"Changing owner of {targetpath=} to {uid=}, {gid=}")

            os.chown(path=targetpath, uid=uid, gid=gid)


async def download_and_extract_tarball(
    *, s3_uri: str, dest: Path, s3_resources: S3Resources
) -> None:
    s3_file = parse_s3_uri(s3_uri=s3_uri)

    with SpooledTemporaryFile(max_size=4 * 1024 * 1024 * 1024) as f:
        async with s3_resources.semaphore:
            await s3_resources.client.download_fileobj(
                Bucket=s3_file.bucket,
                Key=s3_file.key,
                Fileobj=f,
            )

        f.seek(0)

        with ProcUserTarfile.open(fileobj=f, mode="r") as tar:
            tar.extractall(path=dest, filter="data")


class AuxiliaryData:
    def __init__(self, *, s3_resources: S3Resources):
        self._s3_resources = s3_resources

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
                "GRAND_CHALLENGE_COMPONENT_WRITABLE_DIRECTORIES",
                "/opt/ml/output/data:/opt/ml/model:/opt/ml/input/data/ground_truth:/opt/ml/checkpoints:/tmp",
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
                "GRAND_CHALLENGE_COMPONENT_POST_CLEAN_DIRECTORIES",
                "/opt/ml/output/data:/opt/ml/model:/opt/ml/input/data/ground_truth",
            ).split(":")
            if d
        ]
        logger.debug(f"{post_clean_directories=}")
        return post_clean_directories

    async def __aenter__(self) -> "AuxiliaryData":
        logger.info("Setting up Auxiliary Data")

        self.ensure_directories_are_writable()

        async with asyncio.TaskGroup() as task_group:
            task_group.create_task(self.download_model())
            task_group.create_task(self.download_ground_truth())

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        logger.info("Cleaning up Auxiliary Data")
        for p in self.post_clean_directories:
            logger.info(f"Cleaning {p=}")
            clean_path(p)

    def ensure_directories_are_writable(self) -> None:
        for directory in self.writable_directories:
            path = Path(directory)
            path.mkdir(exist_ok=True, parents=True)
            path.chmod(mode=0o777)

    async def download_model(self) -> None:
        if self.model_source is not None:
            logger.info(
                f"Downloading model from {self.model_source=} to {self.model_dest=}"
            )
            self.model_dest.mkdir(parents=True, exist_ok=True)
            await download_and_extract_tarball(
                s3_uri=self.model_source,
                dest=self.model_dest,
                s3_resources=self._s3_resources,
            )

    async def download_ground_truth(self) -> None:
        if self.ground_truth_source is not None:
            logger.info(
                f"Downloading ground truth from {self.ground_truth_source=} "
                f"to {self.ground_truth_dest=}"
            )
            self.ground_truth_dest.mkdir(parents=True, exist_ok=True)
            await download_and_extract_tarball(
                s3_uri=self.ground_truth_source,
                dest=self.ground_truth_dest,
                s3_resources=self._s3_resources,
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

    async def download(
        self, *, input_path: Path, s3_resources: S3Resources
    ) -> None:
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
                    async with s3_resources.semaphore:
                        await s3_resources.client.download_fileobj(
                            Bucket=self.bucket_name,
                            Key=self.bucket_key,
                            Fileobj=f,
                        )

                try:
                    safe_extract(src=zipfile, dest=dest_file.parent)
                except BadZipFile as error:
                    raise UserSafeError(
                        "Input zip file could not be extracted"
                    ) from error
                except OSError as error:
                    if error.errno == errno.ENOSPC:
                        raise UserSafeError(
                            "Contents of zip file too large"
                        ) from error
                    else:
                        raise error
        else:
            async with s3_resources.semaphore:
                await s3_resources.client.download_file(
                    Bucket=self.bucket_name,
                    Key=self.bucket_key,
                    Filename=dest_file,
                )

    async def upload(
        self, *, output_path: Path, s3_resources: S3Resources
    ) -> None:
        """Upload this file to s3"""
        src_file = str(self.local_file(path=output_path))

        logger.info(
            f"Uploading {src_file=} to {self.bucket_name=} with {self.bucket_key=}"
        )

        async with s3_resources.semaphore:
            await s3_resources.client.upload_file(
                Filename=src_file,
                Bucket=self.bucket_name,
                Key=self.bucket_key,
            )


class InferenceResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    pk: str
    return_code: int
    exec_duration: timedelta | None
    invoke_duration: timedelta | None
    outputs: list[InferenceIO]
    sagemaker_shim_version: str = version("sagemaker-shim")


class InferenceTask(ProcUserMixin, BaseModel):
    model_config = ConfigDict(frozen=True)

    pk: str
    inputs: list[InferenceIO]
    output_bucket_name: str
    output_prefix: str
    timeout: timedelta

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
    def input_path(self) -> Path:
        """Local path where the subprocess is expected to read its input files"""
        input_path = Path(
            os.environ.get("GRAND_CHALLENGE_COMPONENT_INPUT_PATH", "/input")
        )
        logger.debug(f"{input_path=}")
        return input_path

    @property
    def linked_input_path(self) -> Path:
        """Local path where the input files will be placed and linked to"""
        linked_input_parent = Path(
            os.environ.get(
                "GRAND_CHALLENGE_COMPONENT_LINKED_INPUT_PARENT",
                "/opt/ml/input/data/",
            )
        )
        linked_input_path = linked_input_parent / f"{self.pk}-input"
        logger.debug(f"{linked_input_path=}")
        return linked_input_path

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
        # We include some secret values in the environment so
        # ensure that they're not passed through to the subprocess
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.casefold().startswith("grand_challenge_")
        }

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

    async def invoke(self, *, s3_resources: S3Resources) -> InferenceResult:
        """Run the inference on a single case"""
        logger.info(f"Awaiting lock for {self.pk=}")

        await asyncio.wait_for(lock.acquire(), timeout=1.0)

        try:
            logger.info(f"Invoking {self.pk=}")
            inference_result = await self._invoke(s3_resources=s3_resources)
            await self.upload_inference_result(
                inference_result=inference_result, s3_resources=s3_resources
            )
        finally:
            lock.release()

        logger.info(f"Invocation {self.pk=} complete")

        return inference_result

    async def _invoke(  # noqa:C901
        self, *, s3_resources: S3Resources
    ) -> InferenceResult:
        try:
            self.reset_io()

            try:
                await self.download_input(s3_resources=s3_resources)
            except ExceptionGroup as exception_group:
                user_safe_errors, rest = exception_group.split(UserSafeError)

                if user_safe_errors:
                    for exception in user_safe_errors.exceptions:
                        self.log_external(
                            level=logging.ERROR, msg=str(exception)
                        )
                    return InferenceResult(
                        pk=self.pk,
                        return_code=1,
                        outputs=[],
                        exec_duration=None,
                        invoke_duration=None,
                    )

                if rest:
                    raise rest

            logger.info(f"Calling {self.proc_args=}")

            exec_start = time.monotonic()

            try:
                return_code = await asyncio.wait_for(
                    self.execute(), timeout=self.timeout.total_seconds()
                )
            except UserSafeError as error:
                self.log_external(level=logging.ERROR, msg=str(error))
                return_code = 1
            except TimeoutError:
                self.log_external(
                    level=logging.ERROR, msg="Time limit exceeded"
                )
                return_code = 1

            exec_duration = time.monotonic() - exec_start

            logger.info(f"{return_code=}")

            if return_code == 0:
                outputs = await self.upload_output(s3_resources=s3_resources)
            else:
                outputs = set()

            return InferenceResult(
                pk=self.pk,
                return_code=return_code,
                outputs=outputs,
                exec_duration=timedelta(seconds=exec_duration),
                invoke_duration=None,
            )
        finally:
            self.reset_io()

    def reset_io(self) -> None:
        """Resets the input and output directories"""
        try:
            clean_path(path=self.input_path)
            clean_path(path=self.output_path)
            self.reset_linked_input()
        except Exception as error:
            logger.critical(f"Could not reset io: {error}")
            raise UserSafeError(
                "The containers input and output directories could not be reset"
            ) from error

    def reset_linked_input(self) -> None:
        """Resets the symlink from the input to the linked directory"""
        if (
            os.environ.get(
                "GRAND_CHALLENGE_COMPONENT_USE_LINKED_INPUT", "True"
            ).lower()
            == "true"
        ):
            logger.info(
                f"Setting up linked input from {self.input_path} "
                f"to {self.linked_input_path}"
            )

            if self.input_path.exists():
                if self.input_path.is_symlink():
                    self.input_path.unlink()
                elif self.input_path.is_dir():
                    self.input_path.rmdir()

            if self.linked_input_path.exists():
                self.linked_input_path.rmdir()

            self.linked_input_path.mkdir(parents=True)
            self.linked_input_path.chmod(0o755)

            self.input_path.symlink_to(
                self.linked_input_path, target_is_directory=True
            )
            self.input_path.chmod(0o755)

    async def download_input(self, *, s3_resources: S3Resources) -> None:
        """Download all the inputs to the input path"""
        async with asyncio.TaskGroup() as task_group:
            for input_file in self.inputs:
                task_group.create_task(
                    input_file.download(
                        input_path=self.input_path, s3_resources=s3_resources
                    )
                )

    async def upload_output(
        self, *, s3_resources: S3Resources
    ) -> set[InferenceIO]:
        """Upload all the outputs from the output path to s3"""
        output_path = self.output_path
        outputs: set[InferenceIO] = set()

        async with asyncio.TaskGroup() as task_group:
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
                    task_group.create_task(
                        output.upload(
                            output_path=self.output_path,
                            s3_resources=s3_resources,
                        )
                    )
                    outputs.add(output)

        return outputs

    async def upload_inference_result(
        self, *, inference_result: InferenceResult, s3_resources: S3Resources
    ) -> None:
        content = inference_result.model_dump_json().encode("utf-8")
        signature = hmac.new(
            key=bytes.fromhex(
                os.environ.get("GRAND_CHALLENGE_COMPONENT_SIGNING_KEY_HEX", "")
            ),
            msg=content,
            digestmod=hashlib.sha256,
        ).hexdigest()

        bucket_key = (
            f"{self.output_prefix}.sagemaker_shim/inference_result.json"
        )

        logger.info(
            f"Uploading {bucket_key=} in "
            f"{self.output_bucket_name=} with {inference_result=}"
        )

        async with s3_resources.semaphore:
            await s3_resources.client.upload_fileobj(
                Fileobj=io.BytesIO(content),
                Bucket=self.output_bucket_name,
                Key=bucket_key,
                ExtraArgs={
                    "Metadata": {"signature_hmac_sha256": signature},
                },
            )

    async def execute(self) -> int:
        """
        Run the original entrypoint and command in a subprocess

        This needs to be as lean as possible as the method is timed
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *self.proc_args,
                user=self.proc_user.uid,
                group=self.proc_user.gid,
                extra_groups=self.extra_groups,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.proc_env,
                # A new process group must be used for cancellation
                start_new_session=True,
                # The following should always be set to protect this environment
                shell=False,
                close_fds=True,
            )
        except PermissionError as error:
            raise UserSafeError(
                "The user defined in the containers USER instruction "
                "does not have permission to execute the command "
                "defined by the containers ENTRYPOINT and CMD instructions"
            ) from error
        except FileNotFoundError as error:
            raise UserSafeError(
                "The command defined by the containers ENTRYPOINT and "
                "CMD instructions does not exist"
            ) from error

        stdout_task = asyncio.create_task(
            self._stream_to_external(stream=process.stdout, level=STDOUT_LEVEL)
        )
        stderr_task = asyncio.create_task(
            self._stream_to_external(
                stream=process.stderr, level=STDOUT_LEVEL + 10
            )
        )

        try:
            await asyncio.gather(stdout_task, stderr_task)
            return await process.wait()

        except asyncio.CancelledError:
            logger.info("Execution was cancelled")
            # shield so termination completes even if cancellation continues
            await asyncio.shield(
                self._terminate_group_and_wait(process=process)
            )
            await self._cancel_tasks(tasks=(stdout_task, stderr_task))
            raise

        except Exception as error:
            logger.critical(f"Exception in execution: {error}")
            await asyncio.shield(
                self._terminate_group_and_wait(process=process)
            )
            await self._cancel_tasks(tasks=(stdout_task, stderr_task))
            raise

        finally:
            # best-effort final cleanup; shielded to avoid being interrupted
            await asyncio.shield(
                self._terminate_group_and_wait(process=process)
            )

    @staticmethod
    async def _cancel_tasks(*, tasks: Iterable[asyncio.Task[None]]) -> None:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def _terminate_group_and_wait(  # noqa:C901
        *, process: asyncio.subprocess.Process
    ) -> None:
        if process.returncode is not None:
            return

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            try:
                process.terminate()
            except Exception:
                logger.info("process.terminate() failed", exc_info=True)

        try:
            # Wait for graceful termination
            await asyncio.wait_for(
                process.wait(),
                timeout=int(
                    os.environ.get(
                        "GRAND_CHALLENGE_COMPONENT_SIGTERM_GRACE_SECONDS", "5"
                    )
                ),
            )
            logger.info("Process group terminated")
            return
        except TimeoutError:
            logger.warning(
                "Process group did not exit within grace period; "
                "escalating to SIGKILL"
            )
        except Exception:
            logger.info(
                "Error while waiting for process.wait()", exc_info=True
            )

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            try:
                process.kill()
            except Exception:
                logger.info("process.kill() failed", exc_info=True)

        try:
            await process.wait()
            logger.info("Process group killed")
        except Exception:
            logger.warning(
                "Failed awaiting process.wait() after kill", exc_info=True
            )

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
