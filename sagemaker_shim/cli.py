import asyncio
import logging.config
import sys
from collections.abc import Callable, Coroutine
from functools import wraps
from json import JSONDecodeError
from typing import Any, TypeVar

import click
import uvicorn
from botocore.exceptions import ClientError, NoCredentialsError
from pydantic import ValidationError

from sagemaker_shim.app import app
from sagemaker_shim.logging import LOGGING_CONFIG
from sagemaker_shim.models import (
    DependentData,
    InferenceTaskList,
    get_s3_file_content,
)

T = TypeVar("T")


def async_to_sync(
    func: Callable[..., Coroutine[Any, Any, T]]
) -> Callable[..., T]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return asyncio.run(func(*args, **kwargs))

    return wrapper


@click.group()
def cli() -> None:
    logging.config.dictConfig(LOGGING_CONFIG)


@cli.command(short_help="Start the model server")
def serve() -> None:
    with DependentData():
        uvicorn.run(
            app=app, host="0.0.0.0", port=8080, log_config=None, workers=1
        )


@cli.command(short_help="Invoke the model")
@click.option(
    "-t",
    "--tasks",
    default=None,
    help="A JSON string of task definitions",
    type=str,
)
@click.option(
    "-f",
    "--file",
    default=None,
    help="S3 URI of a JSON file containing a list of task definitions",
    type=str,
)
@async_to_sync
async def invoke(tasks: str, file: str) -> None:
    tasks_json: str | bytes

    if tasks and file:
        raise click.UsageError("Only one of tasks or file should be set")
    elif tasks:
        tasks_json = tasks
    elif file:
        try:
            tasks_json = get_s3_file_content(s3_uri=file)
        except (ValueError, ClientError, NoCredentialsError) as error:
            raise click.BadParameter(
                f"The value provided for file is invalid:\n\n{error}"
            ) from error
    else:
        raise click.UsageError("One of tasks or file should be set")

    try:
        parsed_tasks = InferenceTaskList.model_validate_json(tasks_json)
    except (ValidationError, JSONDecodeError) as error:
        raise click.BadParameter(
            f"The tasks definition is invalid:\n\n{error}"
        ) from error

    if parsed_tasks.root:
        with DependentData():
            for task in parsed_tasks.root:
                # Only run one task at a time
                await task.invoke()
    else:
        raise click.UsageError("Empty task list provided")


if __name__ == "__main__":
    # https://pyinstaller.org/en/stable/runtime-information.html#run-time-information
    we_are_bundled = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

    if we_are_bundled:
        # https://pyinstaller.org/en/stable/runtime-information.html#using-sys-executable-and-sys-argv-0
        cli(sys.argv[1:])
    else:
        cli()
