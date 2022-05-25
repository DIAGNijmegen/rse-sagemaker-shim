import asyncio
import json
import logging
import logging.config
import sys
from functools import wraps
from typing import Any, Callable, Iterable
from uuid import UUID

import click
import uvicorn

from sagemaker_shim.app import app
from sagemaker_shim.logging import LOGGING_CONFIG
from sagemaker_shim.models import InferenceIO, InferenceTask

logger = logging.getLogger(__name__)


def cli_coroutine(f: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@click.group()
def cli() -> None:
    pass


@cli.command(short_help="Start the model server")
def serve() -> None:
    logging.config.dictConfig(LOGGING_CONFIG)
    uvicorn.run(app=app, host="0.0.0.0", port=8080, log_config=None)


@cli.command(short_help="Run one inference task")
@click.option(
    "--pk",
    required=True,
    type=UUID,
    help="The primary key of the task",
)
@click.option(
    "--input-file",
    type=str,
    multiple=True,
    help="An input to this task",
)
@click.option(
    "--output-bucket-name",
    required=True,
    type=str,
    help="The name of the bucket where the results will be written",
)
@click.option(
    "--output-prefix",
    required=True,
    type=str,
    help="The prefix for the output data from this task",
)
@cli_coroutine
async def invoke(
    pk: UUID,
    input_file: Iterable[str],
    output_bucket_name: str,
    output_prefix: str,
) -> None:
    logging.config.dictConfig(LOGGING_CONFIG)

    inputs = []
    for input_json in input_file:
        # TODO add a test
        inputs.append(InferenceIO(**json.loads(input_json)))

    task = InferenceTask(
        pk=pk,
        inputs=inputs,
        output_bucket_name=output_bucket_name,
        output_prefix=output_prefix,
    )

    result = await task.invoke()

    sys.exit(result.return_code)


if __name__ == "__main__":
    # https://pyinstaller.org/en/stable/runtime-information.html#run-time-information
    we_are_bundled = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

    if we_are_bundled:
        # https://pyinstaller.org/en/stable/runtime-information.html#using-sys-executable-and-sys-argv-0
        cli(sys.argv[1:])
    else:
        cli()
