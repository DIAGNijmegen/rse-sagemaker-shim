import asyncio
import logging
import logging.config
import sys
from functools import wraps
from typing import Any, Callable

import click
import uvicorn

from sagemaker_shim.app import app
from sagemaker_shim.logging import LOGGING_CONFIG

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


if __name__ == "__main__":
    # https://pyinstaller.org/en/stable/runtime-information.html#run-time-information
    we_are_bundled = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

    if we_are_bundled:
        # https://pyinstaller.org/en/stable/runtime-information.html#using-sys-executable-and-sys-argv-0
        cli(sys.argv[1:])
    else:
        cli()
