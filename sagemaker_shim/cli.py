import logging.config
import sys

import click
import uvicorn

from sagemaker_shim.app import app
from sagemaker_shim.logging import LOGGING_CONFIG


@click.group()
def cli() -> None:
    pass


@cli.command(short_help="Start the model server")
def serve() -> None:
    logging.config.dictConfig(LOGGING_CONFIG)
    uvicorn.run(app=app, host="0.0.0.0", port=8080, log_config=None, workers=1)


if __name__ == "__main__":
    # https://pyinstaller.org/en/stable/runtime-information.html#run-time-information
    we_are_bundled = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

    if we_are_bundled:
        # https://pyinstaller.org/en/stable/runtime-information.html#using-sys-executable-and-sys-argv-0
        cli(sys.argv[1:])
    else:
        cli()
