import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

STDOUT_LEVEL = logging.INFO


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        """
        Create a structured log message

        CloudWatch does not separate log streams, so we use source
        to differentiate between stdout and stderr.
        Internal allows us to differentiate between logs from
        this application, and logs from the invocation subprocess.
        """
        message = super().format(record=record)

        # We need to explicitly add the source as an annotation
        # as CloudWatch unifies the logs
        if record.levelno <= STDOUT_LEVEL:
            source = "stdout"
        else:
            source = "stderr"

        internal = getattr(record, "internal", True)
        task = getattr(record, "task", None)
        task_pk = getattr(task, "pk", None)

        return "\n".join(
            json.dumps(
                {
                    "log": m,
                    "level": record.levelname,
                    "source": source,
                    "internal": internal,
                    "task": task_pk,
                }
            )
            for m in message.splitlines()
        )


class StdStreamFilter(logging.Filter):
    """Split stdout and stderr streams"""

    def __init__(self, *args: Any, stdout: bool, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__stdout = stdout

    def filter(self, record: logging.LogRecord) -> bool:
        """Should this log message be displayed?"""
        if self.__stdout:
            # stdout, STDOUT_LEVEL and lower
            return record.levelno <= STDOUT_LEVEL
        else:
            # stderr, greater than STDOUT_LEVEL
            return record.levelno > STDOUT_LEVEL


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "stdout": {
            "()": StdStreamFilter,
            "stdout": True,
        },
        "stderr": {
            "()": StdStreamFilter,
            "stdout": False,
        },
    },
    "formatters": {
        "json": {
            "()": JSONFormatter,
        },
    },
    "handlers": {
        "console_stdout": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": "ext://sys.stdout",
            "filters": ["stdout"],
        },
        "console_stderr": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": "ext://sys.stderr",
            "filters": ["stderr"],
        },
    },
    "root": {
        "level": os.environ.get("LOG_LEVEL", "INFO").upper(),
        "handlers": ["console_stdout", "console_stderr"],
    },
}
