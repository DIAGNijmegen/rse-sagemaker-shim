import json
import logging
import os

logger = logging.getLogger(__name__)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record=record)

        if record.levelno <= logging.INFO:
            source = "stdout"
        else:
            source = "stderr"

        internal = getattr(record, "internal", True)
        task = getattr(record, "task", None)
        task_pk = str(getattr(task, "pk", None))

        # TODO find a way to test the logging

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


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": JSONFormatter,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        },
    },
    "root": {
        "level": os.environ.get("LOGLEVEL", "INFO").upper(),
        "handlers": ["console"],
    },
}
