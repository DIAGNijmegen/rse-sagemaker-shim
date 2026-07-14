import os
from collections.abc import Callable
from pathlib import Path

OOM_SCORE_ADJ_MAX = 1000


def make_child_oom_preexec_fn() -> Callable[[], None]:
    """Return a preexec_fn that makes the child the preferred OOM kill target.

    This runs in the child process after fork() but before exec().
    Setting oom_score_adj to the maximum value (1000) ensures the kernel's
    OOM killer will prefer killing this process (and its descendants) over
    the parent sagemaker-shim process when the container's memory limit
    is reached.

    This value is inherited by any grandchild processes.
    """

    def _adjust_oom_score() -> None:
        try:
            Path(f"/proc/{os.getpid()}/oom_score_adj").write_text(
                str(OOM_SCORE_ADJ_MAX)
            )
        except OSError:
            # Non-fatal: we may not be on Linux or lack permissions.
            # Logging is not safe in preexec_fn (runs post-fork),
            # so we silently ignore.
            pass

    return _adjust_oom_score
