import os
from pathlib import Path
from unittest.mock import patch

import pytest

from sagemaker_shim.oom import OOM_SCORE_ADJ_MAX, make_child_oom_preexec_fn


def test_make_child_oom_preexec_fn_writes_oom_score_adj(tmp_path):
    """The preexec_fn writes the max OOM score to the process's
    oom_score_adj file."""
    fake_proc_path = tmp_path / "oom_score_adj"
    fake_proc_path.write_text("0")

    preexec_fn = make_child_oom_preexec_fn()

    with patch(
        "sagemaker_shim.oom.Path",
        side_effect=lambda p: (
            fake_proc_path if "oom_score_adj" in str(p) else Path(p)
        ),
    ):
        preexec_fn()

    assert fake_proc_path.read_text() == str(OOM_SCORE_ADJ_MAX)


def test_make_child_oom_preexec_fn_constructs_correct_path():
    """The preexec_fn targets /proc/<pid>/oom_score_adj."""
    preexec_fn = make_child_oom_preexec_fn()

    with patch("sagemaker_shim.oom.Path") as mock_path:
        mock_path.return_value.write_text = lambda _: None
        preexec_fn()

    mock_path.assert_called_once_with(f"/proc/{os.getpid()}/oom_score_adj")


def test_make_child_oom_preexec_fn_handles_oserror_gracefully(tmp_path):
    """The preexec_fn silently handles OSError (e.g. not on Linux)."""
    preexec_fn = make_child_oom_preexec_fn()

    with patch("sagemaker_shim.oom.Path") as mock_path:
        mock_path.return_value.write_text.side_effect = OSError(
            "Permission denied"
        )
        # Should not raise
        preexec_fn()


def test_make_child_oom_preexec_fn_returns_callable():
    """make_child_oom_preexec_fn returns a callable with no arguments."""
    preexec_fn = make_child_oom_preexec_fn()
    assert callable(preexec_fn)


@pytest.mark.skipif(
    not Path("/proc/self/oom_score_adj").exists(),
    reason="oom_score_adj not available (not Linux)",
)
def test_make_child_oom_preexec_fn_integration():
    """Integration test: verify the function writes to the real procfs
    when running on Linux."""
    preexec_fn = make_child_oom_preexec_fn()

    # Read the current value to restore after test
    oom_path = Path(f"/proc/{os.getpid()}/oom_score_adj")
    original_value = oom_path.read_text().strip()

    try:
        preexec_fn()
        new_value = oom_path.read_text().strip()
        assert new_value == str(OOM_SCORE_ADJ_MAX)
    finally:
        # Restore original value
        try:
            oom_path.write_text(original_value)
        except OSError:
            pass
