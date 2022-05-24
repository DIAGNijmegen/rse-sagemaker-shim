from uuid import uuid4

from click.testing import CliRunner

from patch_image import encode_b64j
from sagemaker_shim.cli import invoke


def test_invocations_cli(tmp_path, monkeypatch):
    runner = CliRunner()

    input_path = tmp_path / "input"
    output_path = tmp_path / "output"

    input_path.mkdir()
    output_path.mkdir()

    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_ENTRYPOINT_B64J",
        encode_b64j(val=["echo", "hello world"]),
    )
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_INPUT_PATH",
        str(input_path),
    )
    monkeypatch.setenv(
        "GRAND_CHALLENGE_COMPONENT_OUTPUT_PATH",
        str(output_path),
    )

    result = runner.invoke(
        invoke,
        [
            "--pk",
            str(uuid4()),
            "--output-bucket-name",
            "test",
            "--output-prefix",
            "test",
        ],
    )
    assert result.exit_code == 0
