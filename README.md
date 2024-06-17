# SageMaker Shim for Grand Challenge

[![CI](https://github.com/jmsmkn/sagemaker-shim/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jmsmkn/sagemaker-shim/actions/workflows/ci.yml?query=branch%3Amain)
[![PyPI](https://img.shields.io/pypi/v/sagemaker-shim)](https://pypi.org/project/sagemaker-shim/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sagemaker-shim)](https://pypi.org/project/sagemaker-shim/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repo contains a library that adapts algorithms that implement the Grand Challenge inference API for running in SageMaker.

The application contains:

- A `click` cli client with options to launch a web server
- A `fastapi` web server that implements the SageMaker endpoints
- and `pydantic` models that interface between S3, and run the original inference jobs.

The application is compiled on Python 3.11 using `pyinstaller`, and then distributed as a statically linked binary using `staticx`.
It is able to adapt any container, including ones based on `scratch` or `alpine` images.

## Usage

The binary is designed to be added to an existing container image that implements the Grand Challenge API.
On Grand Challenge this happens automatically by using [crane](https://github.com/google/go-containerregistry/blob/main/cmd/crane/doc/crane_mutate.md) to add the binary, directories and environment variables to each container image.
The binary itself will:

1. Download the input files from the provided locations on S3 to `/input`, optionally decompressing the inputs.
1. Execute the original container program in a subprocess.
   This is found by inspecting the following environment variables:
    - `GRAND_CHALLENGE_COMPONENT_CMD_B64J`: the original `cmd` of the container, json encoded as a base64 string.
    - `GRAND_CHALLENGE_COMPONENT_ENTRYPOINT_B64J`: the original `entrypoint` of the container, json encoded as a base64 string.
1. Upload the contents of `/output` to the given output S3 bucket and prefix.

### Logging

CloudWatch does not offer separation of `stdout` and `stderr` by default.
`sagemaker-shim` includes a logging filter and formatter that creates structured logs from the application and subprocess.
This allows grand challenge to separate out internal, external, stdout and stderr streams.
These structured logs are JSON objects with the format:

```js
{
  "log": "",  // The original log message
  "level": "CRITICAL" | "ERROR" | "WARNING" | "INFO" | "DEBUG" | "NOTSET",  // The severity level of the log
  "source": "stdout" | "stderr",   // The source stream
  "internal": true | false,  // Whether the source of the log is from sagemaker shim or the subprocess
  "task": "" | null,  // The ID of the task
}
```

### `sagemaker-shim serve`

This starts the webserver on http://0.0.0.0:8080 which implements the [SageMaker API](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-batch-code.html).
There are three endpoints:

- `/ping` (GET): returns an empty 200 response if the container is healthy
- `/execution-parameters` (GET): returns the preferred execution parameters for AWS SageMaker Batch Inference
- `/invocations` (POST): SageMaker can make POST requests to this endpoint.
  The body contains the json encoded data required to run a single inference task:

  ```json
    {
        "pk": "unique-test-id",
        "inputs": [
            {
                "relative_path": "interface/path",
                "bucket_name": "name-of-input-bucket",
                "bucket_key": "/path/to/input/file/in/bucket",
                "decompress": false,
            },
            ...
        ],
        "output_bucket_name": "name-of-output-bucket",
        "output_prefix": "/prefix/of/output/files",
    }
  ```

  The endpoint will return an object containing the return code of the subprocess in `response["return_code"]`,
  and any outputs will be placed in the output bucket at the output prefix. A file with the inference outputs
  will also be located at `s3://<output_bucket_name>/<output_prefix>/.sagemaker_shim/inference_result.json`

### `sagemaker-shim invoke`

This will invoke the model directly given the arguments. You can specify either:

- `-f / --file`: S3 URI of a JSON file containing a list of task definitions, e.g. `s3://my-bucket/invocations.json`
- `-t / --tasks`: A JSON string of task definitions

In both cases the contents of the file or string will be an array of task objects:

```json
[
    {
        "pk": "unique-test-id-1",
        "inputs": [
            ...
        ],
        "output_bucket_name": "name-of-output-bucket",
        "output_prefix": "/prefix/of/output/files-1",
    },
    {
        "pk": "unique-test-id-2",
        "inputs": [
            ...
        ],
        "output_bucket_name": "name-of-output-bucket",
        "output_prefix": "/prefix/of/output/files-2",
    }
]
```

A file with the inference outputs will be located at `s3://<output_bucket_name>/<output_prefix>/.sagemaker_shim/inference_result.json`.

### Patching an Existing Container

To patch an existing container image in a registry see the example in [tests/utils.py](tests/utils.py).
First you will need to get the original `cmd` and `entrypoint` using `get_new_env_vars` and `get_image_config`.
Then you can add the binary, set the new `cmd`, `entrypoint`, and environment variables with `mutate_image`.
