"""
Environment variables will be passed from those set in CreateModel or
CreateTransformJob. Additionally:
  - SAGEMAKER_BATCH
      always set to true when the container runs in Batch Transform
  - SAGEMAKER_MAX_PAYLOAD_IN_MB
      set to the largest size payload that is sent to the container via HTTP
  - SAGEMAKER_BATCH_STRATEGY
      set to SINGLE_RECORD when the container is sent a single record
      per call to invocations and MULTI_RECORD when the container gets
      as many records as will fit in the payload
  - SAGEMAKER_MAX_CONCURRENT_TRANSFORMS
      set to the maximum number of /invocations requests that can
      be opened simultaneously

Notes:
  - /opt/ml/model will contain the model weights, this can be an empty .tar.gz file
"""

import logging

from fastapi import FastAPI, Response, status

from sagemaker_shim.models import InferenceResult, InferenceTask

logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/ping")
async def ping() -> Response:
    logger.debug("ping called")

    return Response(status_code=status.HTTP_200_OK)


@app.get("/execution-parameters")
async def execution_parameters() -> dict[str, int | str]:
    logger.debug("execution_parameters called")

    return {
        "MaxConcurrentTransforms": 1,
        "BatchStrategy": "SINGLE_RECORD",
        "MaxPayloadInMB": 6,
    }


@app.post("/invocations")
async def invocations(task: InferenceTask) -> InferenceResult:
    logger.debug("invcations called")
    logger.debug(f"{task=}")

    return await task.invoke()
