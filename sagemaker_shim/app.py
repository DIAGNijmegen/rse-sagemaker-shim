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

import asyncio
import logging
import os
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status

from sagemaker_shim.exceptions import UserSafeError
from sagemaker_shim.models import (
    AuxiliaryData,
    InferenceResult,
    InferenceTask,
    RuntimeSetupResult,
    UserProcess,
    get_s3_resources,
)

logger = logging.getLogger(__name__)


USER_PROCESS: UserProcess | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with get_s3_resources() as s3_resources:
        global USER_PROCESS

        auxiliary_data = AuxiliaryData(s3_resources=s3_resources)
        USER_PROCESS = UserProcess()

        try:
            await auxiliary_data.setup()
        except UserSafeError as error:
            logger.error(msg=str(error), extra={"internal": False})
            runtime_setup_result = RuntimeSetupResult(
                return_code=1, user_safe_error_message=str(error)
            )
            await asyncio.shield(
                runtime_setup_result.upload(s3_resources=s3_resources)
            )
            # If subprocess errors are handled our process should exit cleanly
            raise SystemExit(0) from error

        try:
            await USER_PROCESS.setup()
        except UserSafeError as error:
            logger.error(msg=str(error), extra={"internal": False})
            runtime_setup_result = RuntimeSetupResult(
                return_code=1, user_safe_error_message=str(error)
            )
            await asyncio.shield(
                runtime_setup_result.upload(s3_resources=s3_resources)
            )
            # If subprocess errors are handled our process should exit cleanly
            raise SystemExit(0) from error

        runtime_setup_result = RuntimeSetupResult(return_code=0)
        await asyncio.shield(
            runtime_setup_result.upload(s3_resources=s3_resources)
        )

        try:
            yield
        finally:
            await auxiliary_data.teardown()
            await USER_PROCESS.teardown()


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
async def ping() -> Response:
    logger.debug("ping called")

    if USER_PROCESS is None or not USER_PROCESS.healthy:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    return Response(status_code=status.HTTP_200_OK)


@app.get("/execution-parameters")
async def execution_parameters() -> dict[str, int | str]:
    logger.debug("execution_parameters called")

    return {
        "MaxConcurrentTransforms": 1,
        "BatchStrategy": "SINGLE_RECORD",
        "MaxPayloadInMB": 6,
    }


@app.post("/invocations", response_model=InferenceResult)
async def invocations(task: InferenceTask) -> InferenceResult | Response:
    logger.debug("invocations called")
    logger.debug(f"{task=}")

    if USER_PROCESS is None:
        raise RuntimeError("USER_PROCESS should be initialized")

    if not USER_PROCESS.healthy:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    async with get_s3_resources() as s3_resources:
        result = await task.run_inference(
            user_process=USER_PROCESS, s3_resources=s3_resources
        )

    if not USER_PROCESS.healthy:
        logger.error("Container unhealthy, terminating process")
        os.kill(os.getpid(), signal.SIGTERM)

    return result
