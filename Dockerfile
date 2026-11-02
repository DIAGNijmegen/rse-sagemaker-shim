FROM python@sha256:17b9724493d916f56565c131f389faff2597f46dec6ce09516f95125bc9f519f

RUN mkdir -p /opt/src/dist

WORKDIR /opt/src

RUN python -m pip install --upgrade pip && \
    python -m pip install uv && \
    apt-get update && \
    apt-get install -y scons patchelf

COPY uv.lock /opt/src
COPY pyproject.toml /opt/src
COPY README.md /opt/src
COPY sagemaker_shim /opt/src/sagemaker_shim
COPY tests /opt/src/tests

RUN uv sync
