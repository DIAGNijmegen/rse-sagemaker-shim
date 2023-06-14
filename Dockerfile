FROM python:3.10

RUN mkdir -p /opt/src

WORKDIR /opt/src

RUN python -m pip install --upgrade pip && \
    python -m pip install poetry && \
    apt-get update && \
    apt-get install -y scons patchelf

ENV PATH="/opt/src/.venv/bin:${PATH}"\
    PYTHONPATH="/opt/src:${PYTHONPATH}"

COPY poetry.lock /opt/src
COPY pyproject.toml /opt/src

RUN poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-ansi --no-root
