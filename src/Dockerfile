# Base image comes with PyTorch, numpy, flash-attn
ARG BASE
FROM ${BASE} as stable

# Install other dependencies, but not the source code.
COPY pyproject.toml .
COPY src/ladder/__init__.py src/ladder/__init__.py
COPY src/ladder/version.py src/ladder/version.py
RUN pip install --no-cache-dir '.[all]' && \
    pip uninstall -y ladder && \
    rm -rf *

WORKDIR /app/ladder

FROM stable as nightly

ARG NIGHTLY_VERSION
RUN pip install --no-cache-dir --pre torch==${NIGHTLY_VERSION}
