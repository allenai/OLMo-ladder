# Base image comes with PyTorch, numpy, flash-attn
ARG BASE
FROM ${BASE} as stable

# Install other dependencies, but not the source code.
COPY pyproject.toml .
COPY src/my_package/__init__.py src/my_package/__init__.py
COPY src/my_package/version.py src/my_package/version.py
RUN pip install --no-cache-dir '.[all]' && \
    pip uninstall -y my-package && \
    rm -rf *

WORKDIR /app/my-package

FROM stable as nightly

ARG NIGHTLY_VERSION
RUN pip install --no-cache-dir --pre torch==${NIGHTLY_VERSION}
