FROM vastai/pytorch:2.12.0-cu126-cuda-12.9-mini-py312-2026-06-15 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    HOBNOB_INSTALL_DIR=/usr/local/bin \
    PYTHONPATH=/root \
    PYTHONUNBUFFERED=1 \
    PATH=/venv/main/bin:$PATH \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates unzip jq \
    && rm -rf /var/lib/apt/lists/*

RUN bash -o pipefail -c "curl -fsSL https://raw.githubusercontent.com/jakesmd/hobnob/main/install.sh | bash" \
    && which hobnob

ARG ORAS_VERSION=1.3.3
RUN curl -fsSL "https://github.com/oras-project/oras/releases/download/v${ORAS_VERSION}/oras_${ORAS_VERSION}_linux_amd64.tar.gz" \
    | tar -xz -C /usr/local/bin oras

RUN /venv/main/bin/pip install --no-cache-dir python-chess numpy apache-beam

WORKDIR /root


FROM base AS training

COPY hobnob.yml /root/hobnob.yml
COPY instinct_chess_bot/ /root/instinct_chess_bot/
COPY train/ /root/train/
