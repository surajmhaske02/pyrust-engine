# Deploy PyRust-compiled code to a slim image that has no Rust.
#   docker build -t pyrust-demo examples/docker && docker run --rm pyrust-demo
#
# Stage 1 installs Rust and compiles the functions ahead of time into app/__pyrust__/.
# Stage 2 copies the app with that folder; PyRust loads the prebuilt libraries there.
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir "pyrust-engine[arrays]"
COPY app /app
RUN python -m pyrust_engine build /app/kernels.py

FROM python:3.12-slim
RUN pip install --no-cache-dir "pyrust-engine[arrays]"
COPY --from=build /app /app
WORKDIR /app
CMD ["python", "main.py"]
