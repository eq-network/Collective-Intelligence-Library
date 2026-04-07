FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app
COPY . .

RUN python -m venv .venv \
    && . .venv/bin/activate \
    && pip install --no-cache-dir -e ".[cuda]"

ENV PATH="/app/.venv/bin:$PATH"

# Default: quick smoke test. Override with docker run args.
CMD ["python", "-m", "experiments.basin_stability.run_experiment", "--quick", "--vmap"]
