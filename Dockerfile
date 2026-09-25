# Container for the codeQandA web app (used by Render; works on any Docker host).
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Dependencies first, so code changes don't reinstall them.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# The repo the demo answers questions about, pinned to the commit the published eval measured,
# so the numbers, the cached example answers and the live index all describe the same code.
ARG TARGET_REPO=https://github.com/h4rshini/Signal-stockwatchlist.git
ARG TARGET_REF=a37f0cc4f826eb30c2e4ee0aab34e9d4bf4ae2fa
RUN git init -q /home/user/Signal-stockwatchlist && cd /home/user/Signal-stockwatchlist \
    && git fetch -q --depth 1 "$TARGET_REPO" "$TARGET_REF" && git checkout -q FETCH_HEAD

COPY --chown=user . .

# Index at build time: also downloads the local embedding model, so the first visitor doesn't wait.
RUN python -m codeqa.indexer /home/user/Signal-stockwatchlist

# The API key comes from the host's secret env vars at runtime; it is never part of the image.
ENV CODEQA_TRACE_DIR=/tmp/traces
EXPOSE 8000
CMD ["sh", "-c", "uvicorn web.app:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*'"]
