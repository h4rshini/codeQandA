# Hugging Face Space image for codeQandA.
# The app code comes from GitHub, so this Space repo only needs this Dockerfile and its README.md.
# Redeploy after pushing to GitHub: Space settings -> "Factory rebuild".
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Spaces run containers as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user

ARG APP_REPO=https://github.com/h4rshini/codeQandA.git
ARG TARGET_REPO=https://github.com/h4rshini/Signal-stockwatchlist.git
# Pinned to the commit the published eval was measured on, so the numbers, the cached
# example answers and the live index all describe the same code.
ARG TARGET_REF=a37f0cc4f826eb30c2e4ee0aab34e9d4bf4ae2fa

RUN git clone --depth 1 "$APP_REPO" app \
    && git init -q Signal-stockwatchlist && cd Signal-stockwatchlist \
    && git fetch -q --depth 1 "$TARGET_REPO" "$TARGET_REF" && git checkout -q FETCH_HEAD

WORKDIR /home/user/app
RUN pip install --no-cache-dir --user -r requirements.txt

# Index at build time: also downloads the local embedding model, so the first visitor doesn't wait.
RUN python -m codeqa.indexer /home/user/Signal-stockwatchlist

# The API key comes from the Space secret GEMINI_API_KEY at runtime; never baked into the image.
ENV CODEQA_TRACE_DIR=/tmp/traces
EXPOSE 7860
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
