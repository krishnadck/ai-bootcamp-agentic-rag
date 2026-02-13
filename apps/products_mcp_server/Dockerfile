FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

COPY apps/products_mcp_server ./apps/products_mcp_server

ENV UV_COMPILE_BYTECODE=1
ENV PYTHONOPTIMIZE=1
ENV UV_LINK_MODE=copy
ENV PYTHONBUFFERED=1


RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package products-mcp-server

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/apps/products_mcp_server:$PYTHONPATH"

RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    chown -R app:app /app

USER app

ENV HOME=/app

EXPOSE 8000

CMD ["python", "-m", "main"]
