"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(
    title="Industrial AI Platform",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe used by deployment and CI."""
    return {"status": "ok"}
