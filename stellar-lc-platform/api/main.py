from fastapi import FastAPI

app = FastAPI(
    title="Stellar LC Platform API",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    """Simple endpoint used to verify that the backend is running."""
    return {"status": "ok"}
