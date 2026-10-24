import uvicorn

from nexus.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "nexus.api.app:app",
        host="0.0.0.0",
        port=settings.gateway_port,
        reload=True,
    )
