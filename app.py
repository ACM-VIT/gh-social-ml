"""Canonical ASGI entrypoint for the GitSocial ML V2 service."""

import uvicorn

from api.main import app

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
