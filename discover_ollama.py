import sys

import httpx

# -*- coding: utf-8 -*-
"""
Utilities for discovering a running Ollama daemon.

This module provides helpers to probe a set of candidate URLs for an
Ollama server by querying the /api/tags endpoint.
"""

# ------------------------------------------------------------------
# 1️⃣  Discover the Ollama host
# ------------------------------------------------------------------
OLLAMA_PORT = 11434
CANDIDATES = [
    f"http://host.docker.internal:{OLLAMA_PORT}",
    f"http://localhost:{OLLAMA_PORT}",
]


def _is_ollama_up(url: str) -> bool:
    """
    Ping the Ollama server.  Ollama exposes a simple `/api/tags` endpoint
    that returns a JSON list of available models.  If the request succeeds
    (status 200) we consider the host reachable.
    """
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{url}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


def discover_ollama() -> str:
    """
    Return the first URL from CANDIDATES that responds to /api/tags.
    Raise RuntimeError if none are reachable.
    """
    for url in CANDIDATES:
        if _is_ollama_up(url):
            return url
    raise RuntimeError(
        "Could not reach an Ollama daemon on any of the following URLs:\n"
        + "\n".join(f"  - {u}" for u in CANDIDATES)
        + "\n\nMake sure the Ollama server is running and reachable."
    )
