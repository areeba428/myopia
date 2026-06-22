"""
Vercel serverless entrypoint.

Vercel's Python runtime serves any module under /api that exposes an ASGI `app`.
We add the project root to sys.path and re-export the FastAPI app from app/api.py,
which serves both the REST endpoints and the web frontend.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.api import app  # noqa: E402  (ASGI app Vercel will serve)
