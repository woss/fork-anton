"""Publish HTML reports to the anton-services web host."""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

from anton.minds_http import minds_request


DEFAULT_PUBLISH_URL = "https://4eqbi9b2pg.execute-api.us-east-1.amazonaws.com/Prod"


def _zip_html(path: Path) -> bytes:
    """Create a ZIP archive from a single HTML file or a directory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if path.is_file():
            zf.write(path, "index.html")
        else:
            # Directory — include all files, ensure index.html exists
            for f in sorted(path.rglob("*")):
                if f.is_file():
                    zf.write(f, str(f.relative_to(path)))
    return buf.getvalue()


def publish(
    file_path: Path,
    *,
    api_key: str,
    publish_url: str = DEFAULT_PUBLISH_URL,
    ssl_verify: bool = True,
) -> dict:
    """Zip and upload an HTML file/directory. Returns the upload response dict.

    Response keys: user_prefix, md5, view_url, files
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Path not found: {file_path}")

    zipped = _zip_html(file_path)
    payload = json.dumps({"file_payload": base64.b64encode(zipped).decode()}).encode()

    url = f"{publish_url.rstrip('/')}/upload"
    raw = minds_request(url, api_key, method="POST", payload=payload, verify=ssl_verify)
    return json.loads(raw)
