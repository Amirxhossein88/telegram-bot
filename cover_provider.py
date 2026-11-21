"""Generate a new cover using Cloudflare Workers AI."""
import base64
import json
import os
import re

from aiohttp import ClientSession, ClientTimeout, FormData


class CoverConfigurationError(Exception):
    pass


async def generate_cover(folder, prompt):
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if not re.fullmatch(r"[a-fA-F0-9]{32}", account) or not token:
        raise CoverConfigurationError("Cloudflare Workers AI credentials are not configured")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Cover prompt is empty")
    form = FormData()
    form.add_field("prompt", prompt.strip()[:1600] + ", vertical composition for an Instagram Reel cover, high quality, no text, no logos", content_type="text/plain")
    form.add_field("width", "768")
    form.add_field("height", "1344")
    endpoint = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/black-forest-labs/flux-2-klein-4b"
    async with ClientSession(timeout=ClientTimeout(total=150)) as session:
        async with session.post(endpoint, headers={"Authorization": "Bearer " + token}, data=form) as response:
            if response.status != 200:
                raise RuntimeError(f"Cover generation failed (HTTP {response.status})")
            chunks = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                chunks.extend(chunk)
                if len(chunks) > 16_000_000:
                    raise ValueError("Cover response is too large")
            result = json.loads(chunks)
    if not result.get("success") or not isinstance(result.get("result"), dict):
        raise RuntimeError("Cover generation returned an error")
    encoded = result["result"].get("image")
    if not isinstance(encoded, str):
        raise ValueError("Cover image was missing")
    raw = base64.b64decode(encoded, validate=True)
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        suffix = ".png"
    elif raw.startswith(b"\xff\xd8\xff"):
        suffix = ".jpg"
    else:
        raise ValueError("Cover image has an unsupported format")
    cover = folder / ("generated-cover" + suffix)
    cover.write_bytes(raw)
    return cover
