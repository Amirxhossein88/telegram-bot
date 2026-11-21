"""Generate a new cover using Cloudflare Workers AI."""
import base64
import json
import os
import re
import logging

from aiohttp import ClientSession, ClientTimeout, FormData
from cover_design import compose_cover

LOG = logging.getLogger("nexify")


def cover_prompt(prompt):
    return (
        "You are a professional graphic designer creating an exclusive Nexify social media cover. "
        "Brand palette: deep black #050505, neon purple #7B2CFF and light purple #A855F7. "
        "Premium futuristic tech and restrained cyberpunk: purposeful 3D elements, geometric lines, "
        "subtle textures and cinematic purple lighting. Make the specific video topic instantly clear "
        "with one relevant visual or character in the lower half. Reserve the dark upper half for "
        "the exact original Nexify logo and a clear Persian headline, which will be added afterwards. "
        "Vertical 9:16, uncluttered, professional, no generated text, logos, watermarks or unrelated objects. "
        "Video-specific visual concept: " + prompt.strip()[:1200]
    )


class CoverConfigurationError(Exception):
    pass


async def generate_cover(folder, prompt, headline, subtitle="", gap_client=None):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Cover prompt is empty")
    if gap_client is not None:
        try:
            image = await gap_client.images.generate(
                model=os.getenv("GAPGPT_IMAGE_MODEL", "gpt-image-2"),
                prompt=cover_prompt(prompt), size="1024x1536", n=1,
            )
            encoded = image.data[0].b64_json
            if not encoded:
                raise ValueError("GapGPT did not return an inline image")
            raw = base64.b64decode(encoded, validate=True)
            source = folder / "gap-background.png"
            source.write_bytes(raw)
        except Exception as exc:
            LOG.warning("GapGPT image request or decoding failed (%s); trying Cloudflare", type(exc).__name__)
        else:
            # Composition is independent of the provider; do not request another image if it fails.
            return compose_cover(source, folder / "generated-cover.jpg", headline, subtitle)
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if not re.fullmatch(r"[a-fA-F0-9]{32}", account) or not token:
        raise CoverConfigurationError("Cloudflare Workers AI credentials are not configured")
    form = FormData()
    form.add_field("prompt", ("Premium vertical futuristic tech cover. Deep black and neon purple, "
                              "one clear topic-related visual in lower half, dark empty upper half. "
                              "Uncluttered, no text or logos. Subject: " + prompt.strip()[:300]),
                   content_type="text/plain")
    form.add_field("width", "768")
    form.add_field("height", "1344")
    endpoint = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/black-forest-labs/flux-2-klein-4b"
    async with ClientSession(timeout=ClientTimeout(total=150)) as session:
        async with session.post(endpoint, headers={"Authorization": "Bearer " + token}, data=form) as response:
            if response.status != 200:
                try:
                    detail = await response.json(content_type=None)
                    codes = [item.get("code") for item in detail.get("errors", [])
                             if isinstance(item, dict) and isinstance(item.get("code"), int)]
                except (ValueError, TypeError, AttributeError):
                    codes = []
                LOG.error("Cloudflare cover failed: HTTP %s, error codes %s", response.status, codes[:5])
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
    source = folder / ("generated-background" + suffix)
    source.write_bytes(raw)
    cover = folder / "generated-cover.jpg"
    return compose_cover(source, cover, headline, subtitle)
