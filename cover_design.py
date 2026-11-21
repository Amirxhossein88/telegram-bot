"""Compose a branded Persian cover using a generated image and the exact supplied logo."""
from pathlib import Path
import re

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
FONT = ROOT / "Vazirmatn-Bold.ttf"
LOGO = ROOT / "nexify-profile.jpg"
SIZE = (1080, 1920)


def _visual(text):
    return get_display(arabic_reshaper.reshape(text))


def _lines(draw, text, font, width, max_lines):
    words = re.sub(r"\s+", " ", str(text)).strip().split()[:25]
    lines = []
    line = ""
    for word in words:
        candidate = (line + " " + word).strip()
        if draw.textlength(_visual(candidate), font=font) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    truncated = len(lines) > max_lines
    lines = lines[:max_lines]
    for i, line in enumerate(lines):
        ellipsis = truncated and i == len(lines) - 1
        if draw.textlength(_visual(line), font=font) > width:
            ellipsis = True
        if ellipsis:
            line = line.rstrip(' .،')
            while line and draw.textlength(_visual(line + "…"), font=font) > width:
                line = line[:-1].rstrip()
            lines[i] = line + "…"
    return lines


def _draw_heading(draw, text, top, limit, max_lines, largest, smallest, fill):
    for size in range(largest, smallest - 1, -4):
        font = ImageFont.truetype(str(FONT), size, layout_engine=ImageFont.Layout.BASIC)
        lines = _lines(draw, text, font, 900, max_lines)
        if lines and len(lines) * (size + 28) <= limit and all(draw.textlength(_visual(line), font=font) <= 900 for line in lines):
            break
    else:
        raise ValueError("Cover title is too long to render legibly")
    for line in lines:
        shaped = _visual(line)
        bounds = draw.textbbox((0, 0), shaped, font=font, stroke_width=1)
        width = bounds[2] - bounds[0]
        draw.text(((1080 - width) // 2 - bounds[0], top - bounds[1]), shaped, font=font,
                  fill=fill, stroke_width=1, stroke_fill="#050505")
        top += size + 28
    return top


def compose_cover(source, output, headline, subtitle=""):
    headline = re.sub(r"\s+", " ", str(headline)).strip()
    if not headline:
        raise ValueError("Cover headline is missing")
    image = ImageOps.fit(Image.open(source).convert("RGB"), SIZE, method=Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for y in range(1120):
        alpha = 248 if y < 600 else int(248 * (1120 - y) / 520)
        draw.line((0, y, 1080, y), fill=(5, 5, 5, alpha))
    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(image)
    logo = Image.open(LOGO).convert("RGB").crop((20, 52, 134, 98))
    logo = logo.resize((342, 138), Image.Resampling.LANCZOS)
    image.paste(logo, ((1080 - logo.width) // 2, 100))
    draw.line((415, 278, 665, 278), fill="#7B2CFF", width=5)
    bottom = _draw_heading(draw, headline, 330, 280, 2, 104, 52, "#FFFFFF")
    subtitle = re.sub(r"\s+", " ", str(subtitle)).strip()
    if subtitle:
        _draw_heading(draw, subtitle, bottom + 34, 120, 1, 48, 32, "#D8B4FE")
    draw.line((80, 1805, 310, 1805), fill="#7B2CFF", width=5)
    draw.line((770, 1805, 1000, 1805), fill="#A855F7", width=5)
    image.convert("RGB").save(output, "JPEG", quality=92, optimize=True)
    return output
