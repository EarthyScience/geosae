#!/usr/bin/env python3
"""Generate the geosae logo: a mirror-ball Earth.

The idea: an SAE explains a dense Earth embedding with a handful of active features out of a huge
dictionary. The mark is a faceted mirror ball (an Earth tiled in latitude/longitude facets) in which
almost every facet is dull and only four catch the light, each in a feature colour (water, forest,
built-up, climate), with a sparkle. Sparse activation on a planet.

Everything is plain geometry: facets of a tilted orthographic globe, a lambert-lit palette, and the
"geosae" wordmark as glyph outlines (Liberation Sans Bold, SIL OFL 1.1) embedded below, so the SVGs
need only the standard library and are deterministic. PNG/ICO/social-preview export is optional and
uses cairosvg (or the ``rsvg-convert`` binary) plus Pillow when available.

Usage::

    python make_logo.py [OUTDIR] [--no-raster]      # default OUTDIR: docs/assets/logo
"""

from __future__ import annotations

import argparse
import math
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

# --- Palette (nicefigs house colours) -------------------------------------------------------------
ACCENTS = {"blue": "#2c9be2", "green": "#2fa35c", "orange": "#ef5a1c", "gold": "#f2ab2e"}
INK = {"light": "#222222", "dark": "#f0f0f0"}
BALL = {"light": ("#e9eef3", "#aab6c4"), "dark": ("#4b5766", "#232a34")}  # (lit, shadow) facet tones
BALL_ICON = {"light": ("#f4f7fa", "#7f8da0"), "dark": ("#66758a", "#171d26")}  # stronger contrast
BALL_FAVICON = ("#b9c5d3", "#5b6a7c")  # mid tones that read on light and dark browser tabs
TAGLINE = "Sparse autoencoders for Earth embeddings"
TAGLINE_COLOR = "#5b6a7c"

# --- Geometry -------------------------------------------------------------------------------------
SEED = 7
LIGHT_DIR = (-0.5, 0.62, 0.6)
# (latitude, longitude, accent) of the four lit facets; the nearest facet to each is lit.
LIT_FACETS = [(20, 10, "blue"), (0, 50, "green"), (-20, -10, "orange"), (40, 50, "gold")]

# --- Wordmark glyphs: Liberation Sans Bold outlines, 2048 units per em, y up ------------------------
GLYPHS = {
    "g": (
        1251,
        "M596 -434Q398 -434 277.5 -358.5Q157 -283 129 -143L410 -110Q425 -175 474.5 -212Q524 -249 604 -249"
        "Q721 -249 775 -177Q829 -105 829 37V94L831 201H829Q736 2 481 2Q292 2 188 144Q84 286 84 550Q84 815 191 959"
        "Q298 1103 502 1103Q738 1103 829 908H834Q834 943 838.5 1003Q843 1063 848 1082H1114Q1108 974 1108 832V33"
        "Q1108 -198 977 -316Q846 -434 596 -434ZM831 556Q831 723 771.5 816.5Q712 910 602 910Q377 910 377 550"
        "Q377 197 600 197Q712 197 771.5 290.5Q831 384 831 556Z",
    ),
    "e": (
        1139,
        "M586 -20Q342 -20 211 124.5Q80 269 80 546Q80 814 213 958Q346 1102 590 1102Q823 1102 946 947.5"
        "Q1069 793 1069 495V487H375Q375 329 433.5 248.5Q492 168 600 168Q749 168 788 297L1053 274Q938 -20 586 -20Z"
        "M586 925Q487 925 433.5 856Q380 787 377 663H797Q789 794 734 859.5Q679 925 586 925Z",
    ),
    "o": (
        1251,
        "M1171 542Q1171 279 1025 129.5Q879 -20 621 -20Q368 -20 224 130Q80 280 80 542Q80 803 224 952.5"
        "Q368 1102 627 1102Q892 1102 1031.5 957.5Q1171 813 1171 542ZM877 542Q877 735 814 822Q751 909 631 909"
        "Q375 909 375 542Q375 361 437.5 266.5Q500 172 618 172Q877 172 877 542Z",
    ),
    "s": (
        1139,
        "M1055 316Q1055 159 926.5 69.5Q798 -20 571 -20Q348 -20 229.5 50.5Q111 121 72 270L319 307"
        "Q340 230 391.5 198Q443 166 571 166Q689 166 743 196Q797 226 797 290Q797 342 753.5 372.5Q710 403 606 424"
        "Q368 471 285 511.5Q202 552 158.5 616.5Q115 681 115 775Q115 930 234.5 1016.5Q354 1103 573 1103"
        "Q766 1103 883.5 1028Q1001 953 1030 811L781 785Q769 851 722 883.5Q675 916 573 916Q473 916 423 890.5"
        "Q373 865 373 805Q373 758 411.5 730.5Q450 703 541 685Q668 659 766.5 631.5Q865 604 924.5 566"
        "Q984 528 1019.5 468.5Q1055 409 1055 316Z",
    ),
    "a": (
        1139,
        "M393 -20Q236 -20 148 65.5Q60 151 60 306Q60 474 169.5 562Q279 650 487 652L720 656V711"
        "Q720 817 683 868.5Q646 920 562 920Q484 920 447.5 884.5Q411 849 402 767L109 781Q136 939 253.5 1020.5"
        "Q371 1102 574 1102Q779 1102 890 1001Q1001 900 1001 714V320Q1001 229 1021.5 194.5Q1042 160 1090 160"
        "Q1122 160 1152 166V14Q1127 8 1107 3Q1087 -2 1067 -5Q1047 -8 1024.5 -10Q1002 -12 972 -12"
        "Q866 -12 815.5 40Q765 92 755 193H749Q631 -20 393 -20ZM720 501 576 499Q478 495 437 477.5Q396 460 374.5 424"
        "Q353 388 353 328Q353 251 388.5 213.5Q424 176 483 176Q549 176 603.5 212Q658 248 689 311.5Q720 375 720 446Z",
    ),
}
UNITS_PER_EM = 2048


def num(x: float) -> str:
    """Format a coordinate compactly (2 decimals, no trailing zeros)."""
    return f"{x:.2f}".rstrip("0").rstrip(".")


def mix(a: str, b: str, t: float) -> str:
    """Linear blend of two ``#rrggbb`` colours (t=0 gives ``a``, t=1 gives ``b``)."""
    ca, cb = (tuple(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in (a, b))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb, strict=True))


def wordmark(text: str, x: float, baseline: float, size: float) -> tuple[str, float]:
    """Return SVG path data for ``text`` and its width, scaled to ``size`` px with its baseline at ``baseline``."""
    scale, pen, out = size / UNITS_PER_EM, 0.0, []
    for ch in text:
        advance, outline = GLYPHS[ch]
        tokens = re.findall(r"[MLQHVZ]|-?[\d.]+", outline)
        i, cmd, cx, cy = 0, "", 0.0, 0.0
        while i < len(tokens):
            if tokens[i].isalpha():
                cmd = tokens[i]
                i += 1
                if cmd == "Z":
                    out.append("Z")
                    continue
            if cmd == "H":
                cx = float(tokens[i])
                i += 1
                out.append(f"H{num(x + (pen + cx) * scale)}")
            elif cmd == "V":
                cy = float(tokens[i])
                i += 1
                out.append(f"V{num(baseline - cy * scale)}")
            else:
                n = 2 if cmd in "ML" else 4
                pts = [float(t) for t in tokens[i : i + n]]
                i += n
                cx, cy = pts[-2], pts[-1]
                coords = " ".join(
                    f"{num(x + (pen + px) * scale)} {num(baseline - py * scale)}"
                    for px, py in zip(pts[::2], pts[1::2], strict=True)
                )
                out.append(f"{cmd}{coords}")
                if cmd == "M":
                    cmd = "L"  # implicit lineto after a moveto
        pen += advance
    return "".join(out), pen * scale


def sparkle(cx: float, cy: float, r: float, fill: str = "#ffffff") -> str:
    """A four-point star centred at (cx, cy) with radius ``r``."""
    k = r * 0.16
    pts = [(0, -r), (k, -k), (r, 0), (k, k), (0, r), (-k, k), (-r, 0), (-k, -k)]
    return f'<path d="M{" L".join(f"{num(cx + px)} {num(cy + py)}" for px, py in pts)} Z" fill="{fill}"/>'


def project(lat: float, lon: float, tilt: float, rot: float, radius: float, centre: float = 32.0):
    """Orthographic projection of (lat, lon) degrees; returns (x, y, depth) with depth > 0 facing the viewer."""
    la, lo, t = math.radians(lat), math.radians(lon + rot), math.radians(tilt)
    x, y, z = math.cos(la) * math.sin(lo), math.sin(la), math.cos(la) * math.cos(lo)
    y2, z2 = y * math.cos(t) - z * math.sin(t), y * math.sin(t) + z * math.cos(t)
    return centre + radius * x, centre - radius * y2, z2


def facets(step: int, radius: float, tilt: float, rot: float) -> list[dict]:
    """Visible lat/lon facets of the ball: corners, centre depth and lambert brightness ``v`` in [0, 1]."""
    norm = math.sqrt(sum(c * c for c in LIGHT_DIR))
    light = tuple(c / norm for c in LIGHT_DIR)
    out = []
    for lat in [-80 + i * step for i in range(int(160 / step) + 1)]:
        for lon in range(-180, 180, step):
            lat0, lat1 = max(lat - step / 2, -85), min(lat + step / 2, 85)
            mid_lat, mid_lon = (lat0 + lat1) / 2, lon + step / 2
            if project(mid_lat, mid_lon, tilt, rot, radius)[2] < 0.08:
                continue
            corners = [
                project(a, b, tilt, rot, radius)
                for a, b in ((lat0, lon), (lat0, lon + step), (lat1, lon + step), (lat1, lon))
            ]
            la, lo, t = math.radians(mid_lat), math.radians(mid_lon + rot), math.radians(tilt)
            nx, ny, nz = math.cos(la) * math.sin(lo), math.sin(la), math.cos(la) * math.cos(lo)
            ny2, nz2 = ny * math.cos(t) - nz * math.sin(t), ny * math.sin(t) + nz * math.cos(t)
            v = max(0.0, nx * light[0] + ny2 * light[1] + nz2 * light[2])
            out.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "corners": corners,
                    "depth": project(mid_lat, mid_lon, tilt, rot, radius)[2],
                    "v": v,
                }
            )
    return out


def polygon(corners, shrink: float, fill: str, extra: str = "") -> str:
    """SVG polygon of a facet, shrunk towards its centre by ``shrink`` to open the seams."""
    mx = sum(c[0] for c in corners) / 4
    my = sum(c[1] for c in corners) / 4
    pts = " ".join(f"{num(mx + (c[0] - mx) * shrink)},{num(my + (c[1] - my) * shrink)}" for c in corners)
    return f'<polygon points="{pts}" fill="{fill}"{extra}/>'


def ball(
    tones: tuple[str, str],
    prefix: str,
    *,
    step: int = 20,
    shrink: float = 0.86,
    glow: bool = True,
    lit_boost: bool = False,
) -> tuple[str, str]:
    """Build the mirror ball; returns (defs, body). ``glow`` adds blurred halos, ``lit_boost`` enlarges lit facets."""
    rng = random.Random(SEED)
    cells = facets(step, 30.0, 24.0, -20.0)
    lit = {}
    for lat, lon, name in LIT_FACETS:
        best = min(
            range(len(cells)),
            key=lambda i: (cells[i]["lat"] - lat) ** 2 + ((cells[i]["lon"] - lon + 540) % 360 - 180) ** 2,
        )
        lit[best] = name
    defs = (
        (
            f'<filter id="{prefix}glow" x="-60%" y="-60%" width="220%" height="220%">'
            '<feGaussianBlur stdDeviation="1.6"/></filter>'
        )
        if glow
        else ""
    )
    body = []
    for i in sorted(range(len(cells)), key=lambda k: cells[k]["depth"]):
        cell = cells[i]
        if i in lit:
            base = ACCENTS[lit[i]]
            colour = mix(mix(base, "#000000", 0.1), mix(base, "#ffffff", 0.3), cell["v"])
            if glow:
                body.append(polygon(cell["corners"], shrink, base, f' opacity=".55" filter="url(#{prefix}glow)"'))
            body.append(polygon(cell["corners"], shrink, colour))
        else:
            t = min(1.0, max(0.0, cell["v"] * 1.15 + rng.uniform(-0.18, 0.18)))
            body.append(polygon(cell["corners"], shrink, mix(tones[1], tones[0], t)))
    if lit_boost:
        body += [polygon(cells[i]["corners"], 1.0, ACCENTS[name]) for i, name in lit.items()]
    return defs, "".join(body)


def mark_parts(theme: str) -> tuple[str, str]:
    """Rich mark (defs, body): glowing lit facets plus two sparkles."""
    defs, body = ball(BALL[theme], f"m{theme[0]}")
    return defs, body + sparkle(36, 24, 6.5) + sparkle(22, 38, 3.2)


def icon_parts(tones: tuple[str, str], prefix: str, *, spark: bool) -> str:
    """Icon body: same ball with thicker seams, stronger contrast, enlarged lit facets, no glow."""
    _, body = ball(tones, prefix, shrink=0.80, glow=False, lit_boost=True)
    return body + (sparkle(36, 24, 7.5) if spark else "")


def svg(width: float, height: float, defs: str, body: str, scale: float) -> str:
    """Wrap ``body`` in an SVG document with the given viewBox and a nominal pixel size of ``scale`` x viewBox."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {num(width)} {num(height)}" '
        f'width="{round(width * scale)}" height="{round(height * scale)}" role="img"><title>geosae</title>'
        f"<defs>{defs}</defs>{body}</svg>\n"
    )


def build_svgs(theme: str) -> dict[str, str]:
    """All SVG variants for one theme ("light" for white backgrounds, "dark" for #0d1117)."""
    defs, body = mark_parts(theme)
    path, width = wordmark("geosae", 80, 43.5, 46)
    text = f'<path d="{path}" fill="{INK[theme]}"/>'
    stacked_path, stacked_w = wordmark("geosae", 0, 128, 44)
    full_w = round(stacked_w + 24)
    stacked_path, _ = wordmark("geosae", (full_w - stacked_w) / 2, 128, 44)
    stacked = (
        f'<g transform="translate({num((full_w - 96) / 2)} 0) scale(1.5)">{body}</g>'
        f'<path d="{stacked_path}" fill="{INK[theme]}"/>'
    )
    return {
        f"geosae-mark-{theme}.svg": svg(64, 64, defs, body, 8),
        f"geosae-icon-{theme}.svg": svg(64, 64, "", icon_parts(BALL_ICON[theme], f"i{theme[0]}", spark=True), 8),
        f"geosae-icon-small-{theme}.svg": svg(64, 64, "", icon_parts(BALL_ICON[theme], f"s{theme[0]}", spark=False), 8),
        f"geosae-lockup-{theme}.svg": svg(round(80 + width + 4), 64, defs, body + text, 4),
        f"geosae-stacked-{theme}.svg": svg(full_w, 148, defs, stacked, 4),
    }


def rasterise(svg_file: Path, png_file: Path, *, width: int | None = None, height: int | None = None) -> None:
    """Render an SVG to PNG with cairosvg, falling back to the ``rsvg-convert`` binary."""
    try:
        import cairosvg

        cairosvg.svg2png(url=str(svg_file), write_to=str(png_file), output_width=width, output_height=height)
    except ImportError:
        args = ["rsvg-convert"] + (["-w", str(width)] if width else []) + (["-h", str(height)] if height else [])
        subprocess.run([*args, str(svg_file), "-o", str(png_file)], check=True)


def have_rasteriser() -> bool:
    """True if cairosvg, or rsvg-convert, and Pillow are all available."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    try:
        import cairosvg  # noqa: F401

        return True
    except ImportError:
        return shutil.which("rsvg-convert") is not None


def find_font(size: int):
    """A Pillow font for the social-preview tagline (any common sans, else Pillow's default)."""
    from PIL import ImageFont

    for name in ("LiberationSans-Regular.ttf", "DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def export_rasters(out: Path) -> None:
    """PNG sizes, favicon.ico, apple-touch-icon and the 1280x640 social preview (needs Pillow + a rasteriser)."""
    from PIL import Image, ImageDraw

    png = out / "png"
    png.mkdir(exist_ok=True)
    for theme in INK:
        for s in (16, 32):
            rasterise(out / f"geosae-icon-small-{theme}.svg", png / f"geosae-icon-{s}-{theme}.png", width=s, height=s)
        for s in (48, 64):
            rasterise(out / f"geosae-icon-{theme}.svg", png / f"geosae-icon-{s}-{theme}.png", width=s, height=s)
        for s in (128, 256, 512, 1024):
            rasterise(out / f"geosae-mark-{theme}.svg", png / f"geosae-mark-{s}-{theme}.png", width=s, height=s)
        rasterise(out / f"geosae-lockup-{theme}.svg", png / f"geosae-lockup-1024-{theme}.png", width=1024)
        rasterise(out / f"geosae-stacked-{theme}.svg", png / f"geosae-stacked-1024-{theme}.png", height=1024)
    for s in (16, 32, 48, 64):
        rasterise(out / "favicon.svg", png / f"favicon-{s}.png", width=s, height=s)
    icons = [Image.open(png / f"favicon-{s}.png").convert("RGBA") for s in (16, 32, 48)]
    icons[-1].save(out / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)], append_images=icons[:-1])
    tmp = png / "_tmp.png"
    rasterise(out / "geosae-mark-dark.svg", tmp, width=144, height=144)
    tile = Image.new("RGBA", (180, 180), "#0d1117")
    tile.alpha_composite(Image.open(tmp).convert("RGBA"), (18, 18))
    tile.convert("RGB").save(out / "apple-touch-icon.png")
    rasterise(out / "geosae-lockup-light.svg", tmp, width=900)
    lockup = Image.open(tmp).convert("RGBA")
    tmp.unlink()
    card = Image.new("RGB", (1280, 640), "white")
    card.paste(lockup, ((1280 - lockup.width) // 2, 200), lockup)
    draw, font = ImageDraw.Draw(card), find_font(44)
    draw.text(
        ((1280 - draw.textlength(TAGLINE, font=font)) / 2, 200 + lockup.height + 50),
        TAGLINE,
        fill=TAGLINE_COLOR,
        font=font,
    )
    card.save(out / "social-preview-1280x640.png")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate the geosae mirror-ball logo assets.")
    parser.add_argument("outdir", nargs="?", default="docs/assets/logo", type=Path, help="output directory")
    parser.add_argument("--no-raster", action="store_true", help="write SVGs only (skip PNG/ICO/social preview)")
    args = parser.parse_args(argv)
    out: Path = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for theme in INK:
        files.update(build_svgs(theme))
    files["favicon.svg"] = svg(64, 64, "", icon_parts(BALL_FAVICON, "fu", spark=False), 8)
    for name, text in files.items():
        (out / name).write_text(text)
    print(f"wrote {len(files)} SVGs to {out}")
    if args.no_raster:
        return 0
    if not have_rasteriser():
        print("skipping PNG/ICO export: install Pillow and cairosvg (or rsvg-convert)", file=sys.stderr)
        return 0
    export_rasters(out)
    print(f"wrote PNGs, favicon.ico, apple-touch-icon.png and social preview to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
