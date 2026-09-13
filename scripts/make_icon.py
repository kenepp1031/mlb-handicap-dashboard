"""Generates the app's logo assets, all styled as glowing neon-tube signs
on a fully transparent background (no black box behind them):

  - assets/neon_baseball.png -- a cyan neon ring with two red neon seam
    curves, used on the left of the app's top banner.
  - assets/neon_badge.png / assets/app_icon.ico -- an original rounded-badge
    neon sign (in the spirit of bar/man-cave baseball signs, NOT a copy of
    MLB's trademarked batter logo) with a small baseball glyph inside, used
    on the right of the banner and as the desktop shortcut icon.

Run from the project root: python scripts/make_icon.py
"""
from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFilter

SIZE = 512
CYAN_NEON = (60, 230, 255, 255)
CYAN_CORE = (230, 253, 255, 255)
RED_NEON = (255, 45, 60, 255)
RED_CORE = (255, 200, 205, 255)

ASSETS_DIR = Path(__file__).parent.parent / "assets"


# --------------------------------------------------------------- neon tube
def neon_stroke(size, draw_fn, glow_color, core_color, core_width, glow_width, glow_blur):
    """A neon-tube stroke: soft wide halo + solid bright core, tuned to stay
    legible even after downscaling to small icon sizes (16-32px)."""
    result = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    halo = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo)
    draw_fn(hd, glow_color, glow_width)
    halo = halo.filter(ImageFilter.GaussianBlur(glow_blur))
    result.alpha_composite(halo)

    mid = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mdw = ImageDraw.Draw(mid)
    draw_fn(mdw, glow_color, int(core_width * 1.5))
    result.alpha_composite(mid)

    core = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)
    draw_fn(cd, core_color, core_width)
    result.alpha_composite(core)

    return result


def bezier_points(p0, p1, ctrl, n=48):
    pts = []
    for i in range(n):
        t = i / (n - 1)
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * ctrl[0] + t ** 2 * p1[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * ctrl[1] + t ** 2 * p1[1]
        pts.append((x, y))
    return pts


def seam_points(cx, cy, r, angle_start_deg, angle_end_deg, bulge_factor):
    """Two points on the ball's outline joined by a curve that bows outward
    (away from center) rather than crossing through the middle -- matches a
    real baseball seam."""
    a0 = math.radians(angle_start_deg)
    a1 = math.radians(angle_end_deg)
    p0 = (cx + r * math.cos(a0), cy + r * math.sin(a0))
    p1 = (cx + r * math.cos(a1), cy + r * math.sin(a1))
    mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
    ctrl = (cx + (mid[0] - cx) * bulge_factor, cy + (mid[1] - cy) * bulge_factor)
    return bezier_points(p0, p1, ctrl)


def draw_neon_seam(img, size, pts, core_width=13, glow_width=24, glow_blur=15):
    def draw_path(d, col, w):
        d.line(pts, fill=col, width=w, joint="curve")
        rr = w / 2
        for p in (pts[0], pts[-1]):
            d.ellipse([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr], fill=col)

    img.alpha_composite(
        neon_stroke(size, draw_path, RED_NEON, RED_CORE, core_width=core_width, glow_width=glow_width, glow_blur=glow_blur)
    )


def neon_ring_with_seams(size, cx, cy, r, ring_core=16, ring_glow=28, ring_blur=18, seam_core=13, seam_glow=24, seam_blur=15):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    def draw_ring(d, col, w):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=w)

    img.alpha_composite(
        neon_stroke(size, draw_ring, CYAN_NEON, CYAN_CORE, core_width=ring_core, glow_width=ring_glow, glow_blur=ring_blur)
    )

    seam_a = seam_points(cx, cy, r, angle_start_deg=-65, angle_end_deg=172, bulge_factor=1.5)
    seam_b = [(2 * cx - x, 2 * cy - y) for x, y in seam_a]
    draw_neon_seam(img, size, seam_a, seam_core, seam_glow, seam_blur)
    draw_neon_seam(img, size, seam_b, seam_core, seam_glow, seam_blur)

    return img


def neon_baseball(size=SIZE):
    """Standalone neon baseball sign for the banner."""
    cx, cy = size / 2, size / 2
    r = size * 0.42
    return neon_ring_with_seams(size, cx, cy, r)


def neon_badge(size=SIZE):
    """An original rounded-badge neon sign (bar/man-cave sign style) with a
    small baseball glyph inside -- NOT a reproduction of MLB's trademarked
    batter-silhouette logo."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    pad = size * 0.09
    box = [pad, pad, size - pad, size - pad]
    radius = size * 0.22

    def draw_outline(d, col, w):
        d.rounded_rectangle(box, radius=radius, outline=col, width=w)

    img.alpha_composite(
        neon_stroke(size, draw_outline, CYAN_NEON, CYAN_CORE, core_width=14, glow_width=24, glow_blur=16)
    )

    # small red neon accent tab on the top-right corner of the badge
    accent_len = size * 0.16

    def draw_accent(d, col, w):
        d.line([(size - pad - accent_len, pad), (size - pad, pad)], fill=col, width=w)
        d.line([(size - pad, pad), (size - pad, pad + accent_len)], fill=col, width=w)

    img.alpha_composite(
        neon_stroke(size, draw_accent, RED_NEON, RED_CORE, core_width=10, glow_width=18, glow_blur=12)
    )

    ball = neon_ring_with_seams(
        size, size / 2, size / 2 + size * 0.02, size * 0.24,
        ring_core=11, ring_glow=20, ring_blur=13,
        seam_core=8, seam_glow=15, seam_blur=9,
    )
    img.alpha_composite(ball)

    return img


def save_ico(img, path):
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    imgs = [img.resize(s, Image.LANCZOS) for s in sizes]
    imgs[0].save(path, format="ICO", sizes=sizes)


if __name__ == "__main__":
    ASSETS_DIR.mkdir(exist_ok=True)

    neon_baseball().save(ASSETS_DIR / "neon_baseball.png")
    badge = neon_badge()
    badge.save(ASSETS_DIR / "neon_badge.png")
    save_ico(badge, ASSETS_DIR / "app_icon.ico")

    print("Saved neon_baseball.png, neon_badge.png, app_icon.ico")
