from __future__ import annotations

import base64
import io
import json
import math
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE_MAP = ROOT / "base_map.png"
LATEST_PNG = ROOT / "latest.png"
LATEST_ARROW_PNG = ROOT / "latest_arrow.png"
INFO_JSON = ROOT / "latest_info.json"

WIDTH = 600
HEIGHT = 480
MAP_BOTTOM = 350

BASE_URL = "https://www.shmu.sk/data/data002/radar-cappi_z_2_600x480-{stamp}-mosaic--.png"
FRAME_STEP_MIN = 5
LOOKBACK_HOURS = 4
FRAME_COUNT = 7                 # ~30 min histórie
REQUEST_TIMEOUT = 20
USER_AGENT = "shmu-radar-eink/fixed-basemap-overlay-v1"

WHITE = np.array([255, 255, 255], dtype=np.uint8)
BLACK = np.array([0, 0, 0], dtype=np.uint8)
RED = np.array([255, 0, 0], dtype=np.uint8)

# Radar color extraction.
# Static topography/grayscale is ignored automatically because saturation is low.
COLOR_SAT_MIN = 0.22
COLOR_VALUE_MIN = 0.16
COLOR_SPREAD_MIN = 32

# A colored pixel is considered a static SHMU graphic only if it is present
# essentially all the time AND its RGB value is almost unchanged.
STATIC_PRESENCE_MIN = 0.98
STATIC_RGB_STD_MAX = 2.5

# Keep only meaningful rain pixels.
MIN_COLOR_NEIGHBOURS = 2

# Precipitation filtering. Tiny isolated echoes are intentionally suppressed
# because this e-ink map is meant to show meaningful precipitation areas,
# not every weak radar speckle.
MIN_DISPLAY_COMPONENT_PIXELS = 60
TEMPORAL_WINDOW_FRAMES = 3
TEMPORAL_MIN_HITS = 2
MAX_COMPONENT_SHIFT_PER_FRAME = 35

# Long-lived stationary clutter. Small or sparse echoes that remain almost
# fixed for most of the ~30 minute history are treated as radar artifacts.
STATIONARY_MIN_HITS = 5
STATIONARY_MATCH_RADIUS = 10.0
STATIONARY_MAX_CENTROID_SPREAD = 6.0
STATIONARY_SMALL_COMPONENT_MAX_PIXELS = 450
STATIONARY_SPARSE_FILL_RATIO = 0.24

# Motion calculation. The arrow is deliberately conservative: it is shown
# only when several independent phase-correlation measurements agree.
MIN_PRECIP_PIXELS_FOR_MOTION = 1000
MIN_PHASE_CONFIDENCE = 1.25
MAX_SHIFT_PER_FRAME = 35
MIN_MOTION_MAG = 0.8
PHASE_DOWNSCALE = 2
MIN_VALID_PHASE_SHIFTS = 3
MAX_DIRECTION_DEVIATION_DEG = 35.0

# This 1-bit mask is derived from the SHMU 600x480 map itself.
# It fixes the Slovakia contour to the exact SHMU coordinate frame.
_COUNTRY_MASK_ZLIB_B64 = """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"""


def load_country_mask() -> np.ndarray:
    packed = zlib.decompress(base64.b64decode(_COUNTRY_MASK_ZLIB_B64))
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big")
    mask = bits[:WIDTH * HEIGHT].reshape((HEIGHT, WIDTH)).astype(bool)

    # A small margin keeps rain just touching/approaching the state border visible.
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.filter(ImageFilter.MaxFilter(17))  # roughly +8 px
    expanded = np.array(img, dtype=np.uint8) > 0
    expanded[MAP_BOTTOM:, :] = False
    return expanded


def utc_now_rounded() -> datetime:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    return now.replace(minute=(now.minute // FRAME_STEP_MIN) * FRAME_STEP_MIN)


def make_url(dt: datetime) -> str:
    return BASE_URL.format(stamp=dt.strftime("%Y%m%d-%H%M"))


def fetch_image(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
        if r.status_code != 200:
            return None
        data = r.content
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            return None
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if img.size != (WIDTH, HEIGHT):
            return None
        return img
    except Exception:
        return None


def get_latest_frame() -> Tuple[datetime, Image.Image, str]:
    probe = utc_now_rounded()
    attempts = int((LOOKBACK_HOURS * 60) / FRAME_STEP_MIN)

    for i in range(attempts):
        dt = probe - timedelta(minutes=i * FRAME_STEP_MIN)
        url = make_url(dt)
        img = fetch_image(url)
        if img is not None:
            return dt, img, url

    raise RuntimeError("Nepodarilo sa nájsť aktuálnu radarovú snímku SHMÚ.")


def get_history(latest_dt: datetime) -> List[Tuple[datetime, Image.Image]]:
    frames: List[Tuple[datetime, Image.Image]] = []
    for i in range(FRAME_COUNT):
        dt = latest_dt - timedelta(minutes=i * FRAME_STEP_MIN)
        img = fetch_image(make_url(dt))
        if img is not None:
            frames.append((dt, img))
    frames.reverse()
    return frames


def rgb_to_sv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = rgb.astype(np.float32) / 255.0
    maxc = a.max(axis=2)
    minc = a.min(axis=2)
    delta = maxc - minc
    sat = np.where(maxc > 1e-6, delta / np.maximum(maxc, 1e-6), 0.0)
    val = maxc
    return sat, val


def colored_candidate(rgb: np.ndarray) -> np.ndarray:
    sat, val = rgb_to_sv(rgb)
    a = rgb.astype(np.int16)
    spread = a.max(axis=2) - a.min(axis=2)
    return (
        (sat >= COLOR_SAT_MIN)
        & (val >= COLOR_VALUE_MIN)
        & (spread >= COLOR_SPREAD_MIN)
    )


def build_static_color_mask(frame_rgbs: List[np.ndarray]) -> np.ndarray:
    """
    Removes only genuinely static colored SHMU graphics.
    A stationary rain cell should normally still vary in intensity/RGB over 30 min,
    while a map line stays pixel-identical.
    """
    stack = np.stack(frame_rgbs, axis=0).astype(np.float32)
    candidates = np.stack(
        [colored_candidate(x.astype(np.uint8)) for x in stack],
        axis=0,
    )

    presence = candidates.mean(axis=0)
    rgb_std = stack.std(axis=0).mean(axis=2)

    return (
        (presence >= STATIC_PRESENCE_MIN)
        & (rgb_std <= STATIC_RGB_STD_MAX)
    )


def cleanup_mask(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask.astype(np.uint8), 1, mode="constant")
    count = np.zeros(mask.shape, dtype=np.uint8)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            count += p[
                1 + dy:1 + dy + mask.shape[0],
                1 + dx:1 + dx + mask.shape[1],
            ]

    return mask & (count >= MIN_COLOR_NEIGHBOURS)


def connected_components(mask: np.ndarray):
    """Return meaningful 8-connected components from one radar mask."""
    h, w = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components = []

    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue

        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        pixels = []
        sum_x = 0
        sum_y = 0
        min_x = max_x = start_x
        min_y = max_y = start_y

        while stack:
            y, x = stack.pop()
            pixels.append(y * w + x)
            sum_x += x
            sum_y += y
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

            for ny in range(max(0, y - 1), min(h - 1, y + 1) + 1):
                for nx in range(max(0, x - 1), min(w - 1, x + 1) + 1):
                    if ny == y and nx == x:
                        continue
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

        area = len(pixels)
        if area < MIN_DISPLAY_COMPONENT_PIXELS:
            continue

        bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
        components.append(
            {
                "pixels": np.array(pixels, dtype=np.int32),
                "area": area,
                "cx": sum_x / area,
                "cy": sum_y / area,
                "bbox": (min_x, min_y, max_x, max_y),
                "fill_ratio": area / max(1, bbox_area),
            }
        )

    return components


def components_match(a, b, frame_gap: int) -> bool:
    """Match the same precipitation area across nearby 5-minute frames."""
    max_shift = MAX_COMPONENT_SHIFT_PER_FRAME * frame_gap

    ax0, ay0, ax1, ay1 = a["bbox"]
    bx0, by0, bx1, by1 = b["bbox"]
    gap_x = max(bx0 - ax1 - 1, ax0 - bx1 - 1, 0)
    gap_y = max(by0 - ay1 - 1, ay0 - by1 - 1, 0)
    if math.hypot(gap_x, gap_y) > max_shift:
        return False

    centroid_distance = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
    shape_allowance = min(
        15.0,
        0.35 * (math.sqrt(a["area"]) + math.sqrt(b["area"])),
    )
    return centroid_distance <= max_shift + shape_allowance


def stationary_component(component, components_by_frame) -> bool:
    """
    Identify small/sparse clutter that remains nearly fixed for most of
    the seven-frame (~30 min) history. Large compact rain areas are protected.
    """
    matched = []
    for components in components_by_frame:
        nearest = None
        nearest_d = None
        for candidate in components:
            d = math.hypot(
                component["cx"] - candidate["cx"],
                component["cy"] - candidate["cy"],
            )
            if d <= STATIONARY_MATCH_RADIUS and (nearest_d is None or d < nearest_d):
                nearest = candidate
                nearest_d = d
        if nearest is not None:
            matched.append((nearest["cx"], nearest["cy"]))

    if len(matched) < STATIONARY_MIN_HITS:
        return False

    max_spread = 0.0
    for i, (x1, y1) in enumerate(matched):
        for x2, y2 in matched[i + 1:]:
            max_spread = max(max_spread, math.hypot(x1 - x2, y1 - y2))

    if max_spread > STATIONARY_MAX_CENTROID_SPREAD:
        return False

    return (
        component["area"] <= STATIONARY_SMALL_COMPONENT_MAX_PIXELS
        or component["fill_ratio"] <= STATIONARY_SPARSE_FILL_RATIO
    )


def filter_precip_masks(raw_masks: List[np.ndarray]) -> List[np.ndarray]:
    """
    Conservative anti-artifact filter:
      1. remove connected areas smaller than 60 px,
      2. require confirmation in at least 2 of the latest 3 frames,
      3. reject implausible jumps,
      4. remove small/sparse long-lived stationary clutter.
    """
    if not raw_masks:
        return []

    components_by_frame = [connected_components(mask) for mask in raw_masks]
    stationary_flags = [
        [stationary_component(c, components_by_frame) for c in components]
        for components in components_by_frame
    ]

    filtered = []
    for i, components in enumerate(components_by_frame):
        out = np.zeros(raw_masks[i].shape, dtype=bool)

        for ci, current in enumerate(components):
            if stationary_flags[i][ci]:
                continue

            hits = 1
            max_gap = min(TEMPORAL_WINDOW_FRAMES - 1, i)
            for frame_gap in range(1, max_gap + 1):
                prev_i = i - frame_gap
                if any(
                    (not stationary_flags[prev_i][pj])
                    and components_match(current, previous, frame_gap)
                    for pj, previous in enumerate(components_by_frame[prev_i])
                ):
                    hits += 1

            if hits >= TEMPORAL_MIN_HITS:
                out.flat[current["pixels"]] = True

        filtered.append(out)

    return filtered


def extract_precip_masks(frame_rgbs: List[np.ndarray]) -> List[np.ndarray]:
    country = load_country_mask()
    static = build_static_color_mask(frame_rgbs)

    raw_result = []
    for rgb in frame_rgbs:
        rain = colored_candidate(rgb)
        rain &= ~static
        rain &= country
        rain = cleanup_mask(rain)
        raw_result.append(rain)

    return filter_precip_masks(raw_result)


def load_base_map() -> Image.Image:
    if not BASE_MAP.exists():
        raise RuntimeError("Chýba base_map.png v koreňovom priečinku repozitára.")

    img = Image.open(BASE_MAP).convert("RGB")
    if img.size != (WIDTH, HEIGHT):
        raise RuntimeError(
            f"base_map.png musí mať presne {WIDTH}x{HEIGHT} px, má {img.size}."
        )

    # Force base map to pure white/black only.
    a = np.array(img)
    lum = a.mean(axis=2)
    out = np.empty_like(a)
    out[:] = WHITE
    out[lum < 128] = BLACK
    return Image.fromarray(out, mode="RGB")


def compose_radar(base: Image.Image, precip: np.ndarray) -> Image.Image:
    out = np.array(base.convert("RGB"))
    # Rain is deliberately painted LAST so it remains visible over map lines/text.
    out[precip] = RED
    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_PRECIP_PIXELS_FOR_MOTION:
        return None
    return float(xs.mean()), float(ys.mean())


def downsample(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize(
        (w // PHASE_DOWNSCALE, h // PHASE_DOWNSCALE),
        Image.Resampling.BILINEAR,
    )
    img = img.filter(ImageFilter.GaussianBlur(1.0))
    return np.array(img, dtype=np.float32) / 255.0


def phase_shift(a_mask: np.ndarray, b_mask: np.ndarray):
    if (
        int(a_mask.sum()) < MIN_PRECIP_PIXELS_FOR_MOTION
        or int(b_mask.sum()) < MIN_PRECIP_PIXELS_FOR_MOTION
    ):
        return None

    a = downsample(a_mask)
    b = downsample(b_mask)
    a -= a.mean()
    b -= b.mean()

    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fb * np.conj(fa)
    cross /= np.maximum(np.abs(cross), 1e-9)

    corr = np.abs(np.fft.ifft2(cross))
    py, px = np.unravel_index(np.argmax(corr), corr.shape)

    h, w = corr.shape
    if py > h // 2:
        py -= h
    if px > w // 2:
        px -= w

    peak_y = py % h
    peak_x = px % w
    c2 = corr.copy()
    c2[
        max(0, peak_y - 2):min(h, peak_y + 3),
        max(0, peak_x - 2):min(w, peak_x + 3),
    ] = 0

    peak = float(corr[peak_y, peak_x])
    second = float(c2.max())
    confidence = peak / max(second, 1e-9)

    dx = float(px * PHASE_DOWNSCALE)
    dy = float(py * PHASE_DOWNSCALE)

    if abs(dx) > MAX_SHIFT_PER_FRAME or abs(dy) > MAX_SHIFT_PER_FRAME:
        return None
    if confidence < MIN_PHASE_CONFIDENCE:
        return None

    return dx, dy, confidence


def angle_difference_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def consistent_phase_vector(shifts):
    """
    Find a direction cluster among pairwise phase-correlation measurements.
    No centroid fallback is used: uncertain motion intentionally means no arrow.
    """
    usable = []
    for dx, dy, confidence in shifts:
        mag = math.hypot(dx, dy)
        if mag < MIN_MOTION_MAG:
            continue
        angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        usable.append((dx, dy, confidence, angle))

    if len(usable) < MIN_VALID_PHASE_SHIFTS:
        return None

    best_cluster = []
    for candidate in usable:
        cluster = [
            s for s in usable
            if angle_difference_deg(s[3], candidate[3]) <= MAX_DIRECTION_DEVIATION_DEG
        ]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
        elif len(cluster) == len(best_cluster) and cluster:
            if sum(s[2] for s in cluster) > sum(s[2] for s in best_cluster):
                best_cluster = cluster

    if len(best_cluster) < MIN_VALID_PHASE_SHIFTS:
        return None

    vx = float(np.median([s[0] for s in best_cluster]))
    vy = float(np.median([s[1] for s in best_cluster]))
    if math.hypot(vx, vy) < MIN_MOTION_MAG:
        return None

    result_angle = (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0
    agreeing = [
        s for s in best_cluster
        if angle_difference_deg(s[3], result_angle) <= MAX_DIRECTION_DEVIATION_DEG
    ]
    if len(agreeing) < MIN_VALID_PHASE_SHIFTS:
        return None

    return vx, vy, len(agreeing)


def compute_motion(masks: List[np.ndarray]):
    if not masks or int(masks[-1].sum()) < MIN_PRECIP_PIXELS_FOR_MOTION:
        return None, "no_significant_precipitation"

    center = centroid(masks[-1])
    if center is None:
        return None, "no_significant_precipitation"

    shifts = []
    for a, b in zip(masks[:-1], masks[1:]):
        s = phase_shift(a, b)
        if s is not None:
            shifts.append(s)

    vector = consistent_phase_vector(shifts)
    if vector is None:
        return None, "motion_not_reliable"

    vx, vy, _ = vector
    return (vx, vy, center), "phase_correlation_consistent"

def direction_label(vx: float, vy: float) -> str:
    angle = (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0
    dirs = [
        (22.5, "V"),
        (67.5, "JV"),
        (112.5, "J"),
        (157.5, "JZ"),
        (202.5, "Z"),
        (247.5, "SZ"),
        (292.5, "S"),
        (337.5, "SV"),
        (360.0, "V"),
    ]
    for limit, label in dirs:
        if angle < limit:
            return label
    return "V"


def get_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_dynamic_info(
    img: Image.Image,
    latest_dt: datetime,
    motion,
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = get_font(11)
    font_small = get_font(10)

    # Slovak local time via zoneinfo when available.
    try:
        from zoneinfo import ZoneInfo
        local_dt = latest_dt.astimezone(ZoneInfo("Europe/Bratislava"))
    except Exception:
        local_dt = latest_dt.astimezone(timezone(timedelta(hours=2)))

    time_text = f"Radar: {local_dt.strftime('%d.%m.%Y %H:%M')}"
    draw.rectangle((330, 374, 575, 392), fill="white")
    draw.text((335, 376), time_text, fill="black", font=font)

    if motion is not None:
        vx, vy, (cx, cy) = motion
        mag = math.hypot(vx, vy)
        ux, uy = vx / mag, vy / mag

        arrow_len = int(np.clip(45 + mag * 4, 50, 75))
        sx = float(np.clip(cx - ux * 10, 22, WIDTH - 22))
        sy = float(np.clip(cy - uy * 10, 22, MAP_BOTTOM - 15))
        ex = float(np.clip(cx + ux * arrow_len, 22, WIDTH - 22))
        ey = float(np.clip(cy + uy * arrow_len, 22, MAP_BOTTOM - 15))

        # White halo, then black arrow.
        draw.line((sx, sy, ex, ey), fill="white", width=7)
        draw.line((sx, sy, ex, ey), fill="black", width=3)

        ang = math.atan2(ey - sy, ex - sx)
        hl, hw = 11, 6
        left = (
            ex - hl * math.cos(ang) + hw * math.sin(ang),
            ey - hl * math.sin(ang) - hw * math.cos(ang),
        )
        right = (
            ex - hl * math.cos(ang) - hw * math.sin(ang),
            ey - hl * math.sin(ang) + hw * math.cos(ang),
        )
        draw.polygon([(ex, ey), left, right], fill="black")

        txt = f"Smer zrážok: {direction_label(vx, vy)}"
    else:
        txt = "Smer zrážok: —"

    draw.rectangle((330, 397, 575, 416), fill="white")
    draw.text((335, 399), txt, fill="black", font=font_small)
    return out


def enforce_palette(img: Image.Image) -> Image.Image:
    a = np.array(img.convert("RGB"), dtype=np.int32)
    palette = np.array(
        [[255, 255, 255], [0, 0, 0], [255, 0, 0]],
        dtype=np.int32,
    )
    d = a[:, :, None, :] - palette[None, None, :, :]
    idx = np.argmin(np.sum(d * d, axis=3), axis=2)
    return Image.fromarray(palette[idx].astype(np.uint8), mode="RGB")


def palette_stats(img: Image.Image):
    a = np.array(img.convert("RGB"))
    w = np.all(a == WHITE, axis=2)
    b = np.all(a == BLACK, axis=2)
    r = np.all(a == RED, axis=2)
    return {
        "white_pixels": int(w.sum()),
        "black_pixels": int(b.sum()),
        "red_pixels": int(r.sum()),
        "other_pixels": int((~(w | b | r)).sum()),
    }


def main() -> None:
    latest_dt, latest_img, source_url = get_latest_frame()
    history = get_history(latest_dt)
    if not history:
        history = [(latest_dt, latest_img)]

    frame_rgbs = [
        np.array(img.convert("RGB"), dtype=np.uint8)
        for _, img in history
    ]

    precip_masks = extract_precip_masks(frame_rgbs)
    latest_precip = precip_masks[-1]

    base = load_base_map()
    composed = compose_radar(base, latest_precip)

    motion, motion_method = compute_motion(precip_masks)

    latest = enforce_palette(composed)
    latest_arrow = enforce_palette(
        draw_dynamic_info(composed, latest_dt, motion)
    )

    stats = palette_stats(latest_arrow)
    if stats["other_pixels"] != 0:
        raise RuntimeError("Výstup obsahuje farbu mimo biela/čierna/červená.")

    latest.save(LATEST_PNG, format="PNG", optimize=True)
    latest_arrow.save(LATEST_ARROW_PNG, format="PNG", optimize=True)

    info = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "latest_radar_utc": latest_dt.isoformat(),
        "source_url": source_url,
        "mode": "fixed_basemap_overlay_v1",
        "base_map": "base_map.png",
        "precipitation_pixels": int(latest_precip.sum()),
        "motion_method": motion_method,
        "filter_min_component_pixels": MIN_DISPLAY_COMPONENT_PIXELS,
        "filter_temporal_confirmation": f"{TEMPORAL_MIN_HITS}/{TEMPORAL_WINDOW_FRAMES}",
        "motion_min_precip_pixels": MIN_PRECIP_PIXELS_FOR_MOTION,
        "palette": ["#FFFFFF", "#000000", "#FF0000"],
        "palette_stats": stats,
    }

    if motion is not None:
        vx, vy, _ = motion
        info["movement_direction"] = direction_label(vx, vy)
        info["movement_vector_px_per_5min"] = {
            "vx": round(vx, 2),
            "vy": round(vy, 2),
        }

    INFO_JSON.write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Updated: {source_url}")
    print(f"Precipitation pixels: {int(latest_precip.sum())}")
    print(f"Motion: {motion_method}")
    print(f"Palette: {stats}")


if __name__ == "__main__":
    main()
