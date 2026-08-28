from __future__ import annotations

import io
import json
import math
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
LATEST_PNG = ROOT / "latest.png"
LATEST_ARROW_PNG = ROOT / "latest_arrow.png"
INFO_JSON = ROOT / "latest_info.json"

BASE_URL = "https://www.shmu.sk/data/data002/radar-cappi_z_2_600x480-{stamp}-mosaic--.png"
FRAME_STEP_MIN = 5
LOOKBACK_HOURS = 4
FRAME_COUNT_FOR_ANALYSIS = 6
REQUEST_TIMEOUT = 20
USER_AGENT = "shmu-radar-eink/1.3"
LOCAL_TZ = timezone(timedelta(hours=2))


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
        return Image.open(io.BytesIO(r.content)).convert("RGB")
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
    raise RuntimeError("Nepodarilo sa nájsť žiadnu radarovú snímku SHMÚ.")


def get_history_frames(latest_dt: datetime, count: int) -> List[Tuple[datetime, Image.Image]]:
    frames: List[Tuple[datetime, Image.Image]] = []
    for i in range(count):
        dt = latest_dt - timedelta(minutes=i * FRAME_STEP_MIN)
        img = fetch_image(make_url(dt))
        if img is not None:
            frames.append((dt, img))
    frames.reverse()
    return frames


def rgb_to_hsv_arrays(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    v = maxc
    s = np.where(maxc == 0, 0, delta / np.maximum(maxc, 1e-6))
    h = np.zeros_like(maxc)
    mask = delta > 1e-6
    idx = (maxc == r) & mask
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
    idx = (maxc == g) & mask
    h[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2.0
    idx = (maxc == b) & mask
    h[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4.0
    h = h / 6.0
    return h, s, v


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: List[np.ndarray] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            q = deque([(y, x)])
            visited[y, x] = True
            coords = []
            while q:
                cy, cx = q.popleft()
                coords.append((cy, cx))
                for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            comp = np.zeros((h, w), dtype=bool)
            ys, xs = zip(*coords)
            comp[np.array(ys), np.array(xs)] = True
            comps.append(comp)
    return comps


def split_precip_vs_lines(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, s, v = rgb_to_hsv_arrays(rgb)
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    brightness = rgb.mean(axis=2)
    delta = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])

    near_white = (brightness > 242) & (delta < 20)

    # Colored pixels from radar palette or map colored outlines.
    colored = (~near_white) & (s > 0.28) & (v > 0.25)

    precip_mask = np.zeros(colored.shape, dtype=bool)
    line_mask = np.zeros(colored.shape, dtype=bool)

    for comp in connected_components(colored):
        ys, xs = np.nonzero(comp)
        area = len(xs)
        if area == 0:
            continue
        minx, maxx = xs.min(), xs.max()
        miny, maxy = ys.min(), ys.max()
        bw = maxx - minx + 1
        bh = maxy - miny + 1
        fill = area / float(bw * bh)

        # Filled or larger components are precipitation. Thin / sparse components become black lines.
        is_precip = (
            area >= 80 and (
                fill >= 0.18 or
                (bw >= 14 and bh >= 6) or
                (bh >= 14 and bw >= 6) or
                area >= 180
            )
        )
        if is_precip:
            precip_mask |= comp
        else:
            line_mask |= comp

    return precip_mask, line_mask


def convert_to_eink(image: Image.Image) -> Tuple[Image.Image, np.ndarray]:
    rgb = np.array(image.convert("RGB"))
    precip_mask, colored_line_mask = split_precip_vs_lines(rgb)

    brightness = rgb.mean(axis=2)
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    delta = maxc.astype(np.int16) - minc.astype(np.int16)
    _, s, v = rgb_to_hsv_arrays(rgb)

    near_white = (brightness > 242) & (delta < 20)

    # Keep genuinely dark text/lines black. Avoid turning mid-tone background into black.
    dark_neutral = (~near_white) & (brightness < 145) & (s < 0.30)
    dark_colored = (~near_white) & (brightness < 120) & (s >= 0.30) & (~precip_mask)
    black_mask = dark_neutral | dark_colored | colored_line_mask
    black_mask &= ~precip_mask

    out = np.full((rgb.shape[0], rgb.shape[1], 3), 255, dtype=np.uint8)
    out[black_mask] = [0, 0, 0]
    out[precip_mask] = [220, 0, 0]
    return Image.fromarray(out, mode="RGB"), precip_mask


def mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) < 150:
        return None
    return float(xs.mean()), float(ys.mean())


def compute_motion_vector(masks: List[np.ndarray]) -> Optional[Tuple[float, float, Tuple[float, float]]]:
    points: List[Tuple[int, float, float]] = []
    for idx, mask in enumerate(masks):
        c = mask_centroid(mask)
        if c is not None:
            points.append((idx, c[0], c[1]))
    if len(points) < 2:
        return None

    t = np.array([p[0] for p in points], dtype=np.float64)
    xs = np.array([p[1] for p in points], dtype=np.float64)
    ys = np.array([p[2] for p in points], dtype=np.float64)
    t_mean = t.mean()
    denom = ((t - t_mean) ** 2).sum()
    if denom == 0:
        return None

    vx = ((t - t_mean) * (xs - xs.mean())).sum() / denom
    vy = ((t - t_mean) * (ys - ys.mean())).sum() / denom
    mag = math.hypot(vx, vy)
    if mag < 2.0:
        return None
    return vx, vy, (points[-1][1], points[-1][2])


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


def draw_arrow(img: Image.Image, motion: Optional[Tuple[float, float, Tuple[float, float]]], latest_dt: datetime) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    font = ImageFont.load_default()

    if motion is not None:
        vx, vy, (cx, cy) = motion
        mag = math.hypot(vx, vy)
        if mag > 0:
            ux, uy = vx / mag, vy / mag
            arrow_len = min(120, max(70, int(mag * 18)))
            sx, sy = cx - ux * 20, cy - uy * 20
            ex, ey = cx + ux * arrow_len, cy + uy * arrow_len
            sx, sy = min(max(sx, 20), w - 20), min(max(sy, 20), h - 20)
            ex, ey = min(max(ex, 20), w - 20), min(max(ey, 20), h - 20)
            draw.line((sx, sy, ex, ey), fill=(255, 255, 255), width=12)
            draw.line((sx, sy, ex, ey), fill=(0, 0, 0), width=7)
            angle = math.atan2(ey - sy, ex - sx)
            head_len, head_w = 18, 10
            left = (ex - head_len * math.cos(angle) + head_w * math.sin(angle),
                    ey - head_len * math.sin(angle) - head_w * math.cos(angle))
            right = (ex - head_len * math.cos(angle) - head_w * math.sin(angle),
                     ey - head_len * math.sin(angle) + head_w * math.cos(angle))
            draw.polygon([(ex, ey), left, right], fill=(255, 255, 255))
            draw.polygon([(ex, ey), left, right], outline=(0, 0, 0))

    local_dt = latest_dt.astimezone(LOCAL_TZ)
    label = f"Radar {local_dt.strftime('%H:%M')}"
    if motion is not None:
        label += f"  Smer: {direction_label(motion[0], motion[1])}"
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    px, py = 10, h - th - 12
    draw.rectangle((px - 4, py - 3, px + tw + 4, py + th + 3), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((px, py), label, fill=(0, 0, 0), font=font)
    return out


def write_info_json(latest_dt: datetime, source_url: str, motion: Optional[Tuple[float, float, Tuple[float, float]]]) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "latest_radar_utc": latest_dt.isoformat(),
        "latest_radar_local": latest_dt.astimezone(LOCAL_TZ).isoformat(),
        "source_url": source_url,
    }
    if motion is not None:
        payload["movement_direction"] = direction_label(motion[0], motion[1])
        payload["movement_vector"] = {"vx": round(motion[0], 2), "vy": round(motion[1], 2)}
    INFO_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    latest_dt, latest_raw, source_url = get_latest_frame()
    history = get_history_frames(latest_dt, FRAME_COUNT_FOR_ANALYSIS)

    latest_eink, latest_mask = convert_to_eink(latest_raw)
    masks: List[np.ndarray] = []
    for _, frame in history:
        _, m = convert_to_eink(frame)
        masks.append(m)
    if not masks:
        masks = [latest_mask]

    motion = compute_motion_vector(masks)
    latest_arrow = draw_arrow(latest_eink, motion, latest_dt)
    latest_eink.save(LATEST_PNG)
    latest_arrow.save(LATEST_ARROW_PNG)
    write_info_json(latest_dt, source_url, motion)
    print(f"Updated radar from {source_url}")


if __name__ == "__main__":
    main()
