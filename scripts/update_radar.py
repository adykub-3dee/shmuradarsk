from __future__ import annotations

import io
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
LATEST_PNG = ROOT / "latest.png"
LATEST_ARROW_PNG = ROOT / "latest_arrow.png"
INFO_JSON = ROOT / "latest_info.json"

BASE_URL = "https://www.shmu.sk/data/data002/radar-cappi_z_2_600x480-{stamp}-mosaic--.png"
FRAME_STEP_MIN = 5
LOOKBACK_HOURS = 4
FRAME_COUNT_FOR_ANALYSIS = 6
REQUEST_TIMEOUT = 20
USER_AGENT = "shmu-radar-eink/1.2"


def utc_now_rounded() -> datetime:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    return now.replace(minute=(now.minute // FRAME_STEP_MIN) * FRAME_STEP_MIN)


def make_stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d-%H%M")


def make_url(dt: datetime) -> str:
    return BASE_URL.format(stamp=make_stamp(dt))


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


def classify_masks(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = rgb.astype(np.int16)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    delta = maxc - minc
    brightness = arr.mean(axis=2)

    near_white = (brightness > 245) & (delta < 18)

    # Candidate colored pixels from original radar image.
    colored = (delta > 30) & (maxc > 70) & (~near_white)

    # Separate thin colored map lines from filled precipitation blobs.
    mask_img = Image.fromarray(np.where(colored, 255, 0).astype(np.uint8), mode="L")
    density = np.array(mask_img.filter(ImageFilter.BoxBlur(3)), dtype=np.uint8)
    precip_mask = colored & (density >= 85)
    colored_line_mask = colored & (~precip_mask)

    # Only dark or line-like pixels should become black.
    dark_mask = (~near_white) & (brightness < 165) & (delta < 90)
    black_mask = dark_mask | colored_line_mask

    # Prevent overlaps.
    black_mask = black_mask & (~precip_mask)
    return black_mask, precip_mask


def convert_to_eink(image: Image.Image) -> Tuple[Image.Image, np.ndarray]:
    rgb = np.array(image.convert("RGB"))
    black_mask, precip_mask = classify_masks(rgb)

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
        centroid = mask_centroid(mask)
        if centroid is not None:
            points.append((idx, centroid[0], centroid[1]))

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
        vx, vy, center = motion
        cx, cy = center
        mag = math.hypot(vx, vy)
        if mag > 0:
            ux, uy = vx / mag, vy / mag
            arrow_len = min(120, max(70, int(mag * 18)))
            start = (cx - ux * 20, cy - uy * 20)
            end = (cx + ux * arrow_len, cy + uy * arrow_len)
            sx = min(max(start[0], 20), w - 20)
            sy = min(max(start[1], 20), h - 20)
            ex = min(max(end[0], 20), w - 20)
            ey = min(max(end[1], 20), h - 20)

            draw.line((sx, sy, ex, ey), fill=(255, 255, 255), width=12)
            draw.line((sx, sy, ex, ey), fill=(0, 0, 0), width=7)

            head_len = 18
            head_w = 10
            angle = math.atan2(ey - sy, ex - sx)
            left = (
                ex - head_len * math.cos(angle) + head_w * math.sin(angle),
                ey - head_len * math.sin(angle) - head_w * math.cos(angle),
            )
            right = (
                ex - head_len * math.cos(angle) - head_w * math.sin(angle),
                ey - head_len * math.sin(angle) + head_w * math.cos(angle),
            )
            draw.polygon([(ex, ey), left, right], fill=(255, 255, 255))
            draw.polygon([(ex, ey), left, right], outline=(0, 0, 0))

    local_dt = latest_dt.astimezone(timezone(timedelta(hours=2)))
    label = f"Radar {local_dt.strftime('%H:%M')}"
    if motion is not None:
        label += f"  Smer: {direction_label(motion[0], motion[1])}"
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    px, py = 10, h - th - 12
    draw.rectangle((px - 4, py - 3, px + tw + 4, py + th + 3), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((px, py), label, fill=(0, 0, 0), font=font)
    return out


def write_info_json(latest_dt: datetime, source_url: str, motion: Optional[Tuple[float, float, Tuple[float, float]]]) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "latest_radar_utc": latest_dt.isoformat(),
        "latest_radar_local": latest_dt.astimezone(timezone(timedelta(hours=2))).isoformat(),
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
        _, mask = convert_to_eink(frame)
        masks.append(mask)
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
