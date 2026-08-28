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
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

LATEST_PNG = ROOT / "latest.png"
LATEST_ARROW_PNG = ROOT / "latest_arrow.png"
INDEX_HTML = ROOT / "index.html"
INFO_JSON = ROOT / "latest_info.json"

BASE_URL = "https://www.shmu.sk/data/data002/radar-cappi_z_2_600x480-{stamp}-mosaic--.png"
LOOKBACK_HOURS = 4
FRAME_STEP_MIN = 5
FRAME_COUNT_FOR_ANALYSIS = 6
REQUEST_TIMEOUT = 20
USER_AGENT = "shmu-radar-eink/1.0 (+https://github.com/)"


def utc_now_rounded() -> datetime:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rounded = (now.minute // FRAME_STEP_MIN) * FRAME_STEP_MIN
    return now.replace(minute=rounded)


def make_stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d-%H%M")


def make_url(dt: datetime) -> str:
    return BASE_URL.format(stamp=make_stamp(dt))


def fetch_image(url: str) -> Optional[Image.Image]:
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        if response.status_code != 200:
            return None
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    except Exception:
        return None


def get_latest_frame() -> Tuple[datetime, Image.Image, str]:
    probe = utc_now_rounded()
    attempts = int((LOOKBACK_HOURS * 60) / FRAME_STEP_MIN)
    for i in range(attempts):
        candidate = probe - timedelta(minutes=i * FRAME_STEP_MIN)
        url = make_url(candidate)
        image = fetch_image(url)
        if image is not None:
            return candidate, image, url
    raise RuntimeError("Nepodarilo sa nájsť žiadnu radarovú snímku SHMÚ.")


def get_history_frames(latest_dt: datetime, count: int) -> List[Tuple[datetime, Image.Image]]:
    frames: List[Tuple[datetime, Image.Image]] = []
    for i in range(count):
        candidate = latest_dt - timedelta(minutes=i * FRAME_STEP_MIN)
        image = fetch_image(make_url(candidate))
        if image is not None:
            frames.append((candidate, image))
    frames.reverse()
    return frames


# --- PÔVODNÁ DETEKCIA Z PRVEJ VERZIE ---
def classify_red_mask(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.int16)
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    delta = maxc - minc
    brightness = arr.mean(axis=2)

    near_white = (brightness > 245) & (delta < 18)
    strong_color = (delta > 35) & (maxc > 70) & (~near_white)
    return strong_color


def split_colored_mask(red_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pôvodná prvá verzia dávala všetky výrazné farby do červenej.
    Tu iba rozlíšime tenké farebné línie mapy od plôch zrážok.
    Nič ďalšie na geometrii mapy nemeníme.
    """
    mask_img = Image.fromarray((red_mask.astype(np.uint8) * 255), mode="L")

    # Lokálna hustota: tenká 1–2 px línia má nízku hustotu,
    # plocha radarového echa vysokú.
    density = np.array(mask_img.filter(ImageFilter.BoxBlur(2)), dtype=np.uint8)

    precip_mask = red_mask & (density >= 95)
    contour_mask = red_mask & (~precip_mask)

    return contour_mask, precip_mask


def convert_to_eink(image: Image.Image) -> Tuple[Image.Image, np.ndarray]:
    """
    Rovnaký základ ako úplne prvá verzia, iba opravené priradenie farieb:

      pôvodná tmavá plocha -> BIELA
      tenké farebné kontúry -> ČIERNA
      radarové plochy -> ČERVENÁ
    """
    rgb = np.array(image.convert("RGB"))
    red_mask = classify_red_mask(rgb)

    contour_mask, precip_mask = split_colored_mask(red_mask)

    # Výstup začína kompletne biely.
    out = np.full((rgb.shape[0], rgb.shape[1], 3), 255, dtype=np.uint8)

    # Kontúry mapy čierne.
    out[contour_mask] = [0, 0, 0]

    # Zrážky červené.
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

    current_center = (points[-1][1], points[-1][2])
    magnitude = math.hypot(vx, vy)
    if magnitude < 2.0:
        return None

    return vx, vy, current_center


def direction_label(vx: float, vy: float) -> str:
    angle = math.degrees(math.atan2(vy, vx))
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
    angle = (angle + 360.0) % 360.0
    for limit, label in dirs:
        if angle < limit:
            return label
    return "V"


def draw_arrow(img: Image.Image, motion: Optional[Tuple[float, float, Tuple[float, float]]]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size

    if motion is None:
        return out

    vx, vy, center = motion
    cx, cy = center
    mag = math.hypot(vx, vy)
    if mag == 0:
        return out

    ux, uy = vx / mag, vy / mag
    arrow_len = min(120, max(70, int(mag * 18)))
    start = (cx - ux * 20, cy - uy * 20)
    end = (cx + ux * arrow_len, cy + uy * arrow_len)

    sx = min(max(start[0], 20), w - 20)
    sy = min(max(start[1], 20), h - 20)
    ex = min(max(end[0], 20), w - 20)
    ey = min(max(end[1], 20), h - 20)

    # Šípka zostáva čierna s bielym podkladom.
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

    label = f"Smer zrážok: {direction_label(vx, vy)}"
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    px, py = 10, h - th - 12
    draw.rectangle((px - 4, py - 3, px + tw + 4, py + th + 3),
                   fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((px, py), label, fill=(0, 0, 0), font=font)

    return out


def write_index_html(ts_token: str) -> None:
    html = f"""<!doctype html>
<html lang="sk">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
  <meta http-equiv="refresh" content="300">
  <title>SHMÚ radar e-ink</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      background: #ffffff;
      overflow: hidden;
    }}
    body {{
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    img {{
      width: 100vw;
      height: 100vh;
      object-fit: contain;
      display: block;
      background: #ffffff;
    }}
  </style>
</head>
<body>
  <img src="latest_arrow.png?v={ts_token}" alt="SHMÚ radar">
</body>
</html>
"""
    INDEX_HTML.write_text(html, encoding="utf-8")


def write_info_json(latest_dt: datetime, source_url: str,
                    motion: Optional[Tuple[float, float, Tuple[float, float]]]) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "latest_radar_utc": latest_dt.isoformat(),
        "source_url": source_url,
    }
    if motion is not None:
        vx, vy, _ = motion
        payload["movement_direction"] = direction_label(vx, vy)
        payload["movement_vector"] = {"vx": round(vx, 2), "vy": round(vy, 2)}
    INFO_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_placeholder() -> None:
    if LATEST_PNG.exists() and LATEST_ARROW_PNG.exists() and INDEX_HTML.exists():
        return
    img = Image.new("RGB", (600, 480), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    lines = [
        "SHMÚ radar e-ink",
        "Čaká na prvé spustenie GitHub Action",
    ]
    y = 210
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((600 - tw) / 2, y), line, fill="black", font=font)
        y += 18
    img.save(LATEST_PNG)
    img.save(LATEST_ARROW_PNG)
    write_index_html("init")


def main() -> None:
    make_placeholder()
    latest_dt, latest_raw, source_url = get_latest_frame()
    history = get_history_frames(latest_dt, FRAME_COUNT_FOR_ANALYSIS)

    latest_eink, latest_mask = convert_to_eink(latest_raw)
    history_masks: List[np.ndarray] = []
    for _, frame in history:
        _, mask = convert_to_eink(frame)
        history_masks.append(mask)

    if not history_masks:
        history_masks = [latest_mask]

    motion = compute_motion_vector(history_masks)
    latest_arrow = draw_arrow(latest_eink, motion)

    latest_eink.save(LATEST_PNG)
    latest_arrow.save(LATEST_ARROW_PNG)

    ts_token = latest_dt.strftime("%Y%m%d%H%M")
    write_index_html(ts_token)
    write_info_json(latest_dt, source_url, motion)

    print(f"Updated radar from {source_url}")
    print(f"Saved: {LATEST_PNG}")
    print(f"Saved: {LATEST_ARROW_PNG}")


if __name__ == "__main__":
    main()
