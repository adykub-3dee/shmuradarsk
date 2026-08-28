from __future__ import annotations

import io
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == 'scripts' else Path.cwd()
LATEST_PNG = ROOT / "latest.png"
LATEST_ARROW_PNG = ROOT / "latest_arrow.png"
INFO_JSON = ROOT / "latest_info.json"

BASE_URL = "https://www.shmu.sk/data/data002/radar-cappi_z_2_600x480-{stamp}-mosaic--.png"
LOOKBACK_HOURS = 4
FRAME_STEP_MIN = 5
FRAME_COUNT_FOR_ANALYSIS = 8      # ~35 minút
REQUEST_TIMEOUT = 20
USER_AGENT = "shmu-radar-eink/2.0 (+https://github.com/)"
LOCAL_TZ = ZoneInfo("Europe/Bratislava")

RED = (220, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


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
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def get_latest_frame() -> Tuple[datetime, Image.Image, str]:
    probe = utc_now_rounded()
    attempts = int((LOOKBACK_HOURS * 60) / FRAME_STEP_MIN)
    for i in range(attempts):
        dt = probe - timedelta(minutes=i * FRAME_STEP_MIN)
        url = make_url(dt)
        image = fetch_image(url)
        if image is not None:
            return dt, image, url
    raise RuntimeError("Nepodarilo sa nájsť žiadnu radarovú snímku SHMÚ.")


def get_history_frames(latest_dt: datetime, count: int) -> List[Tuple[datetime, Image.Image]]:
    frames: List[Tuple[datetime, Image.Image]] = []
    for i in range(count):
        dt = latest_dt - timedelta(minutes=i * FRAME_STEP_MIN)
        image = fetch_image(make_url(dt))
        if image is not None:
            frames.append((dt, image))
    frames.reverse()
    return frames


def _dilate(mask: np.ndarray, size: int = 5) -> np.ndarray:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.filter(ImageFilter.MaxFilter(size))
    return np.array(img) > 0


def build_eink_frame(frames: List[Image.Image]) -> Tuple[Image.Image, np.ndarray]:
    """
    Vytvorí 3-farebný e-ink obraz:
      - biela: pozadie
      - čierna: statické kontúry mapy
      - červená: dynamické radarové echo

    Základný trik: hranice/mapa sú medzi snímkami statické, radarové echo sa mení.
    Tým sa statické farebné hranice už nepomýlia so zrážkami.
    """
    if not frames:
        raise ValueError("Chýbajú radarové snímky")

    stack = np.stack([np.asarray(im.convert("RGB"), dtype=np.int16) for im in frames], axis=0)
    current = stack[-1]
    baseline = np.median(stack, axis=0).astype(np.int16)

    # Ako veľmi sa daný pixel počas časového okna menil.
    temporal_range = stack.max(axis=0) - stack.min(axis=0)
    temporal_activity = temporal_range.max(axis=2)

    # Farebnosť aktuálneho pixela. Radarová škála je farebná, mapa väčšinou statická.
    cur_max = current.max(axis=2)
    cur_min = current.min(axis=2)
    saturation = cur_max - cur_min

    # Rozdiel aktuálnej snímky od časového mediánu.
    delta_now = np.abs(current - baseline).max(axis=2)

    # Dynamické farebné jadrá radarového echa.
    dynamic_seed = (
        (saturation >= 28)
        & (cur_max >= 75)
        & ((temporal_activity >= 22) | (delta_now >= 18))
    )

    # Rozšírenie jadra zachytí aj svetlejšie okraje zrážok, ale stále iba v okolí dynamiky.
    nearby_dynamic = _dilate(dynamic_seed, 5)
    colored_candidate = (saturation >= 18) & (cur_max >= 65)
    precip_mask = dynamic_seed | (nearby_dynamic & colored_candidate)

    # Statická mapa: medián viacerých snímok + detekcia hrán, nie plošné stmavenie.
    static_consistency = temporal_activity <= 12
    base_rgb = np.clip(baseline, 0, 255).astype(np.uint8)
    gray = Image.fromarray(base_rgb, mode="RGB").convert("L")
    edges = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.uint8)

    # Silnejšie statické hrany = kontúry. Jemne ich zosilníme pre e-ink.
    contour_seed = (edges >= 30) & static_consistency
    contour_mask = _dilate(contour_seed, 3)

    # Nech radar vždy vyhrá nad čiernou mapou.
    contour_mask &= ~_dilate(precip_mask, 3)

    out = np.full((current.shape[0], current.shape[1], 3), 255, dtype=np.uint8)
    out[contour_mask] = BLACK
    out[precip_mask] = RED

    # Odstránenie rámika, ktorý FIND_EDGES zvykne vytvoriť na okrajoch.
    out[:2, :, :] = WHITE
    out[-2:, :, :] = WHITE
    out[:, :2, :] = WHITE
    out[:, -2:, :] = WHITE

    return Image.fromarray(out, mode="RGB"), precip_mask


def _downsample_mask(mask: np.ndarray, factor: int = 4) -> np.ndarray:
    h, w = mask.shape
    nh, nw = max(1, h // factor), max(1, w // factor)
    im = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    im = im.resize((nw, nh), Image.Resampling.NEAREST)
    return np.array(im) > 0


def _shift_overlap(a: np.ndarray, b: np.ndarray, dx: int, dy: int) -> float:
    h, w = a.shape
    if abs(dx) >= w or abs(dy) >= h:
        return 0.0

    ax0 = max(0, -dx); ax1 = min(w, w - dx)
    ay0 = max(0, -dy); ay1 = min(h, h - dy)
    bx0 = max(0, dx);  bx1 = min(w, w + dx)
    by0 = max(0, dy);  by1 = min(h, h + dy)

    aa = a[ay0:ay1, ax0:ax1]
    bb = b[by0:by1, bx0:bx1]
    if aa.size == 0 or bb.size == 0:
        return 0.0
    na = int(aa.sum()); nb = int(bb.sum())
    if na < 8 or nb < 8:
        return 0.0
    inter = int(np.logical_and(aa, bb).sum())
    return inter / math.sqrt(na * nb)


def estimate_motion(masks: List[np.ndarray]) -> Optional[Tuple[float, float, Tuple[float, float], float]]:
    """Odhad dominantného posunu zrážok pomocou korelácie binárnych masiek."""
    valid = [m for m in masks if int(m.sum()) >= 80]
    if len(valid) < 2:
        return None

    factor = 4
    small = [_downsample_mask(m, factor) for m in valid]
    vectors = []

    for prev, curr in zip(small[:-1], small[1:]):
        best = (0.0, 0, 0)
        # ±8 px v zmenšenej mape = ±32 px v origináli za 5 min.
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                score = _shift_overlap(prev, curr, dx, dy)
                if score > best[0]:
                    best = (score, dx, dy)
        score, dx, dy = best
        if score >= 0.12:
            vectors.append((dx * factor, dy * factor, score))

    if not vectors:
        return None

    weights = np.array([v[2] for v in vectors], dtype=float)
    vx = float(np.average([v[0] for v in vectors], weights=weights))
    vy = float(np.average([v[1] for v in vectors], weights=weights))
    confidence = float(np.clip(weights.mean(), 0.0, 1.0))

    if math.hypot(vx, vy) < 2.5:
        return None

    ys, xs = np.nonzero(valid[-1])
    if len(xs) == 0:
        return None
    center = (float(xs.mean()), float(ys.mean()))
    return vx, vy, center, confidence


def direction_label(vx: float, vy: float) -> str:
    # obrazové súradnice: +x vpravo, +y dole
    angle = (math.degrees(math.atan2(vy, vx)) + 360) % 360
    labels = ["V", "JV", "J", "JZ", "Z", "SZ", "S", "SV"]
    idx = int((angle + 22.5) // 45) % 8
    return labels[idx]


def _font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_arrow_and_info(
    img: Image.Image,
    motion: Optional[Tuple[float, float, Tuple[float, float], float]],
    radar_dt_utc: datetime,
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size

    # Lokálny čas radarovej snímky - používateľ nemusí riešiť UTC v zdroji SHMÚ.
    local_dt = radar_dt_utc.astimezone(LOCAL_TZ)
    time_text = f"Radar {local_dt:%H:%M}"
    small_font = _font(16, bold=True)

    tb = draw.textbbox((0, 0), time_text, font=small_font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.rectangle((8, 8, 18 + tw, 16 + th), fill=WHITE)
    draw.text((13, 11), time_text, fill=BLACK, font=small_font)

    if motion is None:
        return out

    vx, vy, center, confidence = motion
    mag = math.hypot(vx, vy)
    ux, uy = vx / mag, vy / mag

    # Dlhá, ľahko čitateľná čierna šípka.
    arrow_len = min(135, max(85, int(mag * 5.5)))
    cx, cy = center
    sx = cx - ux * 25
    sy = cy - uy * 25
    ex = cx + ux * arrow_len
    ey = cy + uy * arrow_len

    # Posun celej šípky dovnútra plátna.
    margin = 28
    dx_fix = 0.0; dy_fix = 0.0
    if min(sx, ex) < margin: dx_fix += margin - min(sx, ex)
    if max(sx, ex) > w - margin: dx_fix -= max(sx, ex) - (w - margin)
    if min(sy, ey) < margin: dy_fix += margin - min(sy, ey)
    if max(sy, ey) > h - margin: dy_fix -= max(sy, ey) - (h - margin)
    sx += dx_fix; ex += dx_fix; sy += dy_fix; ey += dy_fix

    # Biela podkladová čiara + čierna šípka.
    draw.line((sx, sy, ex, ey), fill=WHITE, width=15)
    draw.line((sx, sy, ex, ey), fill=BLACK, width=8)

    ang = math.atan2(ey - sy, ex - sx)
    head_len, head_half = 23, 13
    left = (ex - head_len * math.cos(ang) + head_half * math.sin(ang),
            ey - head_len * math.sin(ang) - head_half * math.cos(ang))
    right = (ex - head_len * math.cos(ang) - head_half * math.sin(ang),
             ey - head_len * math.sin(ang) + head_half * math.cos(ang))
    draw.polygon([(ex, ey), left, right], fill=WHITE)
    # mierne väčšia biela hlava a potom čierna vnútorná hlava
    inner_len, inner_half = 20, 10
    ileft = (ex - inner_len * math.cos(ang) + inner_half * math.sin(ang),
             ey - inner_len * math.sin(ang) - inner_half * math.cos(ang))
    iright = (ex - inner_len * math.cos(ang) - inner_half * math.sin(ang),
              ey - inner_len * math.sin(ang) + inner_half * math.cos(ang))
    draw.polygon([(ex, ey), ileft, iright], fill=BLACK)

    label = f"SMER ZRAZOK: {direction_label(vx, vy)}"
    font = _font(17, bold=True)
    bb = draw.textbbox((0, 0), label, font=font)
    lw, lh = bb[2] - bb[0], bb[3] - bb[1]
    x, y = 10, h - lh - 16
    draw.rectangle((x - 4, y - 4, x + lw + 6, y + lh + 5), fill=WHITE, outline=BLACK, width=2)
    draw.text((x, y), label, fill=BLACK, font=font)

    return out


def write_info_json(latest_dt: datetime, source_url: str, motion) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "latest_radar_utc": latest_dt.isoformat(),
        "latest_radar_local": latest_dt.astimezone(LOCAL_TZ).isoformat(),
        "source_url": source_url,
    }
    if motion is not None:
        vx, vy, _, confidence = motion
        payload["movement_direction"] = direction_label(vx, vy)
        payload["movement_vector"] = {"vx": round(vx, 2), "vy": round(vy, 2)}
        payload["movement_confidence"] = round(confidence, 3)
    INFO_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    latest_dt, latest_raw, source_url = get_latest_frame()
    history = get_history_frames(latest_dt, FRAME_COUNT_FOR_ANALYSIS)

    # Ak niektoré staršie snímky chýbajú, stále musí byť zahrnutá aktuálna.
    frames = [im for _, im in history]
    if not frames or history[-1][0] != latest_dt:
        frames.append(latest_raw)

    eink, precip_current = build_eink_frame(frames)

    precip_masks: List[np.ndarray] = []
    # Pre smer vytvoríme masku každej snímky voči rovnakému časovému oknu.
    # Posuvné okná nie sú potrebné; dynamická mapa je vo všetkých maskách konzistentná.
    if len(frames) >= 2:
        stack = np.stack([np.asarray(im.convert("RGB"), dtype=np.int16) for im in frames], axis=0)
        baseline = np.median(stack, axis=0).astype(np.int16)
        temporal_range = stack.max(axis=0) - stack.min(axis=0)
        temporal_activity = temporal_range.max(axis=2)
        for arr in stack:
            cur_max = arr.max(axis=2); cur_min = arr.min(axis=2)
            saturation = cur_max - cur_min
            delta = np.abs(arr - baseline).max(axis=2)
            seed = (saturation >= 28) & (cur_max >= 75) & ((temporal_activity >= 22) | (delta >= 18))
            near = _dilate(seed, 5)
            colored = (saturation >= 18) & (cur_max >= 65)
            precip_masks.append(seed | (near & colored))
    else:
        precip_masks = [precip_current]

    motion = estimate_motion(precip_masks)
    final = draw_arrow_and_info(eink, motion, latest_dt)

    eink.save(LATEST_PNG)
    final.save(LATEST_ARROW_PNG)
    write_info_json(latest_dt, source_url, motion)

    print(f"Updated radar from {source_url}")
    print(f"Local radar time: {latest_dt.astimezone(LOCAL_TZ):%Y-%m-%d %H:%M}")
    if motion:
        print(f"Direction: {direction_label(motion[0], motion[1])}, confidence={motion[3]:.2f}")
    else:
        print("Direction: not determined")


if __name__ == "__main__":
    main()
