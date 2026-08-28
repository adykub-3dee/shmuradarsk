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

LOOKBACK_HOURS = 4
FRAME_STEP_MIN = 5
FRAME_COUNT = 7                     # ~30 min histórie
REQUEST_TIMEOUT = 20
USER_AGENT = "shmu-radar-eink/strict-3color-v2"

# Výstup je STRICTNE iba v týchto farbách.
WHITE = np.array([255, 255, 255], dtype=np.uint8)
BLACK = np.array([0, 0, 0], dtype=np.uint8)
RED = np.array([255, 0, 0], dtype=np.uint8)

# Ladenie pre SHMÚ radar:
COLOR_SAT_MIN = 0.20                # farebné radarové / mapové prvky
COLOR_VALUE_MIN = 0.18
STATIC_PRESENCE_MIN = 0.80          # pixel je farebný vo väčšine snímok
STATIC_RGB_STD_MAX = 10.0           # a takmer sa nemení -> statická mapa/legenda

NEUTRAL_DEEP_BLACK_LUMA = 92        # jednoznačne tmavý text / čiara
NEUTRAL_EDGE_BLACK_LUMA = 185       # tmavší pixel môže byť čierny, ak tvorí hranu
EDGE_MIN = 18.0                     # lokálny kontrast pre zachovanie textov / hraníc

MIN_PRECIP_PIXELS_FOR_MOTION = 120
PHASE_DOWNSCALE = 2
MAX_SHIFT_PER_FRAME_PX = 35         # v originálnom 600x480 obraze
MIN_PHASE_CONFIDENCE = 1.15
MIN_MOTION_MAGNITUDE = 0.7          # px / 5 min


def utc_now_rounded() -> datetime:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rounded = (now.minute // FRAME_STEP_MIN) * FRAME_STEP_MIN
    return now.replace(minute=rounded)


def make_url(dt: datetime) -> str:
    return BASE_URL.format(stamp=dt.strftime("%Y%m%d-%H%M"))


def fetch_image(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        if r.status_code != 200:
            return None
        data = r.content
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            return None
        return Image.open(io.BytesIO(data)).convert("RGB")
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

    raise RuntimeError("Nepodarilo sa nájsť aktuálnu radarovú PNG snímku SHMÚ.")


def get_history_frames(latest_dt: datetime, count: int) -> List[Tuple[datetime, Image.Image]]:
    frames: List[Tuple[datetime, Image.Image]] = []
    for i in range(count):
        dt = latest_dt - timedelta(minutes=i * FRAME_STEP_MIN)
        img = fetch_image(make_url(dt))
        if img is not None:
            frames.append((dt, img))
    frames.reverse()  # najstaršia -> najnovšia
    return frames


def rgb_to_hsv_arrays(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = rgb.astype(np.float32) / 255.0
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    delta = maxc - minc

    v = maxc
    s = np.where(maxc <= 1e-6, 0.0, delta / np.maximum(maxc, 1e-6))

    h = np.zeros_like(maxc)

    nz = delta > 1e-6
    idx = (maxc == r) & nz
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
    idx = (maxc == g) & nz
    h[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2.0
    idx = (maxc == b) & nz
    h[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4.0
    h /= 6.0

    return h, s, v


def luminance(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    return 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]


def local_edge_strength(gray: np.ndarray) -> np.ndarray:
    """
    Max. rozdiel voči 8 susedom.
    Zachová tmavé texty a jemné hranice, ale neprevedie jednoliate sivé pozadie na čiernu.
    """
    padded = np.pad(gray, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    edge = np.zeros_like(center, dtype=np.float32)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neigh = padded[1 + dy:1 + dy + gray.shape[0], 1 + dx:1 + dx + gray.shape[1]]
            edge = np.maximum(edge, np.abs(center - neigh))

    return edge


def colored_mask(rgb: np.ndarray) -> np.ndarray:
    _, s, v = rgb_to_hsv_arrays(rgb)
    return (s >= COLOR_SAT_MIN) & (v >= COLOR_VALUE_MIN)


def build_static_colored_mask(frame_rgbs: List[np.ndarray]) -> np.ndarray:
    """
    Statické farebné časti SHMÚ mapy (hranice, farebná legenda a pod.)
    rozlíšime podľa toho, že zostávajú na rovnakom mieste a takmer v rovnakej RGB farbe.
    Pohybujúce sa radarové echo sa v čase mení.
    """
    stack = np.stack(frame_rgbs, axis=0).astype(np.float32)

    color_presence = np.stack([colored_mask(x.astype(np.uint8)) for x in stack], axis=0).mean(axis=0)
    rgb_std = stack.std(axis=0).mean(axis=2)

    return (color_presence >= STATIC_PRESENCE_MIN) & (rgb_std <= STATIC_RGB_STD_MAX)


def dilate_mask(mask: np.ndarray, size: int = 3) -> np.ndarray:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.filter(ImageFilter.MaxFilter(size))
    return np.array(img, dtype=np.uint8) > 0


def cleanup_precip_mask(mask: np.ndarray) -> np.ndarray:
    """
    Len veľmi jemné čistenie: odstráni úplne osamelé jednopixelové artefakty,
    ale zachová jemné zrážky.
    """
    p = np.pad(mask.astype(np.uint8), 1, mode="constant")
    neighbours = np.zeros(mask.shape, dtype=np.uint8)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbours += p[1 + dy:1 + dy + mask.shape[0],
                            1 + dx:1 + dx + mask.shape[1]]

    return mask & (neighbours >= 2)


def make_eink_frame(
    rgb: np.ndarray,
    static_colored: np.ndarray,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Výsledok:
      WHITE  = všetko svetlé / sivé pozadie,
      BLACK  = texty, kontúry a statické mapové prvky,
      RED    = aktuálne radarové echo.

    ŽIADNA SIVÁ sa do výsledku nedostane.
    """
    current_colored = colored_mask(rgb)

    # Farebné pixely, ktoré nie sú statickou mapou, považujeme za radarové echo.
    precip = current_colored & (~static_colored)
    precip = cleanup_precip_mask(precip)

    gray = luminance(rgb)
    edge = local_edge_strength(gray)
    _, sat, _ = rgb_to_hsv_arrays(rgb)

    # Neutrálne tmavé pixely: texty a mapové čiary.
    neutral = sat < COLOR_SAT_MIN

    black_deep = neutral & (gray <= NEUTRAL_DEEP_BLACK_LUMA)
    black_edge = neutral & (gray <= NEUTRAL_EDGE_BLACK_LUMA) & (edge >= EDGE_MIN)

    # Statické farebné mapové prvky tiež čierne.
    black = black_deep | black_edge | static_colored

    # Nesmie vzniknúť čierny obrys z antialiasingu okolo radarového echa.
    precip_halo = dilate_mask(precip, 3)
    black &= ~precip_halo

    out = np.empty_like(rgb, dtype=np.uint8)
    out[:, :] = WHITE
    out[black] = BLACK
    out[precip] = RED

    return Image.fromarray(out, mode="RGB"), precip


def centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_PRECIP_PIXELS_FOR_MOTION:
        return None
    return float(xs.mean()), float(ys.mean())


def downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return mask.astype(np.float32)
    h, w = mask.shape
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize(
        (max(1, w // factor), max(1, h // factor)),
        resample=Image.Resampling.BILINEAR,
    )
    img = img.filter(ImageFilter.GaussianBlur(1.0))
    return np.array(img, dtype=np.float32) / 255.0


def phase_shift(a_mask: np.ndarray, b_mask: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Fázová korelácia: odhadne globálny posun B voči A.
    Vracia dx, dy v pixeloch originálneho obrazu + confidence.
    """
    if a_mask.sum() < MIN_PRECIP_PIXELS_FOR_MOTION or b_mask.sum() < MIN_PRECIP_PIXELS_FOR_MOTION:
        return None

    a = downsample_mask(a_mask, PHASE_DOWNSCALE)
    b = downsample_mask(b_mask, PHASE_DOWNSCALE)

    a = a - a.mean()
    b = b - b.mean()

    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)

    cross = fb * np.conj(fa)
    denom = np.abs(cross)
    cross = np.divide(cross, np.maximum(denom, 1e-9))

    corr = np.abs(np.fft.ifft2(cross))
    py, px = np.unravel_index(np.argmax(corr), corr.shape)

    h, w = corr.shape
    if py > h // 2:
        py -= h
    if px > w // 2:
        px -= w

    # Confidence: peak vs. druhý najlepší peak mimo malého okolia.
    peak_y = py % h
    peak_x = px % w
    corr2 = corr.copy()

    y0 = max(0, peak_y - 2)
    y1 = min(h, peak_y + 3)
    x0 = max(0, peak_x - 2)
    x1 = min(w, peak_x + 3)
    corr2[y0:y1, x0:x1] = 0

    peak = float(corr[peak_y, peak_x])
    second = float(corr2.max())
    confidence = peak / max(second, 1e-9)

    dx = float(px * PHASE_DOWNSCALE)
    dy = float(py * PHASE_DOWNSCALE)

    if abs(dx) > MAX_SHIFT_PER_FRAME_PX or abs(dy) > MAX_SHIFT_PER_FRAME_PX:
        return None
    if confidence < MIN_PHASE_CONFIDENCE:
        return None

    return dx, dy, confidence


def centroid_motion(masks: List[np.ndarray]) -> Optional[Tuple[float, float]]:
    """
    Záložná metóda, ak fázová korelácia nemá dostatočnú istotu.
    """
    points: List[Tuple[int, float, float]] = []

    for i, mask in enumerate(masks):
        c = centroid(mask)
        if c is not None:
            points.append((i, c[0], c[1]))

    if len(points) < 2:
        return None

    t = np.array([p[0] for p in points], dtype=np.float64)
    xs = np.array([p[1] for p in points], dtype=np.float64)
    ys = np.array([p[2] for p in points], dtype=np.float64)

    tm = t.mean()
    denom = ((t - tm) ** 2).sum()
    if denom <= 1e-9:
        return None

    vx = float(((t - tm) * (xs - xs.mean())).sum() / denom)
    vy = float(((t - tm) * (ys - ys.mean())).sum() / denom)

    if math.hypot(vx, vy) < MIN_MOTION_MAGNITUDE:
        return None

    return vx, vy


def compute_motion(
    masks: List[np.ndarray],
) -> Tuple[Optional[Tuple[float, float, Tuple[float, float]]], str]:
    shifts: List[Tuple[float, float, float]] = []

    for a, b in zip(masks[:-1], masks[1:]):
        s = phase_shift(a, b)
        if s is not None:
            shifts.append(s)

    current_center = centroid(masks[-1]) if masks else None

    if len(shifts) >= 2 and current_center is not None:
        dxs = np.array([x[0] for x in shifts], dtype=np.float64)
        dys = np.array([x[1] for x in shifts], dtype=np.float64)

        vx = float(np.median(dxs))
        vy = float(np.median(dys))

        if math.hypot(vx, vy) >= MIN_MOTION_MAGNITUDE:
            return (vx, vy, current_center), "phase_correlation"

    fallback = centroid_motion(masks)
    if fallback is not None and current_center is not None:
        return (fallback[0], fallback[1], current_center), "centroid_fallback"

    return None, "undetermined"


def direction_label(vx: float, vy: float) -> str:
    # Obrazové súradnice: +x doprava, +y dole.
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


def draw_arrow(
    img: Image.Image,
    motion: Optional[Tuple[float, float, Tuple[float, float]]],
) -> Image.Image:
    out = img.copy()
    if motion is None:
        return out

    draw = ImageDraw.Draw(out)
    w, h = out.size

    vx, vy, (cx, cy) = motion
    mag = math.hypot(vx, vy)
    if mag <= 1e-9:
        return out

    ux = vx / mag
    uy = vy / mag

    # Smer je dôležitejší než presná dĺžka; preto je šípka dobre viditeľná.
    arrow_len = int(np.clip(75 + mag * 8, 85, 140))

    sx = cx - ux * 18
    sy = cy - uy * 18
    ex = cx + ux * arrow_len
    ey = cy + uy * arrow_len

    sx = float(np.clip(sx, 20, w - 20))
    sy = float(np.clip(sy, 20, h - 20))
    ex = float(np.clip(ex, 20, w - 20))
    ey = float(np.clip(ey, 20, h - 20))

    # Biely podklad + čierna šípka.
    draw.line((sx, sy, ex, ey), fill=(255, 255, 255), width=13)
    draw.line((sx, sy, ex, ey), fill=(0, 0, 0), width=7)

    angle = math.atan2(ey - sy, ex - sx)
    head_len = 20
    head_w = 12

    left = (
        ex - head_len * math.cos(angle) + head_w * math.sin(angle),
        ey - head_len * math.sin(angle) - head_w * math.cos(angle),
    )
    right = (
        ex - head_len * math.cos(angle) - head_w * math.sin(angle),
        ey - head_len * math.sin(angle) + head_w * math.cos(angle),
    )

    draw.polygon([(ex, ey), left, right], fill=(0, 0, 0))

    label = f"Smer zrazok: {direction_label(vx, vy)}"
    font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    px = 10
    py = h - th - 12

    draw.rectangle(
        (px - 5, py - 4, px + tw + 5, py + th + 4),
        fill=(255, 255, 255),
        outline=(0, 0, 0),
        width=1,
    )
    draw.text((px, py), label, fill=(0, 0, 0), font=font)

    return out


def enforce_exact_palette(img: Image.Image) -> Image.Image:
    """
    Posledná poistka: po kreslení nesmie v PNG zostať žiadna sivá
    ani antialiasovaný medzistupeň. Každý pixel sa priradí presne k
    bielej, čiernej alebo červenej.
    """
    rgb = np.array(img.convert("RGB"), dtype=np.int32)

    palette = np.array(
        [
            [255, 255, 255],
            [0, 0, 0],
            [255, 0, 0],
        ],
        dtype=np.int32,
    )

    diff = rgb[:, :, None, :] - palette[None, None, :, :]
    dist = np.sum(diff * diff, axis=3)
    idx = np.argmin(dist, axis=2)

    out = palette[idx].astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def palette_stats(img: Image.Image) -> dict:
    arr = np.array(img.convert("RGB"))
    return {
        "white_pixels": int(np.all(arr == WHITE, axis=2).sum()),
        "black_pixels": int(np.all(arr == BLACK, axis=2).sum()),
        "red_pixels": int(np.all(arr == RED, axis=2).sum()),
        "other_pixels": int(
            (~(
                np.all(arr == WHITE, axis=2)
                | np.all(arr == BLACK, axis=2)
                | np.all(arr == RED, axis=2)
            )).sum()
        ),
    }


def write_info_json(
    latest_dt: datetime,
    source_url: str,
    motion: Optional[Tuple[float, float, Tuple[float, float]]],
    motion_method: str,
    stats: dict,
) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "latest_radar_utc": latest_dt.isoformat(),
        "source_url": source_url,
        "mode": "strict_3color_eink_v2",
        "palette": ["#FFFFFF", "#000000", "#FF0000"],
        "palette_stats": stats,
        "motion_method": motion_method,
    }

    if motion is not None:
        vx, vy, _ = motion
        payload["movement_direction"] = direction_label(vx, vy)
        payload["movement_vector_px_per_5min"] = {
            "vx": round(vx, 2),
            "vy": round(vy, 2),
        }

    INFO_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    latest_dt, latest_img, source_url = get_latest_frame()
    history = get_history_frames(latest_dt, FRAME_COUNT)

    # Ak SHMÚ niektoré medzisnímky nemá, použijeme všetky dostupné.
    if not history:
        history = [(latest_dt, latest_img)]

    frame_rgbs = [np.array(img.convert("RGB"), dtype=np.uint8) for _, img in history]

    # Statická mapa sa odvodí zo všetkých dostupných snímok.
    static_colored = build_static_colored_mask(frame_rgbs)

    eink_frames: List[Image.Image] = []
    precip_masks: List[np.ndarray] = []

    for rgb in frame_rgbs:
        eink, precip = make_eink_frame(rgb, static_colored)
        eink_frames.append(eink)
        precip_masks.append(precip)

    # Najnovšia snímka.
    latest_eink = eink_frames[-1]
    latest_eink = enforce_exact_palette(latest_eink)

    motion, motion_method = compute_motion(precip_masks)

    with_arrow = draw_arrow(latest_eink, motion)
    with_arrow = enforce_exact_palette(with_arrow)

    # GARANCIA: žiadna sivá.
    stats = palette_stats(with_arrow)
    if stats["other_pixels"] != 0:
        raise RuntimeError(
            f"Interná chyba: vo výstupe zostalo {stats['other_pixels']} pixelov mimo 3-farebnej palety."
        )

    latest_eink.save(LATEST_PNG, format="PNG", optimize=True)
    with_arrow.save(LATEST_ARROW_PNG, format="PNG", optimize=True)

    write_info_json(
        latest_dt=latest_dt,
        source_url=source_url,
        motion=motion,
        motion_method=motion_method,
        stats=stats,
    )

    print(f"Updated radar: {source_url}")
    print(f"Mode: strict 3-color e-ink")
    print(f"Motion: {motion_method}")
    print(f"Palette stats: {stats}")
    print(f"Saved: {LATEST_PNG}")
    print(f"Saved: {LATEST_ARROW_PNG}")


if __name__ == "__main__":
    main()
