#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import tifffile as tiff
from PIL import Image

try:
    import openslide  # type: ignore
except Exception:  # pragma: no cover
    openslide = None

CHANNEL_COLORS = [
    (0, 102, 255),
    (255, 96, 32),
    (220, 30, 180),
    (80, 220, 255),
    (150, 255, 120),
    (255, 170, 0),
    (190, 120, 255),
    (255, 70, 120),
    (0, 210, 160),
    (255, 215, 65),
    (135, 200, 255),
    (255, 120, 180),
    (100, 255, 70),
    (235, 130, 40),
    (170, 170, 255),
    (255, 95, 95),
    (90, 255, 220),
    (250, 200, 40),
    (120, 220, 130),
    (255, 145, 55),
    (190, 90, 255),
    (255, 115, 150),
    (110, 230, 255),
]


@dataclass
class ImageSlotState:
    loaded: bool = False
    path: str = ""
    mode: str = "channels"  # channels | rgb
    preview_cyx: np.ndarray | None = None
    preview_rgb: np.ndarray | None = None
    full_shape: Tuple[int, int, int] | None = None
    step: int = 1
    channel_names: List[str] = field(default_factory=list)
    channel_png_cache: Dict[int, bytes] = field(default_factory=dict)
    overlay_png_cache: Dict[Tuple[int, ...], bytes] = field(default_factory=dict)
    base_png_cache: bytes | None = None


@dataclass
class ViewerState:
    he_preview: np.ndarray | None = None
    he_path: str = ""
    he_shape: Tuple[int, int, int] | None = None
    he_step: int = 1
    image_slots: Dict[str, ImageSlotState] = field(default_factory=dict)


STATE = ViewerState()


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def _json(handler: BaseHTTPRequestHandler, payload: dict, code: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _png(handler: BaseHTTPRequestHandler, payload: bytes, code: int = 200) -> None:
    handler.send_response(code)
    handler.send_header("Content-Type", "image/png")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _to_png_bytes(rgb: np.ndarray) -> bytes:
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return buff.getvalue()


def _normalize_u8(channel: np.ndarray) -> np.ndarray:
    ch = channel.astype(np.float32)
    lo = float(np.percentile(ch, 1.0))
    hi = float(np.percentile(ch, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def _normalize_rgb_u8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    out = arr.astype(np.float32)
    lo = float(np.percentile(out, 1.0))
    hi = float(np.percentile(out, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((out - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def _axes_indices(axes: str | None) -> tuple[int | None, int | None, int | None]:
    if not axes:
        return None, None, None
    ax = axes.upper()
    y = ax.find("Y") if "Y" in ax else None
    x = ax.find("X") if "X" in ax else None
    c = ax.find("S") if "S" in ax else (ax.find("C") if "C" in ax else None)
    return y, x, c


def _yx_shape(shape: Sequence[int], axes: str | None) -> tuple[int, int]:
    y, x, _ = _axes_indices(axes)
    if y is not None and x is not None:
        return int(shape[y]), int(shape[x])
    if len(shape) >= 2:
        return int(shape[-2]), int(shape[-1])
    raise ValueError(f"Cannot infer YX shape from shape={shape}, axes={axes}")


def _extract_ome_channel_names(ome_xml: str | None) -> List[str]:
    if not ome_xml:
        return []
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return []

    names: List[str] = []
    for ch in root.findall(".//{*}Channel"):
        name = ch.attrib.get("Name") or ch.attrib.get("ID")
        if name:
            names.append(name)
    return names


def _is_placeholder_channel_name(name: str) -> bool:
    s = name.strip()
    if not s:
        return True
    s_low = s.lower()
    # Handles common generic forms:
    # channel:0, channel:0:1, ch-2, c3, 12, 0:1
    if re.fullmatch(r"(?:(?:channel|ch|c)[\s:_-]*)?\d+(?:[\s:_-]*\d+)*", s_low) is not None:
        return True
    return False


def _resolve_channel_names(
    raw_names: List[str], n_channels: int, fallback_names: List[str]
) -> List[str]:
    fallback = [str(x).strip() for x in fallback_names if str(x).strip()]
    raw = [str(x).strip() for x in raw_names[:n_channels]]
    raw_has_info = any((not _is_placeholder_channel_name(n)) for n in raw)

    out: List[str] = []
    for i in range(n_channels):
        generic = f"Channel-{i+1}"
        f = fallback[i] if i < len(fallback) else generic
        if not raw_has_info:
            out.append(f)
            continue

        if i < len(raw) and raw[i] and not _is_placeholder_channel_name(raw[i]):
            out.append(raw[i])
        else:
            out.append(f)
    return out


def _choose_level(series: tiff.TiffPageSeries, max_dim: int):
    levels = list(getattr(series, "levels", [series]))
    best = None
    best_m = None

    for lvl in levels:
        h, w = _yx_shape(lvl.shape, getattr(lvl, "axes", None))
        m = max(h, w)
        if m >= max_dim and (best is None or m < best_m):
            best = lvl
            best_m = m

    if best is None:
        best = max(levels, key=lambda lv: max(_yx_shape(lv.shape, getattr(lv, "axes", None))))
    return best


def _choose_he_series(tf: tiff.TiffFile):
    best = None
    best_area = -1
    for s in tf.series:
        try:
            h, w = _yx_shape(s.shape, getattr(s, "axes", None))
            area = h * w
            if area > best_area:
                best_area = area
                best = s
        except Exception:
            continue
    return best if best is not None else tf.series[0]


def _choose_image_series(tf: tiff.TiffFile):
    candidates = []
    for s in tf.series:
        axes = getattr(s, "axes", None)
        y, x, c = _axes_indices(axes)
        if y is None or x is None:
            continue
        if c is not None and int(s.shape[c]) >= 1:
            h, w = _yx_shape(s.shape, axes)
            candidates.append((h * w, s))

    if candidates:
        candidates.sort(key=lambda it: it[0], reverse=True)
        return candidates[0][1]
    return _choose_he_series(tf)


def _array_to_rgb(arr: np.ndarray, axes: str | None) -> np.ndarray:
    if arr.ndim == 2:
        return np.repeat(_normalize_u8(arr)[..., None], 3, axis=2)

    if axes:
        y, x, c = _axes_indices(axes)
        if y is not None and x is not None:
            sl: List[object] = [0] * arr.ndim
            sl[y] = slice(None)
            sl[x] = slice(None)
            if c is not None:
                sl[c] = slice(None)
            arr2 = arr[tuple(sl)]
            kept = [i for i, v in enumerate(sl) if isinstance(v, slice)]
            y2 = kept.index(y)
            x2 = kept.index(x)

            if c is not None:
                c2 = kept.index(c)
                arr2 = np.transpose(arr2, (y2, x2, c2))
                if arr2.shape[2] >= 3:
                    return _normalize_rgb_u8(arr2[..., :3])
                return _normalize_rgb_u8(np.repeat(arr2[..., :1], 3, axis=2))

            arr2 = np.transpose(arr2, (y2, x2))
            return np.repeat(_normalize_u8(arr2)[..., None], 3, axis=2)

    if arr.ndim >= 3 and arr.shape[-1] >= 3:
        return _normalize_rgb_u8(arr[..., :3])
    if arr.ndim >= 3 and arr.shape[0] >= 3:
        return _normalize_rgb_u8(np.moveaxis(arr[:3], 0, -1))

    raise ValueError(f"Cannot convert to RGB. shape={arr.shape}, axes={axes}")


def _array_to_cyx(arr: np.ndarray, axes: str | None) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, :, :]

    if axes:
        y, x, c = _axes_indices(axes)
        if y is not None and x is not None and c is not None:
            sl: List[object] = [0] * arr.ndim
            sl[y] = slice(None)
            sl[x] = slice(None)
            sl[c] = slice(None)
            arr2 = arr[tuple(sl)]
            kept = [i for i, v in enumerate(sl) if isinstance(v, slice)]
            y2 = kept.index(y)
            x2 = kept.index(x)
            c2 = kept.index(c)
            return np.transpose(arr2, (c2, y2, x2))

        ax = axes.upper()
        if ax in {"CYX", "SYX"} and arr.ndim == 3:
            return arr
        if ax in {"YXC", "YXS"} and arr.ndim == 3:
            return np.moveaxis(arr, -1, 0)

    if arr.ndim == 3:
        if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= arr.shape[2]:
            return arr
        return np.moveaxis(arr, -1, 0)

    raise ValueError(f"Cannot convert to CYX. shape={arr.shape}, axes={axes}")


def _downsample_rgb(arr: np.ndarray, max_dim: int) -> tuple[np.ndarray, int]:
    h, w = arr.shape[0], arr.shape[1]
    step = max(1, int(np.ceil(max(h, w) / max_dim)))
    return arr[::step, ::step, :], step


def _downsample_cyx(arr: np.ndarray, max_dim: int) -> tuple[np.ndarray, int]:
    h, w = arr.shape[1], arr.shape[2]
    step = max(1, int(np.ceil(max(h, w) / max_dim)))
    return arr[:, ::step, ::step], step


def _read_rgb_with_openslide(path: str, max_dim: int) -> tuple[np.ndarray, tuple[int, int, int], int]:
    if openslide is None:
        raise RuntimeError("OpenSlide python package is not available.")

    slide = openslide.OpenSlide(path)
    w0, h0 = slide.level_dimensions[0]

    best_level = len(slide.level_dimensions) - 1
    for idx, (w, h) in enumerate(slide.level_dimensions):
        if max(w, h) >= max_dim:
            best_level = idx
    w, h = slide.level_dimensions[best_level]

    region = slide.read_region((0, 0), best_level, (w, h)).convert("RGB")
    arr = np.asarray(region, dtype=np.uint8)
    arr_ds, step_local = _downsample_rgb(arr, max_dim=max_dim)
    step = max(1, int(np.ceil(max(w0, h0) / max(arr_ds.shape[0], arr_ds.shape[1]))))
    return arr_ds, (h0, w0, 3), max(step, step_local)


def _read_he_generic(path: str, max_dim: int) -> tuple[np.ndarray, tuple[int, int, int], int, str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        arr_ds, step = _downsample_rgb(arr, max_dim=max_dim)
        return arr_ds, arr.shape, step, "pillow"

    try:
        with tiff.TiffFile(path) as tf:
            series = _choose_he_series(tf)
            level = _choose_level(series, max_dim=max_dim)
            arr = level.asarray()
            axes = getattr(level, "axes", getattr(series, "axes", None))
            rgb = _array_to_rgb(arr, axes)
            full_h, full_w = _yx_shape(series.shape, getattr(series, "axes", None))
            rgb_ds, step_local = _downsample_rgb(rgb, max_dim=max_dim)
            step = max(1, int(np.ceil(max(full_h, full_w) / max(rgb_ds.shape[0], rgb_ds.shape[1]))))
            return rgb_ds, (full_h, full_w, 3), max(step, step_local), "tifffile"
    except Exception:
        rgb_ds, shape, step = _read_rgb_with_openslide(path, max_dim=max_dim)
        return rgb_ds, shape, step, "openslide"


def _read_image_generic(path: str, max_dim: int, fallback_names: List[str]) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        arr_ds, step = _downsample_rgb(arr, max_dim=max_dim)
        return {
            "mode": "rgb",
            "preview_cyx": None,
            "preview_rgb": arr_ds,
            "full_shape": (1, arr.shape[0], arr.shape[1]),
            "step": step,
            "channel_names": ["Image"],
        }

    try:
        with tiff.TiffFile(path) as tf:
            series = _choose_image_series(tf)
            level = _choose_level(series, max_dim=max_dim)
            arr = level.asarray()
            axes = getattr(level, "axes", getattr(series, "axes", None))
            raw_names = _extract_ome_channel_names(tf.ome_metadata)

            try:
                cyx = _array_to_cyx(arr, axes)
                if cyx.ndim == 3 and cyx.shape[0] >= 1:
                    cyx_ds, step_local = _downsample_cyx(cyx, max_dim=max_dim)
                    full_h, full_w = _yx_shape(series.shape, getattr(series, "axes", None))
                    step = max(1, int(np.ceil(max(full_h, full_w) / max(cyx_ds.shape[1], cyx_ds.shape[2]))))

                    names = _resolve_channel_names(raw_names, cyx_ds.shape[0], fallback_names)

                    return {
                        "mode": "channels",
                        "preview_cyx": cyx_ds,
                        "preview_rgb": None,
                        "full_shape": (cyx.shape[0], full_h, full_w),
                        "step": max(step, step_local),
                        "channel_names": names,
                    }
            except Exception:
                pass

            rgb = _array_to_rgb(arr, axes)
            full_h, full_w = _yx_shape(series.shape, getattr(series, "axes", None))
            rgb_ds, step_local = _downsample_rgb(rgb, max_dim=max_dim)
            step = max(1, int(np.ceil(max(full_h, full_w) / max(rgb_ds.shape[0], rgb_ds.shape[1]))))
            return {
                "mode": "rgb",
                "preview_cyx": None,
                "preview_rgb": rgb_ds,
                "full_shape": (1, full_h, full_w),
                "step": max(step, step_local),
                "channel_names": ["Image"],
            }
    except Exception:
        rgb_ds, shape, step = _read_rgb_with_openslide(path, max_dim=max_dim)
        return {
            "mode": "rgb",
            "preview_cyx": None,
            "preview_rgb": rgb_ds,
            "full_shape": (1, shape[0], shape[1]),
            "step": step,
            "channel_names": ["Image"],
        }


def _slot_info(slot: ImageSlotState) -> dict:
    return {
        "loaded": slot.loaded,
        "path": slot.path,
        "mode": slot.mode,
        "shape": list(slot.full_shape or (0, 0, 0)),
        "preview_shape": list(slot.preview_cyx.shape)
        if slot.preview_cyx is not None
        else (list(slot.preview_rgb.shape) if slot.preview_rgb is not None else [0, 0, 0]),
        "step": slot.step,
        "channel_names": slot.channel_names,
    }


def _build_slot(path: str, max_dim: int, fallback_names: List[str]) -> ImageSlotState:
    sig = _read_image_generic(path, max_dim=max_dim, fallback_names=fallback_names)
    return ImageSlotState(
        loaded=True,
        path=path,
        mode=sig["mode"],
        preview_cyx=sig["preview_cyx"],
        preview_rgb=sig["preview_rgb"],
        full_shape=sig["full_shape"],
        step=sig["step"],
        channel_names=sig["channel_names"],
    )


def _load_images(
    he_path: str, image_paths: List[str], max_dim: int, fallback_names: List[str]
) -> dict:
    he_abs = os.path.abspath(os.path.expanduser(he_path))
    if not os.path.exists(he_abs):
        raise FileNotFoundError(f"H&E file does not exist: {he_abs}")

    he_preview, he_shape, he_step, he_source = _read_he_generic(he_abs, max_dim=max_dim)

    slots: Dict[str, ImageSlotState] = {}
    for i, raw in enumerate(image_paths):
        abs_path = os.path.abspath(os.path.expanduser(raw))
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Image {i+1} does not exist: {abs_path}")
        slots[str(i)] = _build_slot(abs_path, max_dim=max_dim, fallback_names=fallback_names)

    STATE.he_preview = he_preview
    STATE.he_shape = he_shape
    STATE.he_step = he_step
    STATE.he_path = he_abs
    STATE.image_slots = slots

    return {
        "he_shape": list(he_shape),
        "he_preview_shape": list(he_preview.shape),
        "he_step": he_step,
        "he_source": he_source,
        "openslide_available": openslide is not None,
        "slot_order": list(slots.keys()),
        "slots": {k: _slot_info(v) for k, v in slots.items()},
    }


def _get_he_png() -> bytes:
    if STATE.he_preview is None:
        raise ValueError("No H&E image loaded.")
    return _to_png_bytes(STATE.he_preview)


def _get_slot(slot: str) -> ImageSlotState:
    s = STATE.image_slots.get(slot)
    if s is None:
        raise ValueError(f"Invalid image slot '{slot}'.")
    if not s.loaded:
        raise ValueError(f"Image slot '{slot}' is not loaded.")
    return s


def _get_slot_base_png(slot: str) -> bytes:
    s = _get_slot(slot)
    if s.base_png_cache is not None:
        return s.base_png_cache

    if s.mode == "rgb":
        if s.preview_rgb is None:
            raise ValueError("Missing RGB preview.")
        s.base_png_cache = _to_png_bytes(s.preview_rgb)
        return s.base_png_cache

    if s.preview_cyx is None or s.preview_cyx.shape[0] == 0:
        raise ValueError("No channels available.")
    base = _normalize_u8(s.preview_cyx[0])
    color = np.array(CHANNEL_COLORS[0], dtype=np.float32)
    rgb = (base[..., None].astype(np.float32) / 255.0 * color).clip(0, 255).astype(np.uint8)
    s.base_png_cache = _to_png_bytes(rgb)
    return s.base_png_cache


def _render_channel_png(slot: str, idx: int) -> bytes:
    s = _get_slot(slot)
    if s.mode != "channels" or s.preview_cyx is None:
        raise ValueError("Channel endpoint is only available for channel-based images.")

    if idx in s.channel_png_cache:
        return s.channel_png_cache[idx]

    base = _normalize_u8(s.preview_cyx[idx])
    color = np.array(CHANNEL_COLORS[idx % len(CHANNEL_COLORS)], dtype=np.float32)
    rgb = (base[..., None].astype(np.float32) / 255.0 * color).clip(0, 255).astype(np.uint8)
    png = _to_png_bytes(rgb)
    s.channel_png_cache[idx] = png
    return png


def _render_overlay_png(slot: str, indices: Sequence[int]) -> bytes:
    s = _get_slot(slot)
    if s.mode == "rgb":
        if s.preview_rgb is None:
            raise ValueError("Missing RGB preview.")
        return _to_png_bytes(s.preview_rgb)

    if s.preview_cyx is None:
        raise ValueError("Missing channel preview.")

    key = tuple(sorted(set(indices)))
    if key in s.overlay_png_cache:
        return s.overlay_png_cache[key]

    y, x = s.preview_cyx.shape[1], s.preview_cyx.shape[2]
    out = np.zeros((y, x, 3), dtype=np.float32)
    for idx in key:
        ch = _normalize_u8(s.preview_cyx[idx]).astype(np.float32) / 255.0
        color = np.array(CHANNEL_COLORS[idx % len(CHANNEL_COLORS)], dtype=np.float32)
        out += ch[..., None] * color
    out = np.clip(out, 0, 255).astype(np.uint8)
    png = _to_png_bytes(out)
    s.overlay_png_cache[key] = png
    return png


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._serve_index()
            return
        if parsed.path == "/api/state":
            self._serve_state()
            return
        if parsed.path == "/api/he.png":
            self._serve_he()
            return
        if parsed.path.startswith("/api/image/") and parsed.path.endswith("/base.png"):
            self._serve_image_base(parsed.path)
            return
        if parsed.path.startswith("/api/image/") and "/channel/" in parsed.path:
            self._serve_image_channel(parsed.path)
            return
        if parsed.path == "/api/image/overlay.png":
            self._serve_image_overlay(parsed.query)
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/load":
            self._load()
            return
        self.send_error(404, "Not found")

    def _serve_index(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        index_path = os.path.join(here, "static", "index.html")
        with open(index_path, "rb") as f:
            body = f.read()

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_state(self) -> None:
        loaded = STATE.he_preview is not None
        payload = {
            "loaded": loaded,
            "he_path": STATE.he_path,
            "openslide_available": openslide is not None,
            "slot_order": list(STATE.image_slots.keys()),
            "slots": {k: _slot_info(v) for k, v in STATE.image_slots.items()},
        }
        _json(self, payload)

    def _serve_he(self) -> None:
        try:
            _png(self, _get_he_png())
        except Exception as exc:
            _json(self, {"error": str(exc)}, code=400)

    def _serve_image_base(self, path: str) -> None:
        try:
            # /api/image/{slot}/base.png
            parts = [p for p in path.split("/") if p]
            slot = parts[2]
            _png(self, _get_slot_base_png(slot))
        except Exception as exc:
            _json(self, {"error": str(exc)}, code=400)

    def _serve_image_channel(self, path: str) -> None:
        try:
            # /api/image/{slot}/channel/{idx}
            parts = [p for p in path.split("/") if p]
            slot = parts[2]
            idx = int(parts[4])
            s = _get_slot(slot)
            if s.preview_cyx is None:
                raise ValueError("This image has no channel stack.")
            if idx < 0 or idx >= s.preview_cyx.shape[0]:
                raise IndexError("Channel index out of range")
            _png(self, _render_channel_png(slot, idx))
        except Exception as exc:
            _json(self, {"error": str(exc)}, code=400)

    def _serve_image_overlay(self, query: str) -> None:
        try:
            params = parse_qs(query)
            slot = params.get("slot", [""])[0]
            if slot == "":
                raise ValueError("slot query parameter is required")
            s = _get_slot(slot)

            if s.mode == "rgb":
                _png(self, _render_overlay_png(slot, []))
                return

            raw = params.get("channels", [""])[0]
            if not raw.strip():
                _png(self, _get_slot_base_png(slot))
                return

            indices = [int(v) for v in raw.split(",") if v.strip()]
            max_idx = s.preview_cyx.shape[0] - 1 if s.preview_cyx is not None else -1
            for idx in indices:
                if idx < 0 or idx > max_idx:
                    raise IndexError("Channel index out of range")
            _png(self, _render_overlay_png(slot, indices))
        except Exception as exc:
            _json(self, {"error": str(exc)}, code=400)

    def _load(self) -> None:
        try:
            body = _read_json_body(self)
            he_path = str(body.get("he_path", "")).strip()
            max_dim = int(body.get("max_dim", 1800))
            max_dim = max(512, min(4096, max_dim))

            image_paths = body.get("image_paths")
            if not isinstance(image_paths, list):
                image_paths = []

            # backward compatibility
            if not image_paths:
                a = str(body.get("signal_a_path", "")).strip()
                b = str(body.get("signal_b_path", "")).strip()
                image_paths = [p for p in [a, b] if p]

            image_paths = [str(p).strip() for p in image_paths if str(p).strip()]
            fallback_channel_names = body.get("fallback_channel_names")
            if not isinstance(fallback_channel_names, list):
                fallback_channel_names = []
            fallback_channel_names = [str(v).strip() for v in fallback_channel_names if str(v).strip()]

            if not he_path:
                raise ValueError("he_path is required")

            info = _load_images(
                he_path, image_paths, max_dim=max_dim, fallback_names=fallback_channel_names
            )
            _json(self, {"ok": True, "info": info})
        except Exception as exc:
            _json(self, {"ok": False, "error": str(exc)}, code=400)

    def log_message(self, fmt: str, *args) -> None:
        return


def run_server(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Viewer running at http://{host}:{port}")
    print(f"OpenSlide available: {'yes' if openslide is not None else 'no'}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TME-viewer")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_server(args.host, args.port)
