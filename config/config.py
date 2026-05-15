# config/config.py
"""Central runtime configuration for the vehicle detection service.

Edit this file when you want readable local defaults.
Environment variables can still override every value.
"""

from dataclasses import dataclass
import os
from typing import ClassVar


def _validate_roi_tuple_impl(vals: tuple) -> tuple:
    if len(vals) != 4:
        raise ValueError(f"Expected 4 ROI values, got {len(vals)}")
    x1, y1, x2, y2 = tuple(float(v) for v in vals)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"ROI values out of range: {(x1, y1, x2, y2)}")
    return (x1, y1, x2, y2)


@dataclass
class Config:
    """Environment-backed runtime settings loaded at import time."""

    # -------------------------------------------------------------------------
    # Parsing helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _parse_bool(val: str, default: bool) -> bool:
        if val is None:
            return default
        return str(val).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _parse_csv_floats(val: str, default: tuple, expected_len: int) -> tuple:
        if not val:
            return default
        try:
            parts = [float(x.strip()) for x in str(val).split(",")]
            if len(parts) != expected_len:
                raise ValueError(f"Expected {expected_len} values, got {len(parts)}")
            return tuple(parts)
        except (ValueError, TypeError) as e:
            print(f"Error parsing CSV floats: {e}. Using default: {default}")
            return default

    @staticmethod
    def _parse_csv_ints(val: str, default: tuple, expected_len=None) -> tuple:
        if val is None:
            return default
        if not str(val).strip():
            return default
        try:
            parts = tuple(int(x.strip()) for x in str(val).split(",") if x.strip())
            if expected_len is not None and len(parts) != expected_len:
                raise ValueError(f"Expected {expected_len} values, got {len(parts)}")
            return parts
        except (ValueError, TypeError) as e:
            print(f"Error parsing CSV ints: {e}. Using default: {default}")
            return default

    @staticmethod
    def _parse_alias(val: str, default: str, alias_map: dict) -> str:
        if val is None:
            return default
        text = str(val).strip().lower()
        if not text:
            return default
        resolved = alias_map.get(text)
        if resolved is None:
            print(f"Error parsing alias '{text}'. Using default: {default}")
            return default
        return resolved

    @staticmethod
    def _validate_roi_tuple(vals: tuple) -> tuple:
        return _validate_roi_tuple_impl(vals)

    @staticmethod
    def _parse_optional_roi(val: str, default: tuple = ()) -> tuple:
        if val is None:
            return default
        text = str(val).strip()
        if not text:
            return default
        if text.lower() in {"0", "false", "off", "none", "disabled"}:
            return ()
        try:
            vals = tuple(float(x.strip()) for x in text.split(","))
            return _validate_roi_tuple_impl(vals)
        except (ValueError, TypeError) as e:
            print(f"Error parsing ROI '{text}': {e}. Using default: {default}")
            return default

    @staticmethod
    def _parse_roi_list(val: str, default: tuple = ()) -> tuple:
        if val is None:
            return default
        rois = []
        for part in str(val).split(";"):
            part = part.strip()
            if not part:
                continue
            try:
                vals = tuple(float(v.strip()) for v in part.split(","))
                rois.append(_validate_roi_tuple_impl(vals))
            except (ValueError, TypeError) as e:
                print(f"Error parsing ROI '{part}': {e}. Skipping.")
        return tuple(rois) if rois else default

    @staticmethod
    def _parse_count_zone_list(val: str, default: tuple = ()) -> tuple:
        if val is None:
            return default
        zones = []
        for part in str(val).split(";"):
            part = part.strip()
            if not part:
                continue
            try:
                if "|" in part:
                    points = []
                    for point_text in part.split("|"):
                        point_text = point_text.strip()
                        if not point_text:
                            continue
                        coords = tuple(float(v.strip()) for v in point_text.split(","))
                        if len(coords) != 2:
                            raise ValueError(f"Expected point as x,y, got {coords}")
                        x, y = coords
                        if not (0 <= x <= 1 and 0 <= y <= 1):
                            raise ValueError(f"Point out of range: {coords}")
                        points.append(coords)
                    if len(points) < 3:
                        raise ValueError(f"Polygon needs at least 3 points, got {len(points)}")
                    zones.append(tuple(points))
                    continue

                vals = tuple(float(v.strip()) for v in part.split(","))
                if len(vals) != 4:
                    raise ValueError(f"Expected 4 rectangle values, got {len(vals)}")
                x1, y1, x2, y2 = vals
                if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                    raise ValueError(f"Rectangle values out of range: {vals}")
                zones.append(((x1, y1), (x2, y1), (x2, y2), (x1, y2)))
            except (ValueError, TypeError) as e:
                print(f"Error parsing count zone '{part}': {e}. Skipping.")
        return tuple(zones) if zones else default

    @staticmethod
    def _parse_zone_class_map(val: str, default: dict | None = None) -> dict:
        if default is None:
            default = {}
        if val is None:
            return dict(default)
        text = str(val).strip()
        if not text:
            return dict(default)

        zone_map = {}
        for part in text.split(";"):
            part = part.strip()
            if not part:
                continue
            try:
                zone_text, class_text = part.split(":", 1)
                zone_id = int(zone_text.strip())
                if zone_id <= 0:
                    raise ValueError(f"Zone id must be positive, got {zone_id}")
                class_ids = tuple(int(x.strip()) for x in class_text.split(",") if x.strip())
                if not class_ids:
                    raise ValueError("Class id list cannot be empty")
                zone_map[zone_id] = class_ids
            except (ValueError, TypeError) as e:
                print(f"Error parsing zone class map '{part}': {e}. Skipping.")
        return zone_map if zone_map else dict(default)

    # -------------------------------------------------------------------------
    # Camera and stream
    # -------------------------------------------------------------------------
    _DEFAULT_RTSP_URL = "rtsp://localhost:8554/cam21"
    _DEFAULT_STREAM_PORT = 8005
    _DEFAULT_RTSP_FPS = "20/1"

    RTSP_URL: str = os.getenv("RTSP_URL", _DEFAULT_RTSP_URL)
    TRACK_RTSP_URL: str = os.getenv("TRACK_RTSP_URL", RTSP_URL)
    RECONNECT_ATTEMPTS: int = int(os.getenv("RECONNECT_ATTEMPTS", "10"))
    RECONNECT_DELAY: int = int(os.getenv("RECONNECT_DELAY", "5"))
    STREAM_PORT: int = int(os.getenv("STREAM_PORT", str(_DEFAULT_STREAM_PORT)))

    # -------------------------------------------------------------------------
    # Decoder
    # -------------------------------------------------------------------------
    DECODER_BACKEND: str = os.getenv("DECODER_BACKEND", "vaapi")
    DECODER_STRICT_GPU: bool = _parse_bool.__func__(os.getenv("DECODER_STRICT_GPU"), False)
    LIBVA_DRIVER_NAME: str = os.getenv("LIBVA_DRIVER_NAME", "i965")
    FFMPEG_BINARY: str = os.getenv("FFMPEG_BINARY", "ffmpeg")
    FFPROBE_BINARY: str = os.getenv("FFPROBE_BINARY", "ffprobe")
    FFMPEG_HWACCEL: str = os.getenv("FFMPEG_HWACCEL", "cuda")
    FFMPEG_HWACCEL_OUTPUT_FORMAT: str = os.getenv("FFMPEG_HWACCEL_OUTPUT_FORMAT", "cuda")
    FFMPEG_VIDEO_CODEC: str = os.getenv("FFMPEG_VIDEO_CODEC", "").strip()
    FFMPEG_LOGLEVEL: str = os.getenv("FFMPEG_LOGLEVEL", "warning")
    FFMPEG_FRAME_WIDTH: int = int(os.getenv("FFMPEG_FRAME_WIDTH", "0"))
    FFMPEG_FRAME_HEIGHT: int = int(os.getenv("FFMPEG_FRAME_HEIGHT", "0"))
    FFMPEG_READ_TIMEOUT_S: float = float(os.getenv("FFMPEG_READ_TIMEOUT_S", "3.0"))
    RTSP_PROTOCOLS: str = os.getenv("RTSP_PROTOCOLS", "tcp")
    RTSP_LATENCY_MS: int = int(os.getenv("RTSP_LATENCY_MS", "200"))
    RTSP_FPS: str = os.getenv("RTSP_FPS", _DEFAULT_RTSP_FPS)
    TRACK_RTSP_FPS: str = os.getenv("TRACK_RTSP_FPS", RTSP_FPS)
    STREAM_HEALTHCHECK_INTERVAL_S: float = float(os.getenv("STREAM_HEALTHCHECK_INTERVAL_S", "1.0"))

    # -------------------------------------------------------------------------
    # Detection thresholds
    # -------------------------------------------------------------------------
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    NMS_THRESHOLD: float = float(os.getenv("NMS_THRESHOLD", "0.4"))
    MERGE_NMS_THRESHOLD: float = float(os.getenv("MERGE_NMS_THRESHOLD", str(NMS_THRESHOLD)))
    PIXEL_CHANGE_THRESHOLD: float = float(os.getenv("PIXEL_CHANGE_THRESHOLD", "0.01"))
    DIFF_THRESHOLD: int = int(os.getenv("DIFF_THRESHOLD", "20"))

    # -------------------------------------------------------------------------
    # Detection regions
    # -------------------------------------------------------------------------
    _DEFAULT_DETECTION_ROI = (0.0, 0.0, 1.0, 1.0)
    _DETECTION_ROI_RAW = os.getenv("DETECTION_ROI", "0, 0, 1, 1")
    try:
        DETECTION_ROI: tuple = _validate_roi_tuple.__func__(
            tuple(float(x.strip()) for x in _DETECTION_ROI_RAW.split(","))
        )
    except (ValueError, TypeError) as e:
        print(f"Error parsing DETECTION_ROI: {e}. Using default: {_DEFAULT_DETECTION_ROI}")
        DETECTION_ROI: tuple = _DEFAULT_DETECTION_ROI

    # Second pass focused on the far part of the scene for small/distant vehicles.
    ENABLE_FAR_DETECTION: bool = _parse_bool.__func__(os.getenv("ENABLE_FAR_DETECTION"), True)
    _DEFAULT_FAR_DETECTION_ROI = (
        DETECTION_ROI[0],
        DETECTION_ROI[1],
        DETECTION_ROI[2],
        DETECTION_ROI[1] + ((DETECTION_ROI[3] - DETECTION_ROI[1]) * 0.55),
    )
    FAR_DETECTION_ROI: tuple = _parse_optional_roi.__func__(
        os.getenv("FAR_DETECTION_ROI"),
        _DEFAULT_FAR_DETECTION_ROI if ENABLE_FAR_DETECTION else (),
    )

    # Counting polygons in normalized coordinates. Format:
    # "x1,y1|x2,y2|x3,y3;..."
    _DEFAULT_COUNT_ZONES = (
        "0.0019,0.2876|0.1647,0.2635|0.1604,0.3539|0.0000,0.3730"
        ";0.9265,0.5572|0.9963,0.6511|0.9897,0.7149|0.9604,0.6462"
        ";0.0209,0.8014|0.1226,0.5065|0.3508,0.5053|0.3201,0.8498"
        ";0.9148,0.3518|0.8212,0.3648|0.7628,0.2664|0.9131,0.2986"
        ";0.4522,0.1027|0.4524,0.1615|0.5730,0.1574|0.5456,0.1075"
    )
    _COUNT_ROIS_SOURCE = os.getenv("COUNT_ROIS", os.getenv("TRIGGER_ROIS", _DEFAULT_COUNT_ZONES))
    _COUNT_ZONES_SOURCE = os.getenv(
        "COUNT_ZONES",
        os.getenv("COUNT_POLYGONS", _COUNT_ROIS_SOURCE),
    )
    COUNT_ROIS: tuple = _parse_roi_list.__func__(
        _COUNT_ROIS_SOURCE if _COUNT_ROIS_SOURCE and "|" not in _COUNT_ROIS_SOURCE else None,
        (),
    )
    COUNT_ZONES: tuple = _parse_count_zone_list.__func__(_COUNT_ZONES_SOURCE, ())
    # Optional per-zone class filters. Format: "1:2,3;2:0,2;..."
    # Zone ids are 1-based and map to COUNT_ZONES order.
    COUNT_ZONE_CLASS_IDS: ClassVar[dict] = _parse_zone_class_map.__func__(
        os.getenv("COUNT_ZONE_CLASS_IDS"),
        {},
    )

    # -------------------------------------------------------------------------
    # Tracking
    # -------------------------------------------------------------------------
    _TRACKER_BACKEND_ALIASES = {
        "custom": "custom",
        "default": "custom",
        "simple": "custom",
        "local": "custom",
        "bytetrack": "bytetrack",
        "byte_track": "bytetrack",
        "byte-track": "bytetrack",
        "botsort": "botsort",
        "bot_sort": "botsort",
        "bot-sort": "botsort",
    }
    TRACKER_BACKEND: str = _parse_alias.__func__(
        os.getenv("TRACKER_BACKEND"),
        "custom",
        _TRACKER_BACKEND_ALIASES,
    )
    MAX_DISTANCE: int = int(os.getenv("MAX_DISTANCE", "800"))
    MAX_AGE: int = int(os.getenv("MAX_AGE", "10"))
    MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.0"))
    TRACKER_DEBUG: bool = _parse_bool.__func__(os.getenv("TRACKER_DEBUG"), False)
    TRACK_ID_PREFIX: str = os.getenv("TRACK_ID_PREFIX", "object")
    TRACKER_HIGH_THRESH: float = float(os.getenv("TRACKER_HIGH_THRESH", "0.25"))
    TRACKER_LOW_THRESH: float = float(os.getenv("TRACKER_LOW_THRESH", "0.10"))
    TRACKER_NEW_TRACK_THRESH: float = float(os.getenv("TRACKER_NEW_TRACK_THRESH", "0.25"))
    TRACKER_BUFFER: int = int(os.getenv("TRACKER_BUFFER", "30"))
    TRACKER_MATCH_THRESH: float = float(os.getenv("TRACKER_MATCH_THRESH", "0.80"))
    TRACKER_FUSE_SCORE: bool = _parse_bool.__func__(os.getenv("TRACKER_FUSE_SCORE"), True)
    TRACKER_GMC_METHOD: str = os.getenv("TRACKER_GMC_METHOD", "sparseOptFlow").strip()
    TRACKER_WITH_REID: bool = _parse_bool.__func__(os.getenv("TRACKER_WITH_REID"), False)
    TRACKER_PROXIMITY_THRESH: float = float(os.getenv("TRACKER_PROXIMITY_THRESH", "0.50"))
    TRACKER_APPEARANCE_THRESH: float = float(os.getenv("TRACKER_APPEARANCE_THRESH", "0.80"))
    TRACKER_REID_MODEL: str = os.getenv("TRACKER_REID_MODEL", "").strip()

    # -------------------------------------------------------------------------
    # Backend event API
    # -------------------------------------------------------------------------
    _COUNTER_RECORD_API_URL_FROM_LEGACY = not bool(os.getenv("COUNTER_RECORD_API_URL"))
    _COUNTER_RECORD_API_USERNAME_FROM_LEGACY = not bool(os.getenv("COUNTER_RECORD_API_USERNAME"))
    _COUNTER_RECORD_API_PASSWORD_FROM_LEGACY = not bool(os.getenv("COUNTER_RECORD_API_PASSWORD")) 
    DEVICE_ID: str = os.getenv("DEVICE_ID", os.getenv("DEVICE", "")).strip()
    CONTAINER_ID: str = os.getenv(
        "CONTAINER_ID",
        f"counter_cctv_{DEVICE_ID}" if DEVICE_ID else "",
    ).strip()
    COUNTER_RECORD_API_URL: str = os.getenv("COUNTER_RECORD_API_URL", "").strip()
    COUNTER_RECORD_API_USERNAME: str = os.getenv("COUNTER_RECORD_API_USERNAME", "").strip()
    COUNTER_RECORD_API_PASSWORD: str = os.getenv("COUNTER_RECORD_API_PASSWORD", "").strip()
    COUNTER_RECORD_API_TIMEOUT_S: float = float(os.getenv("COUNTER_RECORD_API_TIMEOUT_S", "10.0"))
    COUNTER_RECORD_API_FLUSH_INTERVAL_S: float = float(os.getenv("COUNTER_RECORD_API_FLUSH_INTERVAL_S", "0.5"))
    COUNTER_RECORD_API_BATCH_SIZE: int = int(os.getenv("COUNTER_RECORD_API_BATCH_SIZE", "50"))
    COUNTER_RECORD_API_QUEUE_SIZE: int = int(os.getenv("COUNTER_RECORD_API_QUEUE_SIZE", "1000"))
    EVENT_SEVERITY: str = os.getenv("EVENT_SEVERITY", "high").strip()

    # -------------------------------------------------------------------------
    # File upload API
    # -------------------------------------------------------------------------
    FILE_API_BASE_URL: str = os.getenv("FILE_API_BASE_URL", "http://localhost:5001").strip().rstrip("/")
    FILE_API_USERNAME: str = os.getenv("FILE_API_USERNAME", COUNTER_RECORD_API_USERNAME).strip()
    FILE_API_PASSWORD: str = os.getenv("FILE_API_PASSWORD", COUNTER_RECORD_API_PASSWORD).strip()
    FILE_API_TIMEOUT_S: float = float(os.getenv("FILE_API_TIMEOUT_S", "30.0"))
    FILE_IMAGE_UPLOAD_URL: str = os.getenv(
        "FILE_IMAGE_UPLOAD_URL",
        f"{FILE_API_BASE_URL}/file/fs/{{device_id}}",
    ).strip()

    # -------------------------------------------------------------------------
    # Entry media capture
    # -------------------------------------------------------------------------
    ENABLE_ENTRY_MEDIA_CAPTURE: bool = _parse_bool.__func__(os.getenv("ENABLE_ENTRY_MEDIA_CAPTURE"), False)
    ENTRY_MEDIA_DIR: str = os.getenv("ENTRY_MEDIA_DIR", "entry_media").strip()

    # -------------------------------------------------------------------------
    # Processing and preview
    # -------------------------------------------------------------------------
    SKIP_FRAMES: int = int(os.getenv("SKIP_FRAMES", "1"))

    # Historical compatibility: this project currently enables debug draw only
    # when the env value is literally "False".
    DEBUG_DRAW: bool = os.getenv("DEBUG_DRAW", "True") == "False"

    MAX_FPS: int = int(os.getenv("MAX_FPS", "20"))
    PREVIEW_FPS: float = float(os.getenv("PREVIEW_FPS", "20.0"))
    PREVIEW_JPEG_QUALITY: int = int(os.getenv("PREVIEW_JPEG_QUALITY", "99"))
    MOTION_CHECK_INTERVAL_S: float = float(os.getenv("MOTION_CHECK_INTERVAL_S", "0.15"))
    MOTION_DOWNSCALE: float = float(os.getenv("MOTION_DOWNSCALE", "0.5"))
    BBOX_LINE_THICKNESS: int = int(os.getenv("BBOX_LINE_THICKNESS", "2"))
    ENABLE_LOCAL_KEYBOARD_EXIT: bool = _parse_bool.__func__(os.getenv("ENABLE_LOCAL_KEYBOARD_EXIT"), False)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    _DEFAULT_MODEL_INPUT_SIZE = (640, 640)
    _DEFAULT_DETECTION_CLASS_IDS = (0, 2, 3, 5, 7)
    _TORCH_MODEL_VARIANT_ALIASES = {
        "n": "nano",
        "nano": "nano",
        "s": "small",
        "small": "small",
        "m": "medium",
        "medium": "medium",
        "l": "large",
        "large": "large",
    }
    _TORCH_MODEL_PATHS = {
        "nano": "models/yolov8n.pt",
        "small": "models/yolov8s.pt",
        "medium": "models/yolov8m.pt",
        "large": "models/yolov8l.pt",
    }

    MODEL_BACKEND: str = os.getenv("MODEL_BACKEND", "openvino").strip().lower()
    PERFORMANCE_HINT: str = os.getenv("PERFORMANCE_HINT", "LATENCY")
    DETECTION_MODEL_PATH: str = os.getenv(
        "DETECTION_MODEL_PATH",
        "models/yolov8s_openvino_model/yolov8s.xml",
    )
    # Torch model presets for backend dropdowns. Supported values:
    # n|nano, s|small, m|medium, l|large
    TORCH_MODEL_VARIANT: str = _parse_alias.__func__(
        os.getenv("TORCH_MODEL_VARIANT"),
        "small",
        _TORCH_MODEL_VARIANT_ALIASES,
    )
    TORCH_MODEL_PATH: str = os.getenv("TORCH_MODEL_PATH", _TORCH_MODEL_PATHS[TORCH_MODEL_VARIANT])
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "CPU")
    TORCH_HALF: bool = _parse_bool.__func__(os.getenv("TORCH_HALF"), False)
    MODEL_INPUT_SIZE: tuple = _parse_csv_ints.__func__(
        os.getenv("MODEL_INPUT_SIZE"),
        _DEFAULT_MODEL_INPUT_SIZE,
        2,
    )
    DETECTION_OUTPUT_FORMAT: str = os.getenv(
        "DETECTION_OUTPUT_FORMAT",
        "raw_class_scores",
    ).strip().lower()
    DETECTION_CLASS_IDS: tuple = _parse_csv_ints.__func__(
        os.getenv("DETECTION_CLASS_IDS"),
        _DEFAULT_DETECTION_CLASS_IDS,
        None,
    )
    USE_MODEL_NORMALIZATION: bool = _parse_bool.__func__(os.getenv("USE_MODEL_NORMALIZATION"), False)
    MODEL_MEAN: tuple = _parse_csv_floats.__func__(
        os.getenv("MODEL_MEAN"),
        (123.675, 116.28, 103.53),
        3,
    )
    MODEL_STD: tuple = _parse_csv_floats.__func__(
        os.getenv("MODEL_STD"),
        (58.395, 57.12, 57.375),
        3,
    )
