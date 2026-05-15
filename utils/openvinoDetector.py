# utils/openvinoDetector.py
from pathlib import Path
import inspect
import logging
import os
import re
import select
import shutil
import subprocess
import tempfile
import threading
import time
from queue import Empty, Queue

import cv2
import numpy as np

try:
    import openvino as ov
except Exception:
    ov = None

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

BOTSORT = None
BYTETracker = None
IterableSimpleNamespace = None

from config.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_OPENCV_RTSP_TRANSPORT = str(getattr(Config, "RTSP_PROTOCOLS", "tcp") or "tcp").split(",", 1)[0].strip().lower()
if _OPENCV_RTSP_TRANSPORT not in {"tcp", "udp"}:
    _OPENCV_RTSP_TRANSPORT = "tcp"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    f"rtsp_transport;{_OPENCV_RTSP_TRANSPORT}|"
    "stimeout;3000000|rw_timeout;3000000|"
    "probesize;1000000|analyzeduration;1000000|"
    "fflags;nobuffer|flags;low_delay|threads;1"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_BACKEND = getattr(Config, "MODEL_BACKEND", "openvino")
TRACKER_BACKEND = getattr(Config, "TRACKER_BACKEND", "custom")
CONFIDENCE_THRESHOLD = getattr(Config, "CONFIDENCE_THRESHOLD", 0.5)
NMS_THRESHOLD = getattr(Config, "NMS_THRESHOLD", 0.4)
MERGE_NMS_THRESHOLD = getattr(Config, "MERGE_NMS_THRESHOLD", NMS_THRESHOLD)
USE_MODEL_NORMALIZATION = getattr(Config, "USE_MODEL_NORMALIZATION", True)
MODEL_MEAN = np.array(getattr(Config, "MODEL_MEAN", (123.675, 116.28, 103.53)), dtype=np.float32)
MODEL_STD = np.array(getattr(Config, "MODEL_STD", (58.395, 57.12, 57.375)), dtype=np.float32)
DEFAULT_INPUT_SIZE = tuple(getattr(Config, "MODEL_INPUT_SIZE", (416, 416)))
DETECTION_OUTPUT_FORMAT = getattr(Config, "DETECTION_OUTPUT_FORMAT", "single_class")
DETECTION_CLASS_IDS = set(getattr(Config, "DETECTION_CLASS_IDS", ()))
ENABLE_FAR_DETECTION = getattr(Config, "ENABLE_FAR_DETECTION", True)
FAR_DETECTION_ROI = tuple(getattr(Config, "FAR_DETECTION_ROI", ()))
TORCH_MODEL_PATH = getattr(Config, "TORCH_MODEL_PATH", "models/yolov8s.pt")
TORCH_HALF = getattr(Config, "TORCH_HALF", False)
TRACK_ID_PREFIX = getattr(Config, "TRACK_ID_PREFIX", "vehicle")
BBOX_LINE_THICKNESS = max(1, int(getattr(Config, "BBOX_LINE_THICKNESS", 2)))
BBOX_COLOR = (238, 238, 175)  # #afeeee in BGR
DETECTION_CLASS_LABELS = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    4: "Airplane",
    5: "Bus",
    6: "Train",
    7: "Truck",
    8: "Boat",
    9: "Traffic Light",
    10: "Fire Hydrant",
    11: "Stop Sign",
    12: "Parking Meter",
    13: "Bench",
    14: "Bird",
    15: "Cat",
    16: "Dog",
    17: "Horse",
    18: "Sheep",
    19: "Cow",
    20: "Elephant",
    21: "Bear",
    22: "Zebra",
    23: "Giraffe",
    24: "Backpack",
    25: "Umbrella",
    26: "Handbag",
    27: "Tie",
    28: "Suitcase",
    29: "Frisbee",
    30: "Skis",
    31: "Snowboard",
    32: "Sports Ball",
    33: "Kite",
    34: "Baseball Bat",
    35: "Baseball Glove",
    36: "Skateboard",
    37: "Surfboard",
    38: "Tennis Racket",
    39: "Bottle",
    40: "Wine Glass",
    41: "Cup",
    42: "Fork",
    43: "Knife",
    44: "Spoon",
    45: "Bowl",
    46: "Banana",
    47: "Apple",
    48: "Sandwich",
    49: "Orange",
    50: "Broccoli",
    51: "Carrot",
    52: "Hot Dog",
    53: "Pizza",
    54: "Donut",
    55: "Cake",
    56: "Chair",
    57: "Couch",
    58: "Potted Plant",
    59: "Bed",
    60: "Dining Table",
    61: "Toilet",
    62: "TV",
    63: "Laptop",
    64: "Mouse",
    65: "Remote",
    66: "Keyboard",
    67: "Cell Phone",
    68: "Microwave",
    69: "Oven",
    70: "Toaster",
    71: "Sink",
    72: "Refrigerator",
    73: "Book",
    74: "Clock",
    75: "Vase",
    76: "Scissors",
    77: "Teddy Bear",
    78: "Hair Drier",
    79: "Toothbrush",
}


def _display_track_id(track_id) -> str:
    text = str(track_id or "").strip()
    if not text:
        return "#?"
    prefix = f"{TRACK_ID_PREFIX}_"
    if TRACK_ID_PREFIX and text.startswith(prefix):
        text = text[len(prefix):]
    return f"#{text}"

_MODEL_CONTEXT = None
_MODEL_LOCK = threading.Lock()
_POSTPROCESS_LOGGED = False
_GST_ELEMENT_CACHE = {}
_OPENCV_BACKEND_CACHE = {}


def _parse_configured_fps(value, default: float = 30.0) -> float:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_val = float(denominator.strip())
            if denominator_val == 0:
                return default
            return max(float(numerator.strip()) / denominator_val, 1.0)
        return max(float(text), 1.0)
    except Exception:
        return default


def _tracker_frame_rate() -> int:
    source_fps = _parse_configured_fps(getattr(Config, "TRACK_RTSP_FPS", "30/1"), default=30.0)
    skip_frames = max(1, int(getattr(Config, "SKIP_FRAMES", 1)))
    effective_fps = source_fps / skip_frames
    return max(1, int(round(effective_fps)))


def _load_ultralytics_tracker_runtime():
    global BOTSORT, BYTETracker, IterableSimpleNamespace
    if BOTSORT is not None and BYTETracker is not None and IterableSimpleNamespace is not None:
        return BOTSORT, BYTETracker, IterableSimpleNamespace
    try:
        from ultralytics.trackers.bot_sort import BOTSORT as _BOTSORT
        from ultralytics.trackers.byte_tracker import BYTETracker as _BYTETracker
        from ultralytics.utils import IterableSimpleNamespace as _IterableSimpleNamespace
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics tracking runtime is unavailable. Install ultralytics + torch + lap "
            "or use the NVIDIA image."
        ) from exc
    BOTSORT = _BOTSORT
    BYTETracker = _BYTETracker
    IterableSimpleNamespace = _IterableSimpleNamespace
    return BOTSORT, BYTETracker, IterableSimpleNamespace


class _TrackerDetections:
    def __init__(self, xyxy=None, conf=None, cls=None):
        if xyxy is None:
            self.xyxy = np.empty((0, 4), dtype=np.float32)
        else:
            self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        if conf is None:
            self.conf = np.empty((len(self.xyxy),), dtype=np.float32)
        else:
            self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        if cls is None:
            self.cls = np.empty((len(self.xyxy),), dtype=np.float32)
        else:
            self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __len__(self):
        return len(self.xyxy)

    def __getitem__(self, idx):
        return _TrackerDetections(self.xyxy[idx], self.conf[idx], self.cls[idx])

    @property
    def xywh(self):
        if len(self.xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)
        xywh = self.xyxy.copy()
        xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
        xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
        xywh[:, 0] = xywh[:, 0] + (xywh[:, 2] / 2.0)
        xywh[:, 1] = xywh[:, 1] + (xywh[:, 3] / 2.0)
        return xywh


class FrameBuffer:
    def __init__(self, maxsize=1):
        self.queue = Queue(maxsize=maxsize)

    def put(self, frame):
        if not self.queue.full():
            self.queue.put(frame)
            return
        try:
            self.queue.get_nowait()
            self.queue.put(frame)
        except Exception:
            pass

    def get(self, timeout=None):
        try:
            if timeout is None:
                return self.queue.get_nowait()
            return self.queue.get(timeout=timeout)
        except Empty:
            return None


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _dimension_to_int(dim):
    try:
        value = int(dim)
        return value if value > 0 else None
    except Exception:
        return None


def _infer_model_input_size(compiled_model, fallback):
    try:
        partial_shape = compiled_model.input(0).get_partial_shape()
        height = _dimension_to_int(partial_shape[-2])
        width = _dimension_to_int(partial_shape[-1])
        if height and width:
            return (height, width)
    except Exception:
        pass
    return tuple(fallback)


def _normalize_model_backend(name: str) -> str:
    return (str(name or "openvino").strip().lower())


def _normalize_torch_device(device: str) -> str:
    raw = str(device or "cpu").strip()
    upper = raw.upper()
    if upper == "CPU":
        return "cpu"
    if upper == "GPU":
        return "cuda:0"
    if upper == "CUDA":
        return "cuda:0"
    return raw


def _load_openvino_model_context():
    if ov is None:
        raise RuntimeError("OpenVINO is not installed, but MODEL_BACKEND=openvino was requested")

    model_path = _resolve_model_path(getattr(Config, "DETECTION_MODEL_PATH", "models/20251020_094033/exported_model.xml"))
    if not model_path.exists():
        raise FileNotFoundError(f"Detection model not found: {model_path}")

    core = ov.Core()
    model = core.read_model(str(model_path))
    device = getattr(Config, "MODEL_DEVICE", "CPU")
    openvino_config = {"PERFORMANCE_HINT": getattr(Config, "PERFORMANCE_HINT", "LATENCY")}
    compiled_model = core.compile_model(model, device, openvino_config)
    input_size = _infer_model_input_size(compiled_model, DEFAULT_INPUT_SIZE)

    try:
        input_shape = compiled_model.input(0).get_shape()
    except Exception:
        input_shape = compiled_model.input(0).get_partial_shape()
    try:
        output_shape = compiled_model.output(0).get_partial_shape()
    except Exception:
        output_shape = "unknown"

    logging.info(
        "Loaded detection model: backend=%s path=%s device=%s input=%s output=%s format=%s class_ids=%s",
        "openvino",
        model_path,
        device,
        input_shape,
        output_shape,
        DETECTION_OUTPUT_FORMAT,
        sorted(DETECTION_CLASS_IDS) if DETECTION_CLASS_IDS else "all",
    )

    return {
        "backend": "openvino",
        "compiled_model": compiled_model,
        "input_size": input_size,
        "output_format": DETECTION_OUTPUT_FORMAT,
        "allowed_class_ids": DETECTION_CLASS_IDS,
    }


def _load_torch_model_context():
    if YOLO is None or torch is None:
        raise RuntimeError("PyTorch/Ultralytics is not installed, but MODEL_BACKEND=torch was requested")

    model_path = _resolve_model_path(TORCH_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Torch detection model not found: {model_path}")

    requested_device = _normalize_torch_device(getattr(Config, "MODEL_DEVICE", "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"MODEL_BACKEND=torch requested CUDA device '{requested_device}', but torch.cuda.is_available() is False"
        )

    model = YOLO(str(model_path))
    try:
        model.to(requested_device)
    except Exception:
        # Ultralytics also accepts device during predict; keep load resilient.
        logging.warning("Ultralytics model.to(%s) failed during load; will rely on predict(device=...) instead.", requested_device)

    half_enabled = bool(TORCH_HALF and requested_device.startswith("cuda"))
    logging.info(
        "Loaded detection model: backend=%s path=%s device=%s input=%s half=%s class_ids=%s",
        "torch",
        model_path,
        requested_device,
        DEFAULT_INPUT_SIZE,
        half_enabled,
        sorted(DETECTION_CLASS_IDS) if DETECTION_CLASS_IDS else "all",
    )

    return {
        "backend": "torch",
        "yolo_model": model,
        "device": requested_device,
        "input_size": DEFAULT_INPUT_SIZE,
        "allowed_class_ids": DETECTION_CLASS_IDS,
        "half": half_enabled,
    }


def _load_model_context():
    backend = _normalize_model_backend(MODEL_BACKEND)
    if backend == "openvino":
        return _load_openvino_model_context()
    if backend in {"torch", "pytorch", "ultralytics"}:
        return _load_torch_model_context()
    raise ValueError(f"Unsupported MODEL_BACKEND: {backend}")


def get_model_context():
    global _MODEL_CONTEXT
    if _MODEL_CONTEXT is not None:
        return _MODEL_CONTEXT
    with _MODEL_LOCK:
        if _MODEL_CONTEXT is None:
            _MODEL_CONTEXT = _load_model_context()
    return _MODEL_CONTEXT


def _gst_element_available(name: str) -> bool:
    cached = _GST_ELEMENT_CACHE.get(name)
    if cached is not None:
        return cached
    gst_inspect = shutil.which("gst-inspect-1.0")
    if gst_inspect is None:
        _GST_ELEMENT_CACHE[name] = False
        return False
    try:
        result = subprocess.run(
            [gst_inspect, name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        available = result.returncode == 0
    except Exception:
        available = False
    _GST_ELEMENT_CACHE[name] = available
    return available


def _opencv_backend_available(name: str) -> bool:
    cached = _OPENCV_BACKEND_CACHE.get(name)
    if cached is not None:
        return cached

    marker = f"{name}:"
    available = False
    try:
        for line in cv2.getBuildInformation().splitlines():
            stripped = line.strip()
            if stripped.startswith(marker):
                available = stripped.split(":", 1)[1].strip().split()[0] == "YES"
                break
    except Exception:
        available = False

    _OPENCV_BACKEND_CACHE[name] = available
    return available


def _vaapi_device_available() -> bool:
    dri_dir = "/dev/dri"
    if not os.path.isdir(dri_dir):
        return False
    try:
        return any(name.startswith(("renderD", "card")) for name in os.listdir(dri_dir))
    except Exception:
        return False


class _NvidiaFfmpegCapture:
    def __init__(self, rtsp_url, config=None, fps=None, connection_timeout=15):
        self.rtsp_url = rtsp_url
        self.config = config
        self.fps = fps
        self.connection_timeout = max(float(connection_timeout or 15), 1.0)
        self.process = None
        self.width = int(getattr(config, "FFMPEG_FRAME_WIDTH", getattr(Config, "FFMPEG_FRAME_WIDTH", 0)) or 0)
        self.height = int(getattr(config, "FFMPEG_FRAME_HEIGHT", getattr(Config, "FFMPEG_FRAME_HEIGHT", 0)) or 0)
        self.read_timeout_s = max(float(self._config_value("FFMPEG_READ_TIMEOUT_S", 3.0) or 3.0), 0.2)
        self._prefetched_frame = None
        self._stderr_file = None

    def _config_value(self, name: str, default):
        return getattr(self.config, name, getattr(Config, name, default))

    @staticmethod
    def _redact(text: str) -> str:
        return re.sub(r"(rtsp://[^:/\s]+:)[^@/\s]+(@)", r"\1****\2", text or "")

    def stderr_tail(self, limit: int = 4000) -> str:
        if not self._stderr_file:
            return ""
        try:
            self._stderr_file.flush()
            self._stderr_file.seek(0, os.SEEK_END)
            size = self._stderr_file.tell()
            self._stderr_file.seek(max(0, size - limit), os.SEEK_SET)
            return self._redact(self._stderr_file.read().decode("utf-8", errors="replace").strip())
        except Exception:
            return ""

    def _rtsp_transport(self) -> str:
        protocols = str(self._config_value("RTSP_PROTOCOLS", "tcp") or "tcp").strip().lower()
        if "," in protocols:
            protocols = protocols.split(",", 1)[0].strip()
        if protocols not in {"tcp", "udp", "udp_multicast", "http", "https"}:
            return "tcp"
        return protocols

    def _probe_size(self):
        if self.width > 0 and self.height > 0:
            return self.width, self.height

        ffprobe_binary = str(self._config_value("FFPROBE_BINARY", "ffprobe") or "ffprobe")
        cmd = [
            ffprobe_binary,
            "-v",
            "error",
            "-rtsp_transport",
            self._rtsp_transport(),
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            self.rtsp_url,
        ]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.connection_timeout,
                check=False,
            )
        except Exception as exc:
            logging.error("FFprobe failed before NVIDIA decode startup: %s", exc)
            return None

        if result.returncode != 0:
            logging.error("FFprobe could not read RTSP stream dimensions: %s", result.stderr.strip())
            return None

        for line in result.stdout.splitlines():
            text = line.strip()
            if not text or "x" not in text:
                continue
            width_text, height_text = text.split("x", 1)
            try:
                width = int(width_text)
                height = int(height_text)
            except ValueError:
                continue
            if width > 0 and height > 0:
                self.width = width
                self.height = height
                return width, height

        logging.error("FFprobe returned no usable RTSP stream dimensions.")
        return None

    def _build_command(self):
        ffmpeg_binary = str(self._config_value("FFMPEG_BINARY", "ffmpeg") or "ffmpeg")
        hwaccel = str(self._config_value("FFMPEG_HWACCEL", "cuda") or "cuda").strip()
        hwaccel_output_format = str(
            self._config_value("FFMPEG_HWACCEL_OUTPUT_FORMAT", "cuda") or "cuda"
        ).strip()
        codec = str(self._config_value("FFMPEG_VIDEO_CODEC", "") or "").strip()
        loglevel = str(self._config_value("FFMPEG_LOGLEVEL", "warning") or "warning").strip()

        cmd = [
            ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            loglevel,
            "-nostdin",
            "-rtsp_transport",
            self._rtsp_transport(),
            "-rw_timeout",
            str(int(self.read_timeout_s * 1_000_000)),
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-probesize",
            "1000000",
            "-analyzeduration",
            "1000000",
        ]
        if hwaccel:
            cmd.extend(["-hwaccel", hwaccel])
        if hwaccel_output_format:
            cmd.extend(["-hwaccel_output_format", hwaccel_output_format])
        if codec:
            cmd.extend(["-c:v", codec])

        cmd.extend([
            "-i",
            self.rtsp_url,
            "-an",
            "-sn",
            "-dn",
            "-vf",
            "hwdownload,format=nv12",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        return cmd

    def open(self) -> bool:
        if not self._probe_size():
            return False

        cmd = self._build_command()
        try:
            self._stderr_file = tempfile.TemporaryFile(mode="w+b")
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=self._stderr_file,
                bufsize=0,
            )
        except Exception as exc:
            logging.error("Failed to start NVIDIA FFmpeg decoder: %s", exc)
            self.process = None
            return False

        logging.info("Started NVIDIA FFmpeg decoder: size=%dx%d transport=%s", self.width, self.height, self._rtsp_transport())
        return self.isOpened()

    def isOpened(self):
        return bool(self.process and self.process.poll() is None and self.process.stdout)

    def _read_exact(self, size: int):
        if not self.isOpened():
            return None
        data = bytearray()
        deadline = time.monotonic() + self.read_timeout_s
        stdout_fd = self.process.stdout.fileno()
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logging.warning(
                    "FFmpeg decoder produced no complete frame for %.1fs; forcing RTSP reconnect",
                    self.read_timeout_s,
                )
                return None
            readable, _, _ = select.select([stdout_fd], [], [], remaining)
            if not readable:
                logging.warning(
                    "FFmpeg decoder stdout stalled for %.1fs; forcing RTSP reconnect",
                    self.read_timeout_s,
                )
                return None
            chunk = os.read(stdout_fd, min(size - len(data), 1024 * 1024))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    def read(self):
        if self._prefetched_frame is not None:
            frame = self._prefetched_frame
            self._prefetched_frame = None
            return True, frame

        frame_size = int(self.width * self.height * 3)
        raw = self._read_exact(frame_size)
        if raw is None:
            self.release()
            return False, None

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
        return True, frame

    def grab(self):
        ok, frame = self.read()
        if ok:
            self._prefetched_frame = frame
        return ok

    def set(self, *_args):
        return True

    def release(self):
        proc = self.process
        stderr_file = self._stderr_file
        self.process = None
        self._stderr_file = None
        self._prefetched_frame = None
        if not proc:
            if stderr_file:
                try:
                    stderr_file.close()
                except Exception:
                    pass
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        except Exception:
            pass
        finally:
            if stderr_file:
                try:
                    stderr_file.close()
                except Exception:
                    pass


def _decoder_backend_name(config_obj) -> str:
    return (
        getattr(config_obj, "DECODER_BACKEND", getattr(Config, "DECODER_BACKEND", "gstreamer"))
        or "gstreamer"
    ).strip().lower()


def _cuda_model_requested(config_obj) -> bool:
    model_backend = str(getattr(config_obj, "MODEL_BACKEND", getattr(Config, "MODEL_BACKEND", "")) or "").strip().lower()
    model_device = str(getattr(config_obj, "MODEL_DEVICE", getattr(Config, "MODEL_DEVICE", "")) or "").strip().lower()
    return model_backend == "torch" and model_device.startswith("cuda")


def _nvidia_ffmpeg_requested(decoder_backend: str, config_obj) -> bool:
    aliases = {
        "nvidia",
        "nvidia_ffmpeg",
        "nvidia-ffmpeg",
        "cuda",
        "cuda_ffmpeg",
        "cuda-ffmpeg",
        "ffmpeg_cuda",
        "ffmpeg-cuda",
        "nvdec",
        "gpu",
    }
    return decoder_backend in aliases or (decoder_backend == "ffmpeg" and _cuda_model_requested(config_obj))


def _nvidia_ffmpeg_explicit(decoder_backend: str) -> bool:
    return decoder_backend not in {"ffmpeg", "opencv", "native", "cv2", "default"}


def _strict_gpu_decode(config_obj) -> bool:
    return bool(getattr(config_obj, "DECODER_STRICT_GPU", getattr(Config, "DECODER_STRICT_GPU", False)))


def _decoder_chains(decoder_backend: str):
    chains = []
    prefer_vaapi = decoder_backend in {"gstreamer", "auto", "vaapi", "intel", "intel-vaapi", "hardware"}
    software_only = decoder_backend in {"software", "cpu", "sw"}

    if prefer_vaapi and _vaapi_device_available():
        vaapi_found = False
        if _gst_element_available("vaapih264dec") and _gst_element_available("vaapipostproc"):
            chains.append((
                "vaapi",
                "rtph264depay ! h264parse config-interval=-1 disable-passthrough=true "
                "! vaapih264dec ! vaapipostproc ! videoconvert n-threads=1",
            ))
            vaapi_found = True
        if _gst_element_available("vah264dec") and _gst_element_available("vapostproc"):
            chains.append((
                "va",
                "rtph264depay ! h264parse config-interval=-1 disable-passthrough=true "
                "! vah264dec ! vapostproc ! videoconvert n-threads=1",
            ))
            vaapi_found = True
        if _gst_element_available("vaapidecodebin"):
            if _gst_element_available("vaapipostproc"):
                chains.append((
                    "vaapi_bin",
                    "rtph264depay ! h264parse config-interval=-1 disable-passthrough=true "
                    "! vaapidecodebin ! vaapipostproc ! videoconvert n-threads=1",
                ))
            else:
                chains.append((
                    "vaapi_bin",
                    "rtph264depay ! h264parse config-interval=-1 disable-passthrough=true "
                    "! vaapidecodebin ! videoconvert n-threads=1",
                ))
            vaapi_found = True
        if _gst_element_available("vaapidecode"):
            if _gst_element_available("vaapipostproc"):
                chains.append((
                    "vaapi_legacy",
                    "rtph264depay ! h264parse config-interval=-1 disable-passthrough=true "
                    "! vaapidecode ! vaapipostproc ! videoconvert n-threads=1",
                ))
            else:
                chains.append((
                    "vaapi_legacy",
                    "rtph264depay ! h264parse config-interval=-1 disable-passthrough=true "
                    "! vaapidecode ! videoconvert n-threads=1",
                ))
            vaapi_found = True
        if not vaapi_found:
            logging.warning("VAAPI device is present but no VAAPI H.264 decoder plugins were found. Falling back to software decode.")
    elif prefer_vaapi:
        logging.warning("VAAPI decode requested but /dev/dri is not available. Falling back to software decode.")

    if not software_only or not chains:
        chains.append((
            "software",
            "rtph264depay ! h264parse config-interval=-1 disable-passthrough=true "
            "! avdec_h264 max-threads=1 ! videoconvert n-threads=1",
        ))

    return chains


def _gst_pipeline(rtsp_url, decode_stage, protocols="tcp", latency=200, fps="30/1", with_fps_limit=True):
    rate_stage = (
        f'! videorate ! video/x-raw,format=BGR,framerate={fps} '
        if with_fps_limit else
        '! video/x-raw,format=BGR '
    )
    return (
        f'rtspsrc location="{rtsp_url}" protocols={protocols} latency={latency} '
        f'drop-on-latency=true name=src '
        f'src. ! queue ! application/x-rtp,media=video,encoding-name=H264 '
        f'! rtpjitterbuffer latency={latency} '
        f'! {decode_stage} '
        f'{rate_stage}'
        f'! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=2 '
        f'! appsink drop=true sync=false max-buffers=1 '
        f'enable-last-sample=false wait-on-eos=false'
    )


def _gst_pipelines(rtsp_url, decoder_backend="gstreamer", protocols="tcp", latency=200, fps="30/1"):
    pipelines = []
    for decoder_name, decode_stage in _decoder_chains(decoder_backend):
        pipelines.append((
            f"{decoder_name}_fps",
            _gst_pipeline(rtsp_url, decode_stage, protocols=protocols, latency=latency, fps=fps, with_fps_limit=True),
        ))
        pipelines.append((
            f"{decoder_name}_generic",
            _gst_pipeline(rtsp_url, decode_stage, protocols=protocols, latency=latency, fps=fps, with_fps_limit=False),
        ))
    return pipelines


class RTSPConnection:
    def __init__(self, rtsp_url, max_retries=10, retry_delay=2, connection_timeout=15, config=None, fps=None):
        self.rtsp_url = rtsp_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_timeout = connection_timeout
        self.cap = None
        self.is_connected = False
        self.config = config
        self.fps = fps

    def _open_capture(self):
        protocols = getattr(self.config, "RTSP_PROTOCOLS", getattr(Config, "RTSP_PROTOCOLS", "tcp")) or "tcp"
        decoder_backend = _decoder_backend_name(self.config)
        latency = int(getattr(self.config, "RTSP_LATENCY_MS", getattr(Config, "RTSP_LATENCY_MS", 200)))
        fps = self.fps or getattr(self.config, "RTSP_FPS", getattr(Config, "RTSP_FPS", "20/1"))

        logging.info("Decoder backend preference: %s", decoder_backend)
        if _nvidia_ffmpeg_requested(decoder_backend, self.config):
            if self._open_nvidia_ffmpeg_capture():
                return True
            if _nvidia_ffmpeg_explicit(decoder_backend) and _strict_gpu_decode(self.config):
                return False
            logging.warning("NVIDIA FFmpeg decode failed; falling back to native OpenCV FFmpeg.")

        prefer_native_rtsp = decoder_backend in {"ffmpeg", "opencv", "native", "cv2", "default"}
        gstreamer_supported = _opencv_backend_available("GStreamer")

        if prefer_native_rtsp:
            return self._open_native_rtsp_capture()

        if gstreamer_supported:
            for key, gst in _gst_pipelines(
                self.rtsp_url,
                decoder_backend=decoder_backend,
                protocols=protocols,
                latency=latency,
                fps=fps,
            ):
                logging.info("Trying GStreamer pipeline: %s", key)
                cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                if self._use_opened_capture(cap, f"pipeline: {key}"):
                    return True
                if cap:
                    cap.release()
                logging.warning("Pipeline '%s' failed to produce frames; trying next...", key)
            logging.error("❌ Failed to open GStreamer pipelines.")
        else:
            logging.warning("OpenCV was built without GStreamer support. Falling back to native RTSP capture.")

        return self._open_native_rtsp_capture()

    def _open_nvidia_ffmpeg_capture(self) -> bool:
        logging.info("Trying NVIDIA FFmpeg CUDA/NVDEC decoder")
        cap = _NvidiaFfmpegCapture(
            self.rtsp_url,
            config=self.config,
            fps=self.fps,
            connection_timeout=self.connection_timeout,
        )
        if cap.open() and self._use_opened_capture(cap, "NVIDIA FFmpeg CUDA/NVDEC"):
            return True
        stderr = cap.stderr_tail()
        if stderr:
            logging.error("NVIDIA FFmpeg stderr before first-frame failure:\n%s", stderr)
        cap.release()
        logging.error("❌ Failed to open RTSP stream using NVIDIA FFmpeg CUDA/NVDEC.")
        self.is_connected = False
        return False

    def _use_opened_capture(self, cap, label: str) -> bool:
        if cap is None:
            return False
        if cap.isOpened() and cap.grab():
            self.cap = cap
            self.is_connected = True
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_N_THREADS, 1)
            logging.info("✅ RTSP opened successfully using %s", label)
            return True
        return False

    def _open_native_rtsp_capture(self) -> bool:
        ffmpeg_supported = _opencv_backend_available("FFMPEG")
        attempts = []

        if ffmpeg_supported:
            attempts.append(("ffmpeg", cv2.CAP_FFMPEG))
        attempts.append(("default", None))

        tried = set()
        for key, backend in attempts:
            if key in tried:
                continue
            tried.add(key)

            logging.info("Trying native RTSP backend: %s", key)
            cap = cv2.VideoCapture(self.rtsp_url) if backend is None else cv2.VideoCapture(self.rtsp_url, backend)
            if self._use_opened_capture(cap, f"native backend: {key}"):
                return True
            if cap:
                cap.release()
            logging.warning("Native backend '%s' failed to produce frames; trying next...", key)

        logging.error("❌ Failed to open RTSP stream using native OpenCV backends.")
        self.is_connected = False
        return False

    def connect(self):
        return self._open_capture()

    def reconnect(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        return self._open_capture()

    def check_health(self):
        return bool(self.cap and self.cap.isOpened())

    def read(self):
        if not self.cap:
            return False, None
        return self.cap.read()

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_connected = False


class VehicleTracker:
    def __init__(self, debug=False):
        self.backend_name = "custom"
        self.track_history = {}
        self.tracks = []
        self.next_local_id = 1
        self.max_distance = getattr(Config, "MAX_DISTANCE", 300)
        self.max_age = getattr(Config, "MAX_AGE", 20)
        self.min_iou = 0.1
        self.debug = debug
        self.track_id_prefix = TRACK_ID_PREFIX or "vehicle"

    def _new_track(self, det_bbox, reason="new_detection"):
        track_id = f"{self.track_id_prefix}_{self.next_local_id}"
        self.next_local_id += 1
        state = {"passed": False, "last_trigger_ts": 0.0}
        self.track_history[track_id] = state
        track = {
            "id": track_id,
            "bbox": det_bbox,
            "missed": 0,
            "hits": 1,
            "last_update": time.time(),
            "state": state,
        }
        self.tracks.append(track)
        logging.info("🆕 New Track ID: %s (%s)", track_id, reason)
        return track

    def _match_track(self, det_bbox, used_ids=None):
        if used_ids is None:
            used_ids = set()
        best = None
        best_score = -1e9
        debug_candidates = []
        cx_d, cy_d = bbox_center(det_bbox)
        for tr in self.tracks:
            if tr["id"] in used_ids:
                continue
            cx_t, cy_t = bbox_center(tr["bbox"])
            dist = np.hypot(cx_d - cx_t, cy_d - cy_t)
            iou_val = self._bbox_iou(det_bbox, tr["bbox"])
            if dist > (self.max_distance * 2.0) and iou_val < (self.min_iou * 0.5):
                if self.debug:
                    debug_candidates.append((tr["id"], iou_val, dist, None, "far_gate"))
                continue
            score = (2 * iou_val) - (dist / max(self.max_distance, 1.0))
            if self.debug:
                debug_candidates.append((tr["id"], iou_val, dist, score, "ok"))
            if score > best_score:
                best_score = score
                best = tr
        if self.debug and debug_candidates:
            msg = "; ".join(
                f"{tid} iou={iou:.2f} dist={dist:.1f} score={score if score is not None else 'skip'} {flag}"
                for tid, iou, dist, score, flag in debug_candidates
            )
            logging.info("[TRACK] candidates for %s: %s", det_bbox, msg)
        return best

    @staticmethod
    def _bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / float(area_a + area_b - inter + 1e-9)

    def update(self, detections, frame):
        assignments = []
        updated_ids = set()
        used_ids = set()

        for det in detections:
            bbox = det["bbox"]
            match = self._match_track(bbox, used_ids)
            if match is None:
                match = self._new_track(bbox, reason="no_match")
            else:
                match["bbox"] = bbox
                match["missed"] = 0
                match["hits"] += 1
                match["last_update"] = time.time()
            assignments.append((det, match))
            updated_ids.add(match["id"])
            used_ids.add(match["id"])

        for tr in self.tracks:
            if tr["id"] not in updated_ids:
                tr["missed"] += 1
        self.tracks = [tr for tr in self.tracks if tr["missed"] <= self.max_age]

        results = []
        for det, tr in assignments:
            state = tr.get("state", self.track_history.get(tr["id"], {"passed": False, "last_trigger_ts": 0.0}))
            results.append({
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "class_id": det.get("class_id"),
                "track_id": tr["id"],
                "passed": state.get("passed", False),
                "predicted": False,
            })
        if results or self.debug:
            logging.info("🔄 Tracker assignments: %d detections, %d active tracks", len(results), len(self.tracks))
        return results

    def ensure_track(self, track_id):
        if track_id not in self.track_history:
            self.track_history[track_id] = {"passed": False, "last_trigger_ts": 0.0}

    def trigger_once(self, track_id) -> bool:
        self.ensure_track(track_id)
        now = time.time()
        self.track_history[track_id]["passed"] = True
        self.track_history[track_id]["last_trigger_ts"] = now
        logging.info("✅ Track %s TRIGGERED", track_id)
        return True

    def get_active_track_ids(self):
        return {track["id"] for track in self.tracks}


class UltralyticsTrackerAdapter:
    def __init__(self, tracker_backend: str, debug: bool = False):
        if tracker_backend not in {"bytetrack", "botsort"}:
            raise ValueError(f"Unsupported Ultralytics tracker backend: {tracker_backend}")
        tracker_runtime_error = None
        try:
            tracker_bot_sort, tracker_byte, tracker_namespace = _load_ultralytics_tracker_runtime()
        except RuntimeError as exc:
            tracker_runtime_error = exc
            tracker_bot_sort = tracker_byte = tracker_namespace = None
        if tracker_runtime_error is not None:
            raise RuntimeError(
                f"TRACKER_BACKEND={tracker_backend} requires Ultralytics tracking dependencies. "
                "Use the NVIDIA image or install ultralytics + torch + lap in this runtime."
            ) from tracker_runtime_error

        with_reid = bool(getattr(Config, "TRACKER_WITH_REID", False))
        reid_model = str(getattr(Config, "TRACKER_REID_MODEL", "") or "").strip()
        if tracker_backend == "botsort" and with_reid and not reid_model:
            raise RuntimeError(
                "TRACKER_WITH_REID=true requires TRACKER_REID_MODEL in this repo. "
                "Native 'auto' detector features are not wired into the custom detection pipeline."
            )

        tracker_args = tracker_namespace(
            tracker_type=tracker_backend,
            track_high_thresh=float(getattr(Config, "TRACKER_HIGH_THRESH", 0.25)),
            track_low_thresh=float(getattr(Config, "TRACKER_LOW_THRESH", 0.10)),
            new_track_thresh=float(getattr(Config, "TRACKER_NEW_TRACK_THRESH", 0.25)),
            track_buffer=max(1, int(getattr(Config, "TRACKER_BUFFER", 30))),
            match_thresh=float(getattr(Config, "TRACKER_MATCH_THRESH", 0.80)),
            fuse_score=bool(getattr(Config, "TRACKER_FUSE_SCORE", True)),
            gmc_method=str(getattr(Config, "TRACKER_GMC_METHOD", "sparseOptFlow") or "sparseOptFlow"),
            proximity_thresh=float(getattr(Config, "TRACKER_PROXIMITY_THRESH", 0.50)),
            appearance_thresh=float(getattr(Config, "TRACKER_APPEARANCE_THRESH", 0.80)),
            with_reid=with_reid,
            model=reid_model if with_reid else "auto",
        )

        tracker_cls = tracker_byte if tracker_backend == "bytetrack" else tracker_bot_sort
        self.backend_name = tracker_backend
        self.debug = debug
        self.track_history = {}
        self.tracks = []
        self.track_id_prefix = TRACK_ID_PREFIX or "vehicle"
        tracker_kwargs = {"args": tracker_args}
        tracker_frame_rate = _tracker_frame_rate()
        try:
            tracker_signature = inspect.signature(tracker_cls)
            if "frame_rate" in tracker_signature.parameters:
                tracker_kwargs["frame_rate"] = tracker_frame_rate
        except (TypeError, ValueError):
            tracker_kwargs["frame_rate"] = tracker_frame_rate
        try:
            self.tracker = tracker_cls(**tracker_kwargs)
        except TypeError as exc:
            if "frame_rate" not in tracker_kwargs or "frame_rate" not in str(exc):
                raise
            tracker_kwargs.pop("frame_rate", None)
            self.tracker = tracker_cls(**tracker_kwargs)
        logging.info(
            "Initialized tracker backend=%s frame_rate=%s frame_rate_arg=%s with_reid=%s",
            tracker_backend,
            tracker_frame_rate,
            "frame_rate" in tracker_kwargs,
            with_reid,
        )

    def _format_track_id(self, raw_track_id) -> str:
        numeric = int(raw_track_id)
        return f"{self.track_id_prefix}_{numeric}" if self.track_id_prefix else str(numeric)

    def _sync_tracks(self):
        live_ids = set()
        for collection_name in ("tracked_stracks", "lost_stracks"):
            for track in getattr(self.tracker, collection_name, []):
                track_id = getattr(track, "track_id", None)
                if track_id is None:
                    continue
                live_ids.add(self._format_track_id(track_id))
        self.tracks = [{"id": track_id} for track_id in sorted(live_ids)]

    def ensure_track(self, track_id):
        if track_id not in self.track_history:
            self.track_history[track_id] = {"passed": False, "last_trigger_ts": 0.0}

    def trigger_once(self, track_id) -> bool:
        self.ensure_track(track_id)
        now = time.time()
        self.track_history[track_id]["passed"] = True
        self.track_history[track_id]["last_trigger_ts"] = now
        logging.info("✅ Track %s TRIGGERED", track_id)
        return True

    def get_active_track_ids(self):
        return {track["id"] for track in self.tracks}

    def update(self, detections, frame):
        tracker_input = _TrackerDetections(
            [det["bbox"] for det in detections],
            [det["confidence"] for det in detections],
            [det.get("class_id", -1) for det in detections],
        )
        tracked = self.tracker.update(tracker_input, frame)
        self._sync_tracks()

        if tracked.size == 0:
            if self.debug:
                logging.info("🔄 Tracker assignments: 0 detections, %d active tracks", len(self.tracks))
            return []

        results = []
        rows = np.atleast_2d(tracked)
        for row in rows:
            if len(row) < 7:
                continue
            x1, y1, x2, y2, raw_track_id, score, class_id = row[:7]
            track_id = self._format_track_id(raw_track_id)
            self.ensure_track(track_id)
            state = self.track_history[track_id]
            parsed_class_id = None
            try:
                if class_id is not None and not np.isnan(class_id):
                    parsed_class_id = int(round(float(class_id)))
            except Exception:
                parsed_class_id = None
            results.append({
                "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                "confidence": float(score),
                "class_id": parsed_class_id,
                "track_id": track_id,
                "passed": state.get("passed", False),
                "predicted": False,
            })

        if results or self.debug:
            logging.info("🔄 Tracker assignments: %d detections, %d active tracks", len(results), len(self.tracks))
        return results


def build_tracker():
    tracker_backend = str(getattr(Config, "TRACKER_BACKEND", "custom") or "custom").strip().lower()
    debug = bool(getattr(Config, "TRACKER_DEBUG", False))
    if tracker_backend == "custom":
        return VehicleTracker(debug=debug)
    if tracker_backend in {"bytetrack", "botsort"}:
        return UltralyticsTrackerAdapter(tracker_backend=tracker_backend, debug=debug)
    raise ValueError(f"Unsupported TRACKER_BACKEND: {tracker_backend}")


def _guess_format_and_scale(det, input_size):
    x1, y1, x2, y2 = det[:4]
    vals = np.array([x1, y1, x2, y2], dtype=float)
    vmax = float(np.max(np.abs(vals)) if vals.size else 0.0)
    is_xyxy = (x2 > x1) and (y2 > y1)
    scale_type = "norm" if vmax <= 2.0 else "pixel"
    return ("xyxy" if is_xyxy else "xywh"), scale_type


def _extract_output_tensor(output):
    if isinstance(output, dict) or hasattr(output, "values"):
        try:
            return np.array(next(iter(output.values())))
        except Exception:
            pass
    if isinstance(output, (list, tuple)):
        return np.array(output[0])
    return np.array(output)


def _reshape_predictions(tensor):
    tensor = np.asarray(tensor)
    if tensor.ndim == 0:
        return np.empty((0, 0), dtype=np.float32)
    if tensor.ndim == 1:
        return tensor.reshape(1, -1)
    if tensor.ndim == 2:
        # YOLO exports commonly use [channels, candidates] such as [84, 8400].
        if tensor.shape[0] >= 5 and tensor.shape[0] <= 512 and tensor.shape[1] > tensor.shape[0]:
            return tensor.T
        return tensor
    if tensor.ndim == 3:
        if tensor.shape[0] == 1:
            return _reshape_predictions(tensor[0])
        if tensor.shape[1] == 1:
            return _reshape_predictions(tensor[:, 0, :])
    return tensor.reshape(-1, tensor.shape[-1])


def _row_to_confidence_and_class(det, output_format):
    det = np.asarray(det, dtype=np.float32).reshape(-1)
    if det.size < 5:
        return 0.0, None

    if output_format == "single_class":
        return float(det[4]), None

    if output_format == "score_class":
        if det.size < 6:
            return float(det[4]), None
        return float(det[4]), int(round(det[5]))

    if output_format == "raw_with_objectness":
        if det.size <= 5:
            return float(det[4]), None
        class_scores = det[5:]
        if class_scores.size == 0:
            return float(det[4]), None
        class_id = int(np.argmax(class_scores))
        return float(det[4] * class_scores[class_id]), class_id

    if output_format == "raw_class_scores":
        class_scores = det[4:]
        class_id = int(np.argmax(class_scores))
        return float(class_scores[class_id]), class_id

    if det.size == 5:
        return float(det[4]), None

    if det.size == 6:
        cls_val = det[5]
        if abs(cls_val - round(cls_val)) <= 1e-3 and cls_val >= 0:
            return float(det[4]), int(round(cls_val))
        class_scores = det[4:]
        class_id = int(np.argmax(class_scores))
        return float(class_scores[class_id]), class_id

    tail = det[4:]
    if tail.size and float(np.max(np.abs(tail))) <= 1.0 + 1e-3:
        direct_scores = tail
        if det.size > 5:
            obj_scores = det[4] * det[5:]
            if obj_scores.size and float(np.max(obj_scores)) >= float(np.max(direct_scores)) * 0.75:
                class_id = int(np.argmax(obj_scores))
                return float(obj_scores[class_id]), class_id
        class_id = int(np.argmax(direct_scores))
        return float(direct_scores[class_id]), class_id

    if det.size > 5:
        return float(det[4]), int(round(det[5]))
    return float(det[4]), None


def _class_allowed(class_id, allowed_class_ids):
    if not allowed_class_ids or class_id is None:
        return True
    return class_id in allowed_class_ids


def _det_to_xyxy(det, input_size, output_format="single_class"):
    if output_format in {"raw_class_scores", "raw_with_objectness"}:
        fmt = "xywh"
        vals = np.array(det[:4], dtype=float)
        vmax = float(np.max(np.abs(vals)) if vals.size else 0.0)
        scale_type = "norm" if vmax <= 2.0 else "pixel"
    else:
        fmt, scale_type = _guess_format_and_scale(det[:4], input_size)
    a, b, c, d = det[:4]

    if fmt == "xywh":
        xc, yc, ww, hh = a, b, c, d
        if scale_type == "norm":
            xc *= input_size[1]
            yc *= input_size[0]
            ww *= input_size[1]
            hh *= input_size[0]
        return (
            xc - ww / 2.0,
            yc - hh / 2.0,
            xc + ww / 2.0,
            yc + hh / 2.0,
        )

    x1, y1, x2, y2 = a, b, c, d
    if scale_type == "norm":
        x1 *= input_size[1]
        x2 *= input_size[1]
        y1 *= input_size[0]
        y2 *= input_size[0]
    return x1, y1, x2, y2


def postprocess(
    output,
    full_frame_shape,
    scale,
    conf_threshold=0.5,
    iou_threshold=0.4,
    roi_offset=(0, 0),
    pad=None,
    input_size=(416, 416),
    output_format="single_class",
    allowed_class_ids=None,
):
    global _POSTPROCESS_LOGGED

    full_h, full_w = full_frame_shape[:2]
    dw, dh = (0, 0) if pad is None else pad
    dw = int(dw)
    dh = int(dh)

    preds = _reshape_predictions(_extract_output_tensor(output))
    if preds.size == 0:
        return []

    boxes = []
    scores = []
    class_ids = []

    for det in preds:
        confidence, class_id = _row_to_confidence_and_class(det, output_format)
        if confidence <= 0 or confidence < conf_threshold:
            continue
        if not _class_allowed(class_id, allowed_class_ids):
            continue

        x1_pad, y1_pad, x2_pad, y2_pad = _det_to_xyxy(det, input_size, output_format=output_format)
        x1_res = x1_pad - dw
        y1_res = y1_pad - dh
        x2_res = x2_pad - dw
        y2_res = y2_pad - dh

        if scale == 0:
            continue

        x1 = int(np.clip((x1_res / scale) + roi_offset[0], 0, full_w))
        y1 = int(np.clip((y1_res / scale) + roi_offset[1], 0, full_h))
        x2 = int(np.clip((x2_res / scale) + roi_offset[0], 0, full_w))
        y2 = int(np.clip((y2_res / scale) + roi_offset[1], 0, full_h))
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2, y2])
        scores.append(float(confidence))
        class_ids.append(class_id)

        if not _POSTPROCESS_LOGGED:
            logging.info(
                "[PP] First raw det: %.4f,%.4f,%.4f,%.4f conf=%.3f class=%s format=%s",
                float(det[0]),
                float(det[1]),
                float(det[2]),
                float(det[3]),
                float(confidence),
                class_id,
                output_format,
            )
            _POSTPROCESS_LOGGED = True

    if not boxes:
        return []

    boxes_xywh = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in boxes]
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf_threshold, iou_threshold)

    if isinstance(idxs, (list, tuple)):
        idxs = np.array(idxs).reshape(-1)
    elif hasattr(idxs, "flatten"):
        idxs = idxs.flatten()
    else:
        idxs = np.array([])

    if len(idxs) == 0 and len(boxes) == 1:
        idxs = np.array([0])

    detections = []
    for i in idxs:
        idx = int(i)
        detections.append({
            "bbox": boxes[idx],
            "confidence": float(scores[idx]),
            "class_id": class_ids[idx],
        })
    return detections


def preprocess(image, input_size=(416, 416)):
    h, w = image.shape[:2]
    scale = min(input_size[1] / w, input_size[0] / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = input_size[1] - new_w
    pad_h = input_size[0] - new_h
    dw = pad_w // 2
    dh = pad_h // 2

    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    padded[dh:dh + new_h, dw:dw + new_w] = resized

    padded = padded[:, :, ::-1].astype(np.float32)
    if USE_MODEL_NORMALIZATION:
        padded = (padded - MODEL_MEAN) / MODEL_STD
    else:
        padded /= 255.0
    input_tensor = padded.transpose(2, 0, 1)
    return np.expand_dims(input_tensor, 0), scale, (new_h, new_w), (dw, dh)


def _normalize_roi_coords(frame_shape, roi_coords):
    if not roi_coords:
        return None
    full_h, full_w = frame_shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in roi_coords]
    x1 = int(np.clip(x1, 0, full_w))
    y1 = int(np.clip(y1, 0, full_h))
    x2 = int(np.clip(x2, 0, full_w))
    y2 = int(np.clip(y2, 0, full_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _nms_indices(boxes, scores, iou_threshold):
    if not boxes:
        return np.array([], dtype=np.int32)

    boxes_xywh = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in boxes]
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, 0.0, iou_threshold)

    if isinstance(idxs, (list, tuple)):
        idxs = np.array(idxs).reshape(-1)
    elif hasattr(idxs, "flatten"):
        idxs = idxs.flatten()
    else:
        idxs = np.array([], dtype=np.int32)

    if len(idxs) == 0 and len(boxes) == 1:
        idxs = np.array([0], dtype=np.int32)
    return idxs


def _detect_vehicle_region_openvino(image, context, roi_coords=None, label="primary"):
    full_h, full_w = image.shape[:2]
    roi_coords = _normalize_roi_coords(image.shape, roi_coords)

    if roi_coords:
        x1, y1, x2, y2 = roi_coords
        cropped = image[y1:y2, x1:x2].copy()
        roi_offset = (x1, y1)
    else:
        cropped = image
        roi_offset = (0, 0)

    input_tensor, scale, _, pad = preprocess(cropped, input_size=context["input_size"])
    result = context["compiled_model"]([input_tensor])
    detections = postprocess(
        result,
        (full_h, full_w, 3),
        scale,
        conf_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=NMS_THRESHOLD,
        roi_offset=roi_offset,
        pad=pad,
        input_size=context["input_size"],
        output_format=context["output_format"],
        allowed_class_ids=context["allowed_class_ids"],
    )

    if detections:
        logging.info(
            "[DET:%s] scale=%.5f pad=%s roi=%s full=%s first_box=%s count=%d",
            label,
            scale,
            pad,
            roi_coords if roi_coords else "full",
            image.shape[:2],
            detections[0]["bbox"],
            len(detections),
        )
    for det in detections:
        det["source"] = label
    return detections


def _detect_vehicle_region_torch(image, context, roi_coords=None, label="primary"):
    full_h, full_w = image.shape[:2]
    roi_coords = _normalize_roi_coords(image.shape, roi_coords)

    if roi_coords:
        x1, y1, x2, y2 = roi_coords
        cropped = image[y1:y2, x1:x2].copy()
        roi_offset = (x1, y1)
    else:
        cropped = image
        roi_offset = (0, 0)

    allowed_class_ids = sorted(context["allowed_class_ids"]) if context["allowed_class_ids"] else None
    imgsz = max(int(context["input_size"][0]), int(context["input_size"][1]))
    results = context["yolo_model"].predict(
        source=cropped,
        imgsz=imgsz,
        conf=CONFIDENCE_THRESHOLD,
        iou=NMS_THRESHOLD,
        device=context["device"],
        classes=allowed_class_ids,
        half=context.get("half", False),
        verbose=False,
    )

    detections = []
    result = results[0] if results else None
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else None
        for idx, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.tolist()
            gx1 = int(np.clip(x1 + roi_offset[0], 0, full_w))
            gy1 = int(np.clip(y1 + roi_offset[1], 0, full_h))
            gx2 = int(np.clip(x2 + roi_offset[0], 0, full_w))
            gy2 = int(np.clip(y2 + roi_offset[1], 0, full_h))
            if gx2 <= gx1 or gy2 <= gy1:
                continue
            class_id = int(classes[idx]) if classes is not None else None
            detections.append({
                "bbox": [gx1, gy1, gx2, gy2],
                "confidence": float(confs[idx]),
                "class_id": class_id,
                "source": label,
            })

    if detections:
        logging.info(
            "[DET:%s] backend=%s device=%s roi=%s full=%s first_box=%s count=%d",
            label,
            "torch",
            context["device"],
            roi_coords if roi_coords else "full",
            image.shape[:2],
            detections[0]["bbox"],
            len(detections),
        )
    return detections


def _detect_vehicle_region(image, context, roi_coords=None, label="primary"):
    backend = context.get("backend", "openvino")
    if backend == "openvino":
        return _detect_vehicle_region_openvino(image, context, roi_coords=roi_coords, label=label)
    if backend == "torch":
        return _detect_vehicle_region_torch(image, context, roi_coords=roi_coords, label=label)
    raise ValueError(f"Unsupported detection backend: {backend}")


def _merge_detections(detection_groups, iou_threshold):
    combined = []
    for detections in detection_groups:
        for det in detections:
            combined.append({
                "bbox": list(det["bbox"]),
                "confidence": float(det["confidence"]),
                "class_id": det.get("class_id"),
                "source": det.get("source"),
            })

    if len(combined) <= 1:
        return combined

    boxes = [det["bbox"] for det in combined]
    scores = [det["confidence"] for det in combined]
    idxs = _nms_indices(boxes, scores, iou_threshold)
    if len(idxs) == 0:
        return []
    return [combined[int(i)] for i in idxs]


def detect_vehicles(image, roi_coords=None):
    context = get_model_context()
    primary_roi = _normalize_roi_coords(image.shape, roi_coords)
    detections_primary = _detect_vehicle_region(image, context, roi_coords=primary_roi, label="near")

    detection_groups = [detections_primary]
    far_roi = None
    if ENABLE_FAR_DETECTION and FAR_DETECTION_ROI:
        far_roi = _normalize_roi_coords(image.shape, get_roi_coordinates(image.shape, FAR_DETECTION_ROI))
        if far_roi == primary_roi:
            far_roi = None

    detections_far = []
    if far_roi is not None:
        detections_far = _detect_vehicle_region(image, context, roi_coords=far_roi, label="far")
        detection_groups.append(detections_far)

    merged = _merge_detections(detection_groups, iou_threshold=MERGE_NMS_THRESHOLD)
    if far_roi is not None and (detections_far or len(merged) != len(detections_primary)):
        logging.info(
            "Merged multi-pass detections: near=%d far=%d merged=%d far_roi=%s",
            len(detections_primary),
            len(detections_far),
            len(merged),
            far_roi,
        )
    return merged


def refine_bbox_to_edges(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return [x1, y1, x2, y2]


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def get_roi_coordinates_multi(frame_shape, roi_list_ratios):
    height, width = frame_shape[:2]
    return [
        (
            int(roi[0] * width),
            int(roi[1] * height),
            int(roi[2] * width),
            int(roi[3] * height),
        )
        for roi in roi_list_ratios
    ]


def get_polygon_coordinates_multi(frame_shape, polygon_list_ratios):
    height, width = frame_shape[:2]
    polygons = []
    for polygon in polygon_list_ratios:
        points = []
        for x_ratio, y_ratio in polygon:
            points.append((int(round(x_ratio * width)), int(round(y_ratio * height))))
        if len(points) >= 3:
            polygons.append(np.array(points, dtype=np.int32))
    return polygons


def get_roi_coordinates(frame_shape, roi_ratios):
    height, width = frame_shape[:2]
    return (
        int(roi_ratios[0] * width),
        int(roi_ratios[1] * height),
        int(roi_ratios[2] * width),
        int(roi_ratios[3] * height),
    )


def is_bbox_in_roi(bbox, roi):
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]


def is_bbox_in_polygon(bbox, polygon):
    cx, cy = bbox_center(bbox)
    contour = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0


def draw_roi(image, roi_coords):
    cv2.rectangle(image, (roi_coords[0], roi_coords[1]), (roi_coords[2], roi_coords[3]), (255, 0, 0), 3)


def draw_count_rois(image, count_rois, roi_counts):
    palette = [
        (0, 255, 255),
        (255, 255, 0),
        (255, 128, 0),
        (255, 0, 255),
        (0, 180, 255),
        (180, 255, 0),
    ]
    for idx, roi in enumerate(count_rois):
        x1, y1, x2, y2 = roi
        color = palette[idx % len(palette)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        count = roi_counts[idx] if idx < len(roi_counts) else 0
        label = f"Count {idx + 1}: {count}"
        cv2.putText(image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_count_polygons(image, count_polygons, roi_counts):
    palette = [
        (0, 255, 255),
        (255, 255, 0),
        (255, 128, 0),
        (255, 0, 255),
        (0, 180, 255),
        (180, 255, 0),
    ]
    overlay = image.copy()
    for idx, polygon in enumerate(count_polygons):
        contour = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
        color = palette[idx % len(palette)]
        cv2.fillPoly(overlay, [contour], color)
    cv2.addWeighted(overlay, 0.12, image, 0.88, 0, image)

    for idx, polygon in enumerate(count_polygons):
        contour = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
        flat = contour.reshape(-1, 2)
        color = palette[idx % len(palette)]
        cv2.polylines(image, [contour], isClosed=True, color=color, thickness=2)
        count = roi_counts[idx] if idx < len(roi_counts) else 0
        label = f"Count {idx + 1}: {count}"
        label_x = int(np.min(flat[:, 0]))
        label_y = int(np.min(flat[:, 1]))
        cv2.putText(image, label, (label_x, max(20, label_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_detections(image, detections, roi_coords):
    for detection in detections:
        if detection["confidence"] == 0 and not detection["predicted"]:
            continue
        x1, y1, x2, y2 = map(int, detection["bbox"])
        in_roi = is_bbox_in_roi(detection["bbox"], roi_coords)
        if detection.get("zone_class_match"):
            color = (0, 0, 255)
        else:
            color = BBOX_COLOR
        cv2.rectangle(image, (x1, y1), (x2, y2), color, BBOX_LINE_THICKNESS)

        class_id = detection.get("class_id")
        class_label = DETECTION_CLASS_LABELS.get(class_id, "Detection")
        label = f"{class_label} {_display_track_id(detection.get('track_id'))} {detection['confidence']:.2f}"
        if class_id is not None:
            label += f" C:{class_id}"
        if in_roi:
            label += " [ROI]"
        cv2.putText(image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


class AsyncVehicleDetector:
    def __init__(self):
        self.frame_buffer = FrameBuffer(maxsize=1)
        self.result_buffer = FrameBuffer(maxsize=1)
        self.processing = True
        self.detection_thread = None
        self.tracker = build_tracker()

    def start_processing(self):
        self.detection_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.detection_thread.start()

    def _process_frames(self):
        while self.processing:
            item = self.frame_buffer.get(timeout=0.05)
            if item is None:
                continue
            frame, roi_coords = item
            try:
                raw_detections = detect_vehicles(frame, roi_coords)
                tracked = self.tracker.update(raw_detections, frame)
                self.result_buffer.put((frame.copy(), tracked, roi_coords))
            except Exception as exc:
                logging.error("Detection error: %s", exc)

    def add_frame(self, frame, roi_coords):
        self.frame_buffer.put((frame.copy(), roi_coords))

    def get_result(self):
        return self.result_buffer.get()

    def get_tracker(self):
        return self.tracker

    def stop(self):
        self.processing = False
        if self.detection_thread:
            self.detection_thread.join()


# Backward-compatible aliases while the rest of the repo transitions off plate naming.
PlateTracker = VehicleTracker
AsyncPlateDetector = AsyncVehicleDetector
detect_plates = detect_vehicles
