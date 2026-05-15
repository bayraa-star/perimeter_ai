# AGENT.md

## Purpose

This repository is a Python service for vehicle event capture from RTSP streams.
It uses OpenVINO for vehicle detection, a lightweight in-process tracker for ID assignment,
Flask for MJPEG preview streaming, and a browser-based polygon ROI editor for configuring
counting zones over the live preview.

The code is optimized for low latency more than for clean separation of concerns.
Most operational behavior is concentrated in `main.py`.

## Repository Map

- `main.py`
  Main orchestration loop. Starts Flask, opens RTSP streams, handles reconnects,
  runs motion gating, drains detector results, updates per-zone counts, and
  pushes preview frames.
- `utils/openvinoDetector.py`
  OpenVINO model loading, preprocessing, postprocessing, RTSP capture wrapper,
  simple tracker, and async detector worker.
- `utils/capture.py`
  Producer thread for RTSP frames. Keeps only the freshest frame in the queue.
- `service/flask_server.py`
  `/video_feed` MJPEG endpoint backed by a shared frame queue and `/roi_editor`
  polygon ROI editor UI.
- `config/config.py`
  Environment-driven config dataclass. Most values behave like class attributes;
  the hot-path detection and trigger tuning lives here.
- `utils/modelTest.py`
  Standalone offline model experiment script. Not part of the live service path.
- `test.py`
  Minimal OpenCV/GStreamer check.
- `models/20251020_094033/`
  Current OpenVINO model artifacts.

## Runtime Flow

1. `main.py` instantiates `Config`.
2. Flask preview server starts in a daemon thread.
3. Tracking RTSP connection is opened.
4. `AsyncVehicleDetector` starts a worker thread.
5. Capture threads feed fresh frames into small queues.
6. Main loop:
   - checks stream health and reconnects if needed
   - pulls the newest tracking frame
   - computes motion inside `DETECTION_ROI`
   - every `SKIP_FRAMES`, submits a frame to the detector when motion is present
   - drains detector results to the latest one
   - increments polygon zone counters per tracked detection
   - draws overlay and pushes frame to Flask preview

## Configuration Notes

Important environment variables:

- `TRACK_RTSP_URL`
- `STREAM_PORT`
- `DETECTION_ROI`
- `COUNT_ZONES`
- `CONFIDENCE_THRESHOLD`
- `SKIP_FRAMES`
- `MAX_FPS`
- `TRACK_RTSP_FPS`
- `DECODER_BACKEND`
- `LIBVA_DRIVER_NAME`
- `LIBVA_DRIVERS_PATH`
- `GST_VAAPI_DRM_DEVICE`
- `GST_VAAPI_ALL_DRIVERS`
- `RTSP_LATENCY_MS`
- `PREVIEW_FPS`
- `MOTION_CHECK_INTERVAL_S`
- `MOTION_DOWNSCALE`
- `ENABLE_LOCAL_KEYBOARD_EXIT`

Behavioral quirks:

- Most values in `Config` are class-style attributes populated from env at
  import time.
- `DEBUG_DRAW` is inverted:
  `os.getenv("DEBUG_DRAW", "True") == "False"` means the default evaluates to `False`.
- `RTSPConnection` now reads `RTSP_LATENCY_MS` for the GStreamer RTSP latency setting.

## Intel VAAPI Decode

The tracking pipeline supports software decode and Intel VAAPI decode.
`DECODER_BACKEND=vaapi` is not enough by itself; the container also needs Intel
GPU device access and the correct VAAPI env vars.

Required Docker/runtime settings:

- mount `/dev/dri` into the container
- `DECODER_BACKEND=vaapi`
- `LIBVA_DRIVER_NAME=<driver>`
- `LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri`
- `GST_VAAPI_DRM_DEVICE=/dev/dri/renderDXXX`
- `GST_VAAPI_ALL_DRIVERS=1`

Validated patrol PC settings:

- GPU: Intel HD Graphics 630
- render node: `/dev/dri/renderD128`
- working driver: `LIBVA_DRIVER_NAME=i965`
- successful pipeline name in logs: `vaapi_fps`
- successful decode stage in logs:
  `vaapih264dec ! vaapipostproc ! videoconvert`

Expected success signals in logs:

- `Trying GStreamer pipeline: vaapi_fps`
- `RTSP opened successfully using pipeline: vaapi_fps`
- no fallback message about software decode

Fallback behavior:

- if `/dev/dri` is missing, the service falls back to software decode
- if VAAPI decoder elements are unavailable, the service falls back to
  `avdec_h264`

Deployment note for container managers:

- In Docker CLI this means `--device /dev/dri:/dev/dri`
- In Dockerode/Node `HostConfig.Devices` must include a mapping for `/dev/dri`
- If the device mapping is missing, the VAAPI env vars alone do nothing

## Dependency and Environment Notes

- `cv2` is used throughout the codebase but no OpenCV package is listed.
- The local `.venv` in this workspace does not currently contain the required runtime packages.
- `requirements.txt` also contains packages that do not appear in the live code path:
  `psutil`, `deep_sort_realtime`, `torch`, `torchvision`.
- The runtime image, not `requirements.txt`, is the real source of truth for OpenCV,
  GStreamer, VAAPI, and Intel media-driver packages.

Safe validation command:

```bash
python3 -m py_compile main.py service/flask_server.py utils/openvinoDetector.py utils/capture.py config/config.py
```

## Current Risks and Footguns

- `utils/openvinoDetector.py` loads the OpenVINO model at import time from a hardcoded
  relative path. Running outside the repo root can fail immediately.
- `RTSPConnection.cap` is accessed from multiple threads without synchronization.
  Reconnect and `read()` can race.
- Track state is duplicated:
  `main.py` keeps zone-count state, while `VehicleTracker` keeps `track_history`.
  Both can grow over long runtimes.
- The Flask preview uses one shared queue for all clients, so multiple viewers compete
  for frames instead of each client receiving an independent live stream.

## When Editing This Code

- Start from `main.py`. Most behavior changes route through it.
- Treat the repo as latency-sensitive. Queues are intentionally tiny.
- When changing decode behavior, verify the logs show the expected pipeline name.
  `software_fps` means CPU decode; `vaapi_fps` means Intel VAAPI decode.
- Be careful with import-time side effects in `utils/openvinoDetector.py`.
- Avoid adding blocking work inside the main loop.
- If you change counting behavior, inspect both `main.py` and `VehicleTracker`.

## Suggested Stabilization Order

1. Make model loading explicit and configurable instead of import-time and hardcoded.
2. Add synchronization around RTSP capture/reconnect lifecycle.
3. Consolidate zone-count state into one owner.
4. Fix dependency declarations and add a reproducible setup path.
5. Split the ROI editor HTML/JS out of the Python string template.

## Practical Commands

Syntax-only check:

```bash
python3 -m py_compile main.py service/flask_server.py utils/openvinoDetector.py utils/capture.py config/config.py
```

Search the hot path:

```bash
rg -n "count|zone|reconnect|detect|frame_queue" main.py utils service config
```

Check project files quickly:

```bash
rg --files
```
