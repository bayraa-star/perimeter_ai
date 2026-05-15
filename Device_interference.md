# Device Inference Profiles

This note defines which model and decoder variables are needed for each hardware profile.

Important distinction:

- `MODEL_BACKEND` and `MODEL_DEVICE` control inference.
- `DECODER_BACKEND` controls RTSP video decode.
- `DETECTION_MODEL_PATH` is used only with `MODEL_BACKEND=openvino`.
- `TORCH_MODEL_PATH` is used only with `MODEL_BACKEND=torch`.
- `BBOX_LINE_THICKNESS` controls preview bounding-box line width only.
- Red bbox highlighting uses `COUNT_ZONES` and `COUNT_ZONE_CLASS_IDS`: a bbox turns red when its class is valid for the zone it intersects.
- `TRACKER_BACKEND` controls tracking algorithm selection independently from inference backend.

Tracker note:

- `TRACKER_BACKEND=custom` works with the existing local tracker.
- `TRACKER_BACKEND=bytetrack` or `TRACKER_BACKEND=botsort` require a runtime that has `ultralytics` and `lap` tracking dependencies available.

## Frontend Form Behavior

If the frontend uses dropdowns for `MODEL_BACKEND`, `MODEL_DEVICE`, and `DETECTION_MODEL_PATH`, the expected behavior is:

- CPU profile:
  - set `MODEL_BACKEND=openvino`
  - set `MODEL_DEVICE=CPU`
  - keep `DETECTION_MODEL_PATH`
- Intel GPU profile:
  - set `MODEL_BACKEND=openvino`
  - set `MODEL_DEVICE=GPU`
  - keep `DETECTION_MODEL_PATH`
- NVIDIA GPU profile:
  - set `MODEL_BACKEND=torch`
  - set `MODEL_DEVICE=cuda:0`
  - do not require `DETECTION_MODEL_PATH`
  - require `TORCH_MODEL_PATH` instead

## Required Variables By Hardware

### 1. CPU

Inference:

- `MODEL_BACKEND=openvino`
- `MODEL_DEVICE=CPU`
- `DETECTION_MODEL_PATH=models/yolov8s_openvino_model/yolov8s.xml`

Decoder:

- `DECODER_BACKEND=ffmpeg` or `DECODER_BACKEND=software`
- `RTSP_LATENCY_MS=200`

Notes:

- `LIBVA_DRIVER_NAME` is not needed.
- `TORCH_MODEL_PATH` is not needed.

### 2. Intel GPU

Inference:

- `MODEL_BACKEND=openvino`
- `MODEL_DEVICE=GPU`
- `DETECTION_MODEL_PATH=models/yolov8s_openvino_model/yolov8s.xml`

Decoder:

- `DECODER_BACKEND=vaapi`
- `LIBVA_DRIVER_NAME=i965`
- `RTSP_LATENCY_MS=200`

Optional but commonly needed in deployment:

- `LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri`
- `GST_VAAPI_DRM_DEVICE=/dev/dri/renderD128`
- `GST_VAAPI_ALL_DRIVERS=1`

Notes:

- Container must have `/dev/dri` mounted.
- `TORCH_MODEL_PATH` is not needed.

### 3. NVIDIA GPU

Inference:

- `MODEL_BACKEND=torch`
- `MODEL_DEVICE=cuda:0`
- `TORCH_MODEL_VARIANT=nano|small|medium|large`
- `TORCH_MODEL_PATH=models/yolov8s.pt`
- `TORCH_HALF=true` recommended

Decoder:

- `DECODER_BACKEND=nvidia_ffmpeg`
- `RTSP_LATENCY_MS=200`

Notes:

- `DETECTION_MODEL_PATH` is not needed for this profile.
- `TORCH_MODEL_PATH` can still override the preset selected by `TORCH_MODEL_VARIANT`.
- This profile should use the image built from `dockerfile.nvidia`.
- Container must be started with NVIDIA GPU access and `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`.

## Short Mapping Table

| Hardware | MODEL_BACKEND | MODEL_DEVICE | Model path field | DECODER_BACKEND |
| --- | --- | --- | --- | --- |
| CPU | `openvino` | `CPU` | `DETECTION_MODEL_PATH` | `ffmpeg` or `software` |
| Intel GPU | `openvino` | `GPU` | `DETECTION_MODEL_PATH` | `vaapi` |
| NVIDIA GPU | `torch` | `cuda:0` | `TORCH_MODEL_PATH` or `TORCH_MODEL_VARIANT` | `nvidia_ffmpeg` |

## Frontend Rule

When `MODEL_BACKEND=torch`:

- hide or disable `DETECTION_MODEL_PATH`
- show `TORCH_MODEL_PATH`
- optionally show `TORCH_MODEL_VARIANT` as dropdown with `nano`, `small`, `medium`, `large`

When `MODEL_BACKEND=openvino`:

- show `DETECTION_MODEL_PATH`
- hide or ignore `TORCH_MODEL_PATH`
