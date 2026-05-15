# Project Capabilities

## Purpose

This project is a Python RTSP video analytics service for traffic, perimeter, and crossroad-style camera scenes.

It can ingest an RTSP stream, run object detection, assign tracker IDs, count tracked objects inside configured polygon zones, stream a live annotated preview, send structured backend events, and optionally capture/upload raw entry photos.

The service is designed to run as a Docker container and be controlled by environment variables from an external backend.

## Runtime Interfaces

The Flask preview server exposes:

- `/`
  Simple index page with links to the preview and ROI editor.

- `/video_feed`
  Live MJPEG stream of the processed frame with overlays.

- `/roi_editor`
  Browser-based polygon editor for building `COUNT_ZONES` values.

Outbound integrations:

- JSON analytics event batches to `COUNTER_RECORD_API_URL`
- Multipart image uploads to `FILE_IMAGE_UPLOAD_URL`

## Current Functional Capabilities

### 1. RTSP Stream Ingestion

- Connects to a tracking RTSP camera stream.
- Performs health checks and reconnect attempts.
- Uses a one-frame queue in the hot path so stale frames are dropped instead of building latency.
- Supports configurable RTSP transport, latency, FPS, and stream dimensions.

Main config:

- `RTSP_URL`
- `TRACK_RTSP_URL`
- `RECONNECT_ATTEMPTS`
- `RECONNECT_DELAY`
- `RTSP_PROTOCOLS`
- `RTSP_LATENCY_MS`
- `RTSP_FPS`
- `TRACK_RTSP_FPS`
- `STREAM_HEALTHCHECK_INTERVAL_S`

### 2. Decode Backends

Supported decode paths include:

- GStreamer/OpenCV pipelines
- Native OpenCV/FFmpeg RTSP capture
- Intel VAAPI-oriented decode paths
- NVIDIA FFmpeg CUDA/NVDEC raw-frame capture

Main config:

- `DECODER_BACKEND`
- `LIBVA_DRIVER_NAME`
- `FFMPEG_BINARY`
- `FFPROBE_BINARY`
- `FFMPEG_HWACCEL`
- `FFMPEG_HWACCEL_OUTPUT_FORMAT`
- `FFMPEG_VIDEO_CODEC`
- `FFMPEG_LOGLEVEL`
- `FFMPEG_FRAME_WIDTH`
- `FFMPEG_FRAME_HEIGHT`

### 3. Detection

- Supports OpenVINO inference.
- Supports Torch/Ultralytics inference.
- Supports model input resizing and optional normalization.
- Filters detections by configured class IDs.
- Runs detection only inside `DETECTION_ROI`.
- Optionally runs a second pass on `FAR_DETECTION_ROI` to improve small/far object detection.
- Merges near/far pass detections with NMS.

Main config:

- `MODEL_BACKEND`
- `MODEL_DEVICE`
- `PERFORMANCE_HINT`
- `DETECTION_MODEL_PATH`
- `TORCH_MODEL_VARIANT`
- `TORCH_MODEL_PATH`
- `TORCH_HALF`
- `MODEL_INPUT_SIZE`
- `DETECTION_OUTPUT_FORMAT`
- `DETECTION_CLASS_IDS`
- `USE_MODEL_NORMALIZATION`
- `MODEL_MEAN`
- `MODEL_STD`
- `CONFIDENCE_THRESHOLD`
- `NMS_THRESHOLD`
- `MERGE_NMS_THRESHOLD`
- `DETECTION_ROI`
- `ENABLE_FAR_DETECTION`
- `FAR_DETECTION_ROI`

Typical COCO class IDs used here:

- `0` = person
- `2` = car
- `3` = motorcycle
- `5` = bus
- `7` = truck

### 4. Motion Gating

- Computes motion inside `DETECTION_ROI`.
- Skips detector submission when recent ROI motion is below threshold.
- Supports downscaled motion comparison for lower CPU use.

Main config:

- `PIXEL_CHANGE_THRESHOLD`
- `DIFF_THRESHOLD`
- `MOTION_CHECK_INTERVAL_S`
- `MOTION_DOWNSCALE`
- `SKIP_FRAMES`
- `MAX_FPS`

### 5. Tracking

Supported tracker backends:

- `custom`
- `bytetrack`
- `botsort`

The custom tracker uses local distance/age matching. ByteTrack and BoT-SORT use Ultralytics tracker runtime when dependencies are present.

Main config:

- `TRACKER_BACKEND`
- `MAX_DISTANCE`
- `MAX_AGE`
- `MIN_CONFIDENCE`
- `TRACKER_DEBUG`
- `TRACK_ID_PREFIX`
- `TRACKER_HIGH_THRESH`
- `TRACKER_LOW_THRESH`
- `TRACKER_NEW_TRACK_THRESH`
- `TRACKER_BUFFER`
- `TRACKER_MATCH_THRESH`
- `TRACKER_FUSE_SCORE`
- `TRACKER_GMC_METHOD`
- `TRACKER_WITH_REID`
- `TRACKER_PROXIMITY_THRESH`
- `TRACKER_APPEARANCE_THRESH`
- `TRACKER_REID_MODEL`

### 6. Polygon Zone Counting

- Supports normalized polygon zones through `COUNT_ZONES`.
- Also accepts legacy rectangle-style `COUNT_ROIS` / `TRIGGER_ROIS` input for compatibility.
- Counts a track once per zone.
- Uses the detection bounding-box center point for polygon inclusion.
- Allows per-zone class filters.
- Maintains cumulative per-zone totals in memory for the running process.

Main config:

- `COUNT_ZONES`
- `COUNT_ROIS`
- `TRIGGER_ROIS`
- `COUNT_POLYGONS`
- `COUNT_ZONE_CLASS_IDS`

Important behavior:

- Zone IDs are 1-based and follow the order in `COUNT_ZONES`.
- If zones overlap, the same track can be counted once in each matching zone.
- Counts reset when the Python process restarts.

### 7. Zone/Class Highlighting

- Draws detections that match a count zone and its class rules in red.
- Applies the same class rules used for counting:
  - `COUNT_ZONE_CLASS_IDS` for zones with explicit filters
  - `DETECTION_CLASS_IDS` for zones without explicit filters

### 8. Live Preview Overlay

The MJPEG preview can draw:

- detection ROI
- count-zone polygons
- cumulative count labels
- detection boxes
- class label
- tracker ID
- confidence
- class ID
- `[ROI]` marker

Main config:

- `STREAM_PORT`
- `PREVIEW_FPS`
- `PREVIEW_JPEG_QUALITY`
- `BBOX_LINE_THICKNESS`

### 9. Browser ROI Editor

The browser ROI editor:

- runs at `/roi_editor`
- displays the current detection ROI
- lets the user draw polygon count zones
- exports a ready-to-use `COUNT_ZONES='...'` value
- stores draft editor state in browser `localStorage`

The editor is a helper UI. It does not write to the backend or mutate container environment variables by itself.

### 10. Structured Zone Visit Events

The service sends structured events when a tracked object finishes a zone visit or when a track becomes stale while still inside a zone.

Events are queued and sent in batches by `CounterRecordApiClient`, so network I/O does not block the main detection loop.

Batch payload shape:

```json
{
  "device_id": "DEVICE_ID",
  "container_id": "CONTAINER_OR_PROFILE_ID",
  "events": [
    {
      "event_type": "zone_visit",
      "severity": "high",
      "ts": "2026-05-13T10:00:01.240Z",
      "track_id": "12",
      "class_id": 2,
      "zone_id": 1,
      "confidence": 0.87,
      "entered_at": "2026-05-13T10:00:00.000Z",
      "exited_at": "2026-05-13T10:00:01.240Z",
      "duration_s": 1.24,
      "entry_bbox": [0.41, 0.32, 0.52, 0.48],
      "exit_bbox": [0.44, 0.35, 0.55, 0.51],
      "entry_center": { "x": 0.465, "y": 0.4 },
      "exit_center": { "x": 0.495, "y": 0.43 },
      "count_total_after_increment": 7,
      "zone_totals": { "1": 7, "2": 3 },
      "entry_roi": [[0.1, 0.2], [0.4, 0.2], [0.4, 0.5], [0.1, 0.5]],
      "zone_roi": [[0.1, 0.2], [0.4, 0.2], [0.4, 0.5], [0.1, 0.5]],
      "entry_photo_path": "entry_media/...",
      "entry_photo_bbox": [0.41, 0.32, 0.52, 0.48],
      "entry_photo_center": { "x": 0.465, "y": 0.4 },
      "entry_photo_upload": { "status": 200, "response": {} },
      "entry_photo_file_url": "/file/fs/..."
    }
  ]
}
```

Main config:

- `DEVICE_ID`
- `CONTAINER_ID`
- `COUNTER_RECORD_API_URL`
- `COUNTER_RECORD_API_USERNAME`
- `COUNTER_RECORD_API_PASSWORD`
- `COUNTER_RECORD_API_TIMEOUT_S`
- `COUNTER_RECORD_API_FLUSH_INTERVAL_S`
- `COUNTER_RECORD_API_BATCH_SIZE`
- `COUNTER_RECORD_API_QUEUE_SIZE`
- `EVENT_SEVERITY`

### 11. Entry Photo Capture

When a new zone visit starts, the service can optionally:

- save a raw entry JPEG immediately, before ROI polygons or bbox overlays are drawn
- upload the raw image to the file service
- attach local paths and upload responses to the later `zone_visit` event
- attach `entry_bbox` / `entry_center` and `entry_roi` / `zone_roi` in the event payload
- attach `entry_photo_bbox` / `entry_photo_center` as explicit aliases for the raw image

This feature is disabled by default.

Main config:

- `ENABLE_ENTRY_MEDIA_CAPTURE`
- `ENTRY_MEDIA_DIR`

### 12. File Upload API

File uploads use multipart form-data and Basic authentication.

Image upload:

- endpoint default: `{FILE_API_BASE_URL}/file/fs/{device_id}`
- form field: `upload`

The image upload request sends the raw image file only. Bbox and ROI metadata are sent in the structured `zone_visit` event payload.

Main config:

- `FILE_API_BASE_URL`
- `FILE_API_USERNAME`
- `FILE_API_PASSWORD`
- `FILE_API_TIMEOUT_S`
- `FILE_IMAGE_UPLOAD_URL`

Example environment:

```bash
ENABLE_ENTRY_MEDIA_CAPTURE=true

DEVICE_ID=DEVICE_ID
CONTAINER_ID=CONTAINER_OR_PROFILE_ID

COUNTER_RECORD_API_URL=http://localhost:3000/api/events
COUNTER_RECORD_API_USERNAME=odt
COUNTER_RECORD_API_PASSWORD=odt123456

FILE_API_BASE_URL=http://localhost:5001
FILE_API_USERNAME=odt
FILE_API_PASSWORD=odt123456
FILE_IMAGE_UPLOAD_URL=http://localhost:5001/file/fs/{device_id}
```

## What The Service Computes Internally

For each tracked detection:

- `track_id`
- exported API track ID
- `class_id`
- confidence
- bounding box
- normalized bounding box
- center point
- normalized center point
- zone hits
- whether the detection matches the zone/class rules

For each zone:

- running cumulative count total
- active visits by track
- per-track counted zone state

For media capture:

- entry photo path
- entry photo upload status/response when file upload is enabled
- entry bbox and entry ROI in the event payload

## Current Limitations

- Counts are process-local and reset on restart.
- There is no database persistence inside this Python service.
- There is no JSON HTTP endpoint for current counts or active tracks.
- The ROI editor exports config text but does not save config directly.
- The raw uploaded image does not contain drawn ROI polygons or bbox overlays.
- Video capture/upload is intentionally disabled in the current flow.
- Movement direction, turn inference, lane mapping, and origin/destination analytics are not implemented.

## Operational Notes

- The service is optimized for low latency rather than frame-perfect retention.
- Frame queues intentionally keep the freshest frame and drop stale ones.
- Most runtime behavior is concentrated in `main.py`.
- Detection/tracking logic is concentrated in `utils/openvinoDetector.py`.
- Event and file upload clients are in `utils/api_client.py`.
- Browser preview/editor code is in `service/flask_server.py`.
- Practical resource pressure comes from model runtime, decoded frame size, image uploads, and tracker state.

## Short Summary

The service currently can:

- read an RTSP stream
- decode through CPU/OpenCV, GStreamer/VAAPI-oriented, or NVIDIA FFmpeg paths
- run detection with OpenVINO or Torch
- filter object classes
- track objects
- count per polygon zone
- apply per-zone class filters
- highlight related zone/class detections
- stream live annotated preview
- provide a polygon ROI editor
- send structured `zone_visit` events to a backend
- optionally save and upload raw entry photos

The service does not currently:

- persist counts in a database
- expose current counts/active tracks as local JSON endpoints
- infer turn direction or movement paths between zones
- capture or upload videos
