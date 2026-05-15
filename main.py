#!/usr/bin/env python3
# main.py
import logging
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from urllib.parse import urlparse, urlunparse

import cv2
import numpy as np

from config.config import Config
from service.flask_server import frame_queue, run_flask
from utils.api_client import CounterRecordApiClient, FileUploadApiClient
from utils.capture import capture_thread_func
from utils.openvinoDetector import (
    AsyncVehicleDetector,
    RTSPConnection,
    bbox_center,
    draw_count_polygons,
    draw_detections,
    draw_roi,
    get_polygon_coordinates_multi,
    get_roi_coordinates,
    is_bbox_in_polygon,
)

logging.info("\nOpenCV build info:\n%s", cv2.getBuildInformation())
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def redact_rtsp(url:str) -> str:
    try:
        u = urlparse(url)
        if u.username is not None:
            if u.password is not None:
                userinfo = f"{u.username}:****"
            else:
                userinfo = f"{u.username}"
            hostpart = u.hostname or ""
            if u.port:
                hostpart += f":{u.port}"
            netloc = f"{userinfo}@{hostpart}"
        else:
            netloc = u.netloc
        return urlunparse((u.scheme, netloc, u.path, u.params, u.query, u.fragment))
    except Exception:
        return "<invalid-rtsp-url>"

def log_rtsp_details(label: str, url: str):
    redacted = redact_rtsp(url)
    try:
        u = urlparse(url)
    except Exception:
        u = None
    logging.info(f"🎬 {label} RTSP URL: {redacted}")
    if u:
        if u.username:
            logging.info(f"{label} username: {u.username}")
        if u.password:
            logging.info(f"{label} password: ****")
        if u.hostname:
            logging.info(f"{label} host: {u.hostname}{f':{u.port}' if u.port else ''}")

def connect_with_retries(conn: RTSPConnection, label: str, retries: int, retry_delay: float) -> bool:
    attempt = 0
    while attempt < retries:
        if conn.connect():
            if conn.cap:
                conn.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logging.info(f"✅ Connected {label} stream")
            return True
        attempt += 1
        logging.warning(f"{label} stream connect failed. Retry {attempt}/{retries}")
        time.sleep(retry_delay)
    logging.error(f"❌ Failed to establish {label} stream after retries")
    return False

def compute_motion_signature(roi_frame, motion_downscale: float):
    if roi_frame.size == 0:
        return None
    motion_frame = roi_frame
    if 0 < motion_downscale < 1.0:
        motion_frame = cv2.resize(
            roi_frame,
            None,
            fx=motion_downscale,
            fy=motion_downscale,
            interpolation=cv2.INTER_AREA,
        )
    gray_roi = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray_roi, (9, 9), 0)


def is_normalized_roi_valid(roi) -> bool:
    if not roi or len(roi) != 4:
        return False
    x1, y1, x2, y2 = roi
    return 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1


def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_bbox(frame_shape, bbox):
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = bbox
    return [
        round(float(x1) / float(width), 6),
        round(float(y1) / float(height), 6),
        round(float(x2) / float(width), 6),
        round(float(y2) / float(height), 6),
    ]


def normalize_center(frame_shape, center):
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return {"x": 0.0, "y": 0.0}
    cx, cy = center
    return {
        "x": round(float(cx) / float(width), 6),
        "y": round(float(cy) / float(height), 6),
    }


def normalize_roi_value(roi):
    return [
        [round(float(x), 6), round(float(y), 6)]
        for x, y in (roi or ())
    ]


def export_track_id(track_id, prefix: str) -> str:
    text = str(track_id or "").strip()
    if not text:
        return ""
    normalized_prefix = f"{prefix}_"
    if prefix and text.startswith(normalized_prefix):
        return text[len(normalized_prefix):]
    return text


def snapshot_zone_totals(roi_counts):
    return {
        str(idx + 1): int(total)
        for idx, total in enumerate(roi_counts)
    }


def open_zone_visit(
    *,
    zone_id: int,
    track_id: str,
    class_id,
    entered_at_ts: float,
    confidence,
    severity: str,
    normalized_bbox,
    normalized_center,
    normalized_roi,
    count_total_after_increment: int,
):
    return {
        "zone_id": int(zone_id),
        "track_id": track_id,
        "class_id": class_id,
        "confidence": round(float(confidence), 6) if confidence is not None else None,
        "severity": str(severity or ""),
        "entered_at_ts": float(entered_at_ts),
        "last_seen_ts": float(entered_at_ts),
        "entry_bbox": normalized_bbox,
        "entry_center": normalized_center,
        "entry_roi": normalized_roi,
        "zone_roi": normalized_roi,
        "exit_bbox": normalized_bbox,
        "exit_center": normalized_center,
        "count_total_after_increment": int(count_total_after_increment),
    }


def close_zone_visit(visit: dict, end_ts: float, zone_totals: dict):
    duration_s = max(0.0, float(end_ts) - float(visit["entered_at_ts"]))
    media_info = visit.get("entry_media") or visit
    return {
        "event_type": "zone_visit",
        "severity": visit.get("severity"),
        "ts": iso_utc(end_ts),
        "track_id": visit["track_id"],
        "class_id": visit.get("class_id"),
        "zone_id": int(visit["zone_id"]),
        "confidence": visit.get("confidence"),
        "entered_at": iso_utc(visit["entered_at_ts"]),
        "exited_at": iso_utc(end_ts),
        "duration_s": round(duration_s, 3),
        "entry_bbox": visit.get("entry_bbox"),
        "entry_roi": visit.get("entry_roi"),
        "zone_roi": visit.get("zone_roi"),
        "exit_bbox": visit.get("exit_bbox"),
        "entry_center": visit.get("entry_center"),
        "exit_center": visit.get("exit_center"),
        "count_total_after_increment": int(visit["count_total_after_increment"]),
        "zone_totals": dict(zone_totals),
        "entry_photo_path": media_info.get("entry_photo_path"),
        "entry_photo_bbox": visit.get("entry_bbox"),
        "entry_photo_center": visit.get("entry_center"),
        "entry_photo_upload": media_info.get("entry_photo_upload"),
        "entry_photo_file_url": media_info.get("entry_photo_file_url"),
    }


def safe_media_token(value) -> str:
    text = str(value if value is not None else "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "unknown"


class EntryMediaCapture:
    def __init__(self, config, file_upload_client: FileUploadApiClient | None = None):
        self.enabled = bool(getattr(config, "ENABLE_ENTRY_MEDIA_CAPTURE", False))
        self.media_dir = Path(getattr(config, "ENTRY_MEDIA_DIR", "entry_media") or "entry_media")
        self.file_upload_client = file_upload_client
        if self.enabled:
            self.media_dir.mkdir(parents=True, exist_ok=True)
            logging.info("Entry image capture enabled: dir=%s", self.media_dir)

    def start_entry_capture(
        self,
        frame,
        *,
        track_id,
        zone_id,
        class_id,
        entered_at_ts: float,
    ):
        if not self.enabled:
            return {}
        timestamp = datetime.fromtimestamp(float(entered_at_ts), timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        basename = (
            f"{timestamp}_track-{safe_media_token(track_id)}"
            f"_zone-{safe_media_token(zone_id)}_class-{safe_media_token(class_id)}"
        )
        photo_path = self.media_dir / f"{basename}.jpg"
        media_info = {}
        ok = cv2.imwrite(str(photo_path), frame)
        if ok:
            logging.info("Entry photo saved: %s", photo_path)
            media_info["entry_photo_path"] = str(photo_path)
            try:
                upload_result = (
                    self.file_upload_client.upload_image(str(photo_path))
                    if self.file_upload_client
                    else None
                )
            except Exception as exc:
                logging.error("Entry photo upload failed: %s", exc)
                upload_result = {"error": str(exc)}
            if upload_result is not None:
                media_info["entry_photo_upload"] = upload_result
                if self.file_upload_client:
                    media_info["entry_photo_file_url"] = self.file_upload_client.image_file_url_from_upload(upload_result)
        else:
            logging.error("Entry photo save failed: %s", photo_path)
            photo_path = None

        if photo_path is None:
            media_info["entry_photo_path"] = None
        media_info["entry_media"] = media_info
        return media_info

    def update(self, frame, now_ts: float):
        return

    def close(self):
        return

def main():
    config = Config()
    count_class_ids = set(getattr(config, "DETECTION_CLASS_IDS", ()))
    count_zone_class_map = {
        int(zone_id): set(class_ids)
        for zone_id, class_ids in getattr(config, "COUNT_ZONE_CLASS_IDS", {}).items()
    }
    track_state = defaultdict(
        lambda: {
            "counted_zones": set(),
            "active_visits": {},
        }
    )

    tracking_rtsp_url = config.TRACK_RTSP_URL

    log_rtsp_details("Tracking", tracking_rtsp_url)
    logging.info(f"Tracking stream target FPS: {config.TRACK_RTSP_FPS}")
    logging.info(f"Configured count zones: {config.COUNT_ZONES}")
    logging.info(
        "Model runtime: backend=%s device=%s",
        getattr(config, "MODEL_BACKEND", "openvino"),
        getattr(config, "MODEL_DEVICE", "CPU"),
    )
    logging.info(
        "Tracker runtime: backend=%s",
        getattr(config, "TRACKER_BACKEND", "custom"),
    )
    logging.info(
        "Counter record API: enabled=%s device_id=%s container_id=%s endpoint=%s",
        bool(
            getattr(config, "COUNTER_RECORD_API_URL", "")
            and getattr(config, "COUNTER_RECORD_API_USERNAME", "")
            and getattr(config, "COUNTER_RECORD_API_PASSWORD", "")
            and getattr(config, "DEVICE_ID", "")
            and getattr(config, "CONTAINER_ID", "")
        ),
        getattr(config, "DEVICE_ID", ""),
        getattr(config, "CONTAINER_ID", ""),
        getattr(config, "COUNTER_RECORD_API_URL", ""),
    )
    if (
        getattr(config, "_COUNTER_RECORD_API_URL_FROM_LEGACY", False)
        or getattr(config, "_COUNTER_RECORD_API_USERNAME_FROM_LEGACY", False)
        or getattr(config, "_COUNTER_RECORD_API_PASSWORD_FROM_LEGACY", False)
    ):
        logging.warning(
            "Counter record API is using legacy fallback envs: "
            "url_from_legacy=%s username_from_legacy=%s password_from_legacy=%s. "
            "Prefer COUNTER_RECORD_API_URL / COUNTER_RECORD_API_USERNAME / COUNTER_RECORD_API_PASSWORD.",
            getattr(config, "_COUNTER_RECORD_API_URL_FROM_LEGACY", False),
            getattr(config, "_COUNTER_RECORD_API_USERNAME_FROM_LEGACY", False),
            getattr(config, "_COUNTER_RECORD_API_PASSWORD_FROM_LEGACY", False),
        )
    if config.ENABLE_FAR_DETECTION and config.FAR_DETECTION_ROI:
        logging.info(f"Configured far detection ROI: {config.FAR_DETECTION_ROI}")
    else:
        logging.info("Configured far detection ROI: disabled")
    logging.info(
        "CPU tuning: motion_check_interval=%.2fs motion_downscale=%.2f preview_fps=%.2f",
        config.MOTION_CHECK_INTERVAL_S,
        config.MOTION_DOWNSCALE,
        config.PREVIEW_FPS,
    )

    if not is_normalized_roi_valid(config.DETECTION_ROI):
        logging.error("❌ Invalid DETECTION_ROI values. Must be in [0,1] with x1<x2, y1<y2.")
        return
    if config.ENABLE_FAR_DETECTION and config.FAR_DETECTION_ROI and not is_normalized_roi_valid(config.FAR_DETECTION_ROI):
        logging.error("❌ Invalid FAR_DETECTION_ROI values. Must be in [0,1] with x1<x2, y1<y2.")
        return

    logging.info(f"Using DETECTION_ROI: {config.DETECTION_ROI}")
    print("🚀 Initializing Vehicle Detection with RTSP...")

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logging.info(f"Streaming server started on port {config.STREAM_PORT}")

    tracking_conn = RTSPConnection(
        rtsp_url=tracking_rtsp_url,
        max_retries=config.RECONNECT_ATTEMPTS,
        retry_delay=config.RECONNECT_DELAY,
        connection_timeout=15,
        config=config,
        fps=config.TRACK_RTSP_FPS,
    )

    stream_healthcheck_interval_s = max(0.1, config.STREAM_HEALTHCHECK_INTERVAL_S)
    motion_check_interval_s = max(0.0, config.MOTION_CHECK_INTERVAL_S)
    motion_downscale = min(max(config.MOTION_DOWNSCALE, 0.1), 1.0)
    preview_interval_s = 0.0 if config.PREVIEW_FPS <= 0 else (1.0 / max(config.PREVIEW_FPS, 0.1))

    if not connect_with_retries(
        tracking_conn,
        "tracking",
        config.RECONNECT_ATTEMPTS,
        config.RECONNECT_DELAY,
    ):
        return

    detector = AsyncVehicleDetector()
    detector.start_processing()
    file_upload_client = FileUploadApiClient(
        image_endpoint=config.FILE_IMAGE_UPLOAD_URL,
        video_endpoint="",
        username=config.FILE_API_USERNAME,
        password=config.FILE_API_PASSWORD,
        device_id=config.DEVICE_ID,
        timeout_s=config.FILE_API_TIMEOUT_S,
        video_profile_type="manual",
    )
    entry_media_capture = EntryMediaCapture(config, file_upload_client=file_upload_client)
    api_client = CounterRecordApiClient(
        endpoint=config.COUNTER_RECORD_API_URL,
        username=config.COUNTER_RECORD_API_USERNAME,
        password=config.COUNTER_RECORD_API_PASSWORD,
        device_id=config.DEVICE_ID,
        container_id=config.CONTAINER_ID,
        timeout_s=config.COUNTER_RECORD_API_TIMEOUT_S,
        flush_interval_s=config.COUNTER_RECORD_API_FLUSH_INTERVAL_S,
        batch_size=config.COUNTER_RECORD_API_BATCH_SIZE,
        queue_size=config.COUNTER_RECORD_API_QUEUE_SIZE,
    )
    api_client.start()

    tracking_frame_queue = Queue(maxsize=1)
    stop_event = threading.Event()
    tracking_capture_thread = threading.Thread(
        target=capture_thread_func,
        args=(tracking_conn, tracking_frame_queue, stop_event),
        daemon=True,
    )
    tracking_capture_thread.start()

    frame_count = 0
    last_process_time = time.time()
    fps_limit_delay = 1.0 / config.MAX_FPS

    process_times = []
    connection_stats = {
        'tracking_reconnections': 0,
        'frames_processed': 0,
        'tracking_uptime': time.time(),
    }

    prev_gray_roi = None
    motion_detected = False
    roi_counts = [0 for _ in config.COUNT_ZONES]
    last_motion_check_ts = 0.0
    last_preview_push_ts = 0.0
    last_healthcheck_ts = 0.0
    last_frame_wait_log_ts = 0.0

    def get_count_zones_px(frame_shape):
        return get_polygon_coordinates_multi(frame_shape, config.COUNT_ZONES)

    try:
        while True:
            loop_now = time.time()
            if loop_now - last_healthcheck_ts >= stream_healthcheck_interval_s:
                last_healthcheck_ts = loop_now

                # Health check & reconnect (tracking stream is mandatory)
                if not tracking_conn.check_health():
                    logging.warning("⚠️ Tracking stream health check failed")
                    tracking_conn.is_connected = False

                if not tracking_conn.is_connected:
                    logging.info("🔄 Attempting to reconnect tracking stream...")
                    connection_stats['tracking_reconnections'] += 1
                    connection_stats['tracking_uptime'] = time.time()
                    if not tracking_conn.reconnect():
                        logging.error(
                            "❌ Tracking stream reconnection failed; will retry in %.1fs.",
                            stream_healthcheck_interval_s,
                        )
                        continue
                    if tracking_conn.cap:
                        tracking_conn.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    logging.info("✅ Tracking stream reconnected")

            try:
                frame = tracking_frame_queue.get(timeout=0.5)
            except Empty:
                if loop_now - last_frame_wait_log_ts >= 5.0:
                    last_frame_wait_log_ts = loop_now
                    logging.info(
                        "Waiting for tracking frames... connected=%s cap_open=%s",
                        tracking_conn.is_connected,
                        tracking_conn.check_health(),
                    )
                logging.debug("No tracking frame available, skipping iteration")
                continue
            loop_now = time.time()

            frame_count += 1
            connection_stats['frames_processed'] = frame_count
            entry_media_capture.update(frame, loop_now)

            roi_coords = get_roi_coordinates(frame.shape, config.DETECTION_ROI)
            x1r, y1r, x2r, y2r = roi_coords
            roi_frame = frame[y1r:y2r, x1r:x2r]
            should_check_motion = (
                prev_gray_roi is None or
                motion_check_interval_s == 0.0 or
                (loop_now - last_motion_check_ts) >= motion_check_interval_s
            )
            if should_check_motion:
                had_motion_baseline = prev_gray_roi is not None
                gray_blur_roi = compute_motion_signature(roi_frame, motion_downscale)
                if gray_blur_roi is None:
                    motion_detected = True
                elif prev_gray_roi is not None:
                    frame_delta = cv2.absdiff(prev_gray_roi, gray_blur_roi)
                    thresh_delta = cv2.threshold(frame_delta, config.DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
                    changed_pixels = np.count_nonzero(thresh_delta)
                    total_pixels = thresh_delta.shape[0] * thresh_delta.shape[1]
                    motion_ratio = (changed_pixels / total_pixels) if total_pixels else 0.0
                    motion_detected = motion_ratio > config.PIXEL_CHANGE_THRESHOLD
                    logging.debug(
                        f"Frame {frame_count}: Motion in ROI: {motion_detected} "
                        f"({motion_ratio:.4f} changed, scale={motion_downscale:.2f})"
                    )
                else:
                    motion_detected = True
                prev_gray_roi = gray_blur_roi.copy() if gray_blur_roi is not None else None
                last_motion_check_ts = loop_now

            processed_frame = frame
            det_roi_coords = roi_coords
            detections = []
            count_zones_px = get_count_zones_px(processed_frame.shape)

            if (frame_count % config.SKIP_FRAMES == 0) and motion_detected:
                try:
                    detector.add_frame(frame, roi_coords)
                except Exception as e:
                    logging.error(f"detector.add_frame error: {e}")

                latest = None
                try:
                    while True:
                        r = detector.get_result()
                        if not r:
                            break
                        latest = r
                except Exception as e:
                    logging.error(f"detector.get_result error: {e}")

                if latest:
                    start_time = time.time()
                    processed_frame, detections, det_roi_coords = latest
                    count_zones_px = get_count_zones_px(processed_frame.shape)
                    frame_events = []
                    seen_detection_track_ids = set()
                    seen_visit_zone_ids_by_track = defaultdict(set)

                    for detection in detections:
                        track_id = detection.get("track_id", -1)
                        seen_detection_track_ids.add(track_id)
                        api_track_id = export_track_id(track_id, config.TRACK_ID_PREFIX)
                        bbox = detection["bbox"]
                        confidence = detection['confidence']
                        class_id = detection.get("class_id")
                        track_entry = track_state[track_id]
                        center = bbox_center(bbox)
                        normalized_bbox = normalize_bbox(processed_frame.shape, bbox)
                        normalized_center = normalize_center(processed_frame.shape, center)
                        zones_hit = [
                            zone_idx
                            for zone_idx, count_zone in enumerate(count_zones_px)
                            if is_bbox_in_polygon(bbox, count_zone)
                        ]
                        count_zones_hit = []
                        for zone_idx in zones_hit:
                            zone_id = zone_idx + 1
                            allowed_count_classes = count_zone_class_map.get(zone_id)
                            if allowed_count_classes is None:
                                if count_class_ids and class_id not in count_class_ids:
                                    continue
                            elif class_id not in allowed_count_classes:
                                continue
                            count_zones_hit.append(zone_idx)
                        detection["zone_class_match"] = bool(count_zones_hit)

                        if confidence < config.MIN_CONFIDENCE:
                            continue

                        for zone_idx in count_zones_hit:
                            zone_id = zone_idx + 1
                            seen_visit_zone_ids_by_track[track_id].add(zone_id)
                            visit = track_entry["active_visits"].get(zone_id)
                            if zone_idx in track_entry["counted_zones"]:
                                if visit is None:
                                    continue
                            else:
                                track_entry["counted_zones"].add(zone_idx)
                                roi_counts[zone_idx] += 1
                                logging.info(
                                    "🚗 Count zone #%d incremented to %d by track %s",
                                    zone_id,
                                    roi_counts[zone_idx],
                                    track_id,
                                )
                                visit = open_zone_visit(
                                    zone_id=zone_id,
                                    track_id=api_track_id,
                                    class_id=class_id,
                                    entered_at_ts=loop_now,
                                    confidence=confidence,
                                    severity=config.EVENT_SEVERITY,
                                    normalized_bbox=normalized_bbox,
                                    normalized_center=normalized_center,
                                    normalized_roi=normalize_roi_value(config.COUNT_ZONES[zone_idx]),
                                    count_total_after_increment=roi_counts[zone_idx],
                                )
                                visit.update(
                                    entry_media_capture.start_entry_capture(
                                        # Capture the clean detector frame before preview overlays are drawn.
                                        processed_frame.copy(),
                                        track_id=api_track_id,
                                        zone_id=zone_id,
                                        class_id=class_id,
                                        entered_at_ts=loop_now,
                                    )
                                )
                                track_entry["active_visits"][zone_id] = visit

                            visit["last_seen_ts"] = loop_now
                            visit["class_id"] = class_id
                            visit["confidence"] = round(float(confidence), 6) if confidence is not None else None
                            visit["exit_bbox"] = normalized_bbox
                            visit["exit_center"] = normalized_center

                    tracker = detector.get_tracker()
                    if hasattr(tracker, "get_active_track_ids"):
                        active_track_ids = tracker.get_active_track_ids()
                    else:
                        active_track_ids = {track["id"] for track in tracker.tracks}
                    zone_totals_snapshot = snapshot_zone_totals(roi_counts)
                    for active_track_id in list(active_track_ids):
                        if active_track_id not in seen_detection_track_ids:
                            continue
                        track_entry = track_state.get(active_track_id)
                        if not track_entry:
                            continue
                        seen_zone_ids = seen_visit_zone_ids_by_track.get(active_track_id, set())
                        for zone_id in list(track_entry["active_visits"].keys()):
                            if zone_id not in seen_zone_ids:
                                visit = track_entry["active_visits"].pop(zone_id)
                                frame_events.append(close_zone_visit(visit, loop_now, zone_totals_snapshot))
                    for stale_track_id in list(track_state.keys()):
                        if stale_track_id not in active_track_ids:
                            stale_entry = track_state.get(stale_track_id) or {}
                            for zone_id, visit in list((stale_entry.get("active_visits") or {}).items()):
                                frame_events.append(close_zone_visit(visit, loop_now, zone_totals_snapshot))
                            track_state.pop(stale_track_id, None)
                    if frame_events:
                        api_client.enqueue_events(frame_events)

                    process_time = time.time() - start_time
                    process_times.append(process_time)
                    if len(process_times) > 100:
                        process_times.pop(0)
                    if detections:
                        logging.info(
                            "Frame %d: Found %d detection(s) (Process time: %.3fs)",
                            frame_count,
                            len(detections),
                            process_time,
                        )

            draw_roi(processed_frame, det_roi_coords)
            if count_zones_px:
                draw_count_polygons(processed_frame, count_zones_px, roi_counts)
            if detections:
                draw_detections(processed_frame, detections, det_roi_coords)

            if preview_interval_s == 0.0 or (loop_now - last_preview_push_ts) >= preview_interval_s:
                try:
                    while frame_queue.qsize() > 0:
                        frame_queue.get_nowait()
                    frame_queue.put(processed_frame.copy(), block=False)
                    last_preview_push_ts = loop_now
                except Full:
                    logging.warning("Frame queue full, dropping frame")

            elapsed = time.time() - last_process_time
            target_delay = fps_limit_delay - elapsed
            if target_delay > 0:
                time.sleep(min(target_delay, 0.005))
            last_process_time = time.time()

            if config.ENABLE_LOCAL_KEYBOARD_EXIT:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

    except KeyboardInterrupt:
        logging.info("\n🛑 Interrupted by user")

    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        logging.info("🧹 Cleaning up...")
        stop_event.set()
        tracking_capture_thread.join()
        detector.stop()
        api_client.stop()
        entry_media_capture.close()
        tracking_conn.close()
        cv2.destroyAllWindows()

        if process_times:
            logging.info(f"\n📊 Performance Stats:")
            logging.info(f"Average process time: {np.mean(process_times):.3f}s")
            logging.info(f"Min process time: {np.min(process_times):.3f}s")
            logging.info(f"Max process time: {np.max(process_times):.3f}s")

        logging.info(f"\n🌐 Connection Stats:")
        logging.info(f"Total frames processed: {connection_stats['frames_processed']}")
        logging.info(f"Tracking reconnections: {connection_stats['tracking_reconnections']}")
        logging.info(f"Tracking uptime: {time.time() - connection_stats['tracking_uptime']:.0f}s")

if __name__ == "__main__":
    main()
