import base64
import json
import logging
import mimetypes
import os
import threading
import time
from queue import Empty, Full, Queue
from urllib import error, parse, request


class CounterRecordApiClient:
    def __init__(
        self,
        endpoint: str,
        username: str,
        password: str,
        device_id: str,
        container_id: str,
        timeout_s: float = 10.0,
        flush_interval_s: float = 0.5,
        batch_size: int = 50,
        queue_size: int = 1000,
    ):
        self.endpoint = str(endpoint or "").strip()
        self.username = str(username or "")
        self.password = str(password or "")
        self.device_id = str(device_id or "").strip()
        self.container_id = str(container_id or "").strip()
        self.timeout_s = max(float(timeout_s), 0.1)
        self.flush_interval_s = max(float(flush_interval_s), 0.1)
        self.batch_size = max(int(batch_size), 1)
        self.queue = Queue(maxsize=max(int(queue_size), 1))
        self.stop_event = threading.Event()
        self.worker = None
        self.enabled = bool(self.endpoint and self.username and self.password and self.device_id and self.container_id)
        if not self.enabled:
            logging.info(
                "Counter record API client disabled: endpoint=%s device_id=%s container_id=%s auth=%s",
                bool(self.endpoint),
                bool(self.device_id),
                bool(self.container_id),
                bool(self.username and self.password),
            )
        else:
            logging.info(
                "Counter record API client enabled: endpoint=%s auth_username=%s device_id=%s container_id=%s batch_size=%d flush_interval=%.2fs",
                self.endpoint,
                self.username,
                self.device_id,
                self.container_id,
                self.batch_size,
                self.flush_interval_s,
            )

    def start(self):
        if not self.enabled or self.worker is not None:
            return
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        if self.worker:
            self.worker.join(timeout=max(self.timeout_s, 2.0))

    def enqueue_events(self, events):
        if not self.enabled or not events:
            return
        logging.info("Counter record API enqueueing %d event(s)", len(events))
        for event in events:
            try:
                self.queue.put_nowait(event)
            except Full:
                logging.warning("Counter record API queue is full; dropping event %s", event.get("event_type"))

    def _authorization_header(self) -> str:
        token = base64.b64encode(f"{self.username}:{self.password}".encode("utf-8")).decode("ascii")
        return f"Basic {token}"

    @staticmethod
    def _mask_header_value(value: str) -> str:
        text = str(value or "")
        if len(text) <= 16:
            return text
        return f"{text[:12]}...{text[-4:]}"

    def _payload(self, events):
        return {
            "device_id": self.device_id,
            "container_id": self.container_id,
            "events": events,
        }

    def _post_events(self, events):
        payload = self._payload(events)
        body_text = json.dumps(payload)
        headers = {
            "Content-Type": "application/json",
            "Authorization": self._authorization_header(),
        }
        logged_headers = {
            key: (self._mask_header_value(value) if key.lower() == "authorization" else value)
            for key, value in headers.items()
        }
        logging.info("Counter record API request URL: %s", self.endpoint)
        logging.info("Counter record API auth username: %s", self.username)
        logging.info("Counter record API request headers: %s", json.dumps(logged_headers))
        logging.info("Counter record API request body: %s", body_text)
        body = body_text.encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers=headers,
        )
        with request.urlopen(req, timeout=self.timeout_s) as resp:
            resp.read()
            status_code = getattr(resp, "status", None) or resp.getcode()
        logging.info("Counter record API sent %d event(s), status=%s", len(events), status_code)

    def _drain_batch(self, first_event):
        batch = [first_event]
        deadline = time.time() + self.flush_interval_s
        while len(batch) < self.batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                batch.append(self.queue.get(timeout=remaining))
            except Empty:
                break
        return batch

    def _run(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                first_event = self.queue.get(timeout=self.flush_interval_s)
            except Empty:
                continue
            batch = self._drain_batch(first_event)
            try:
                self._post_events(batch)
            except error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="ignore")
                logging.error("Counter record API HTTP error %s: %s", exc.code, details)
            except Exception as exc:
                logging.error("Counter record API send failed: %s", exc)


class FileUploadApiClient:
    def __init__(
        self,
        *,
        image_endpoint: str,
        video_endpoint: str,
        username: str,
        password: str,
        device_id: str,
        timeout_s: float = 30.0,
        video_profile_type: str = "manual",
    ):
        self.image_endpoint = str(image_endpoint or "").strip()
        self.video_endpoint = str(video_endpoint or "").strip()
        self.username = str(username or "")
        self.password = str(password or "")
        self.device_id = str(device_id or "").strip()
        self.timeout_s = max(float(timeout_s), 0.1)
        self.video_profile_type = str(video_profile_type or "manual")
        self.enabled = bool(
            self.device_id
            and self.username
            and self.password
            and (self.image_endpoint or self.video_endpoint)
        )
        if not self.enabled:
            logging.info(
                "File upload API disabled: image_endpoint=%s video_endpoint=%s device_id=%s auth=%s",
                bool(self.image_endpoint),
                bool(self.video_endpoint),
                bool(self.device_id),
                bool(self.username and self.password),
            )
        else:
            logging.info(
                "File upload API enabled: image_endpoint=%s video_endpoint=%s device_id=%s",
                self.image_endpoint,
                self.video_endpoint,
                self.device_id,
            )

    def _authorization_header(self) -> str:
        token = base64.b64encode(f"{self.username}:{self.password}".encode("utf-8")).decode("ascii")
        return f"Basic {token}"

    def _format_endpoint(self, endpoint: str) -> str:
        return (
            str(endpoint or "")
            .replace("{{device_id}}", self.device_id)
            .replace("{device_id}", self.device_id)
        )

    @staticmethod
    def _guess_content_type(path: str) -> str:
        content_type, _ = mimetypes.guess_type(path)
        return content_type or "application/octet-stream"

    @staticmethod
    def _json_or_text(body: bytes):
        text = body.decode("utf-8", errors="ignore")
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    @staticmethod
    def _find_response_value(value, keys):
        if isinstance(value, dict):
            for key in keys:
                if key in value and value[key]:
                    return value[key]
            for nested in value.values():
                found = FileUploadApiClient._find_response_value(nested, keys)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = FileUploadApiClient._find_response_value(item, keys)
                if found:
                    return found
        return None

    def _endpoint_origin(self, endpoint: str) -> str:
        parsed = parse.urlparse(self._format_endpoint(endpoint))
        if not parsed.scheme or not parsed.netloc:
            return ""
        return parse.urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))

    def _absolute_url(self, endpoint: str, url: str):
        text = str(url or "").strip()
        if not text:
            return None
        if text.startswith(("http://", "https://")):
            return text
        if text.startswith("/"):
            origin = self._endpoint_origin(endpoint)
            return f"{origin}{text}" if origin else text
        return text

    def _extract_file_url(self, endpoint: str, upload_result):
        response = (upload_result or {}).get("response")
        url = self._find_response_value(response, ("fileUrl", "fileURL", "url", "downloadUrl", "downloadURL"))
        if url:
            return self._absolute_url(endpoint, url)
        if isinstance(response, str) and response.startswith(("/", "http://", "https://")):
            return self._absolute_url(endpoint, response)
        return None

    def video_file_url_from_upload(self, upload_result):
        file_url = self._extract_file_url(self.video_endpoint, upload_result)
        if file_url:
            return file_url

        response = (upload_result or {}).get("response")
        file_id = self._find_response_value(response, ("id", "_id", "fileId", "videoId"))
        if not file_id:
            return None
        origin = self._endpoint_origin(self.video_endpoint)
        encoded_id = parse.quote(str(file_id), safe="")
        return f"{origin}/file/videos/{encoded_id}/file" if origin else f"/file/videos/{encoded_id}/file"

    def image_file_url_from_upload(self, upload_result):
        return self._extract_file_url(self.image_endpoint, upload_result)

    def _post_multipart(self, url: str, fields: dict, files: dict):
        boundary = f"----perimeter-ai-{int(time.time() * 1000000)}"
        body_parts = []
        for name, value in fields.items():
            body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
            body_parts.append(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")
            )
        for field_name, file_path in files.items():
            filename = os.path.basename(file_path)
            content_type = self._guess_content_type(file_path)
            body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
            body_parts.append(
                (
                    f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode("utf-8")
            )
            with open(file_path, "rb") as fp:
                body_parts.append(fp.read())
            body_parts.append(b"\r\n")
        body_parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(body_parts)
        req = request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": self._authorization_header(),
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
        )
        with request.urlopen(req, timeout=self.timeout_s) as resp:
            response_body = resp.read()
            status_code = getattr(resp, "status", None) or resp.getcode()
        return {
            "status": status_code,
            "response": self._json_or_text(response_body),
        }

    def upload_image(self, image_path: str, fields: dict | None = None):
        if not self.enabled or not self.image_endpoint or not image_path:
            return None
        url = self._format_endpoint(self.image_endpoint)
        logging.info("Uploading entry image: %s -> %s", image_path, url)
        result = self._post_multipart(url, fields=fields or {}, files={"upload": image_path})
        logging.info(
            "Entry image upload finished: status=%s response=%s",
            result.get("status"),
            json.dumps(result.get("response"), ensure_ascii=True),
        )
        return result

    def upload_video(self, video_path: str, fields: dict | None = None):
        if not self.enabled or not self.video_endpoint or not video_path:
            return None
        url = self._format_endpoint(self.video_endpoint)
        logging.info("Uploading entry video: %s -> %s", video_path, url)
        upload_fields = {"profileType": self.video_profile_type}
        upload_fields.update(fields or {})
        result = self._post_multipart(
            url,
            fields=upload_fields,
            files={"video": video_path},
        )
        logging.info(
            "Entry video upload finished: status=%s response=%s",
            result.get("status"),
            json.dumps(result.get("response"), ensure_ascii=True),
        )
        return result
