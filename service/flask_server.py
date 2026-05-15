# flask_server.py
from flask import Flask, Response, render_template_string
from flask_cors import CORS
from queue import Queue
import cv2
from config.config import Config

app = Flask(__name__)
CORS(app)
frame_queue = Queue(maxsize=30)  # Increased for more buffering
PREVIEW_JPEG_QUALITY = max(1, min(100, int(getattr(Config, "PREVIEW_JPEG_QUALITY", 75))))

ROI_EDITOR_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ROI Editor</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #11161b;
      --panel: #1a222b;
      --panel-2: #202b36;
      --line: #31404d;
      --text: #e7edf3;
      --muted: #98a6b5;
      --accent: #56d4ff;
      --danger: #ff7c6b;
      --ok: #81f495;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #0d1217 0%, #131a21 100%);
      color: var(--text);
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(480px, 1fr) 380px;
      min-height: 100vh;
    }
    .viewer {
      padding: 18px;
    }
    .stage {
      position: relative;
      width: 100%;
      background: #06090d;
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.35);
    }
    .stage img {
      display: block;
      width: 100%;
      height: auto;
      user-select: none;
      -webkit-user-drag: none;
    }
    .stage canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      cursor: crosshair;
    }
    .sidebar {
      border-left: 1px solid var(--line);
      background: rgba(15, 21, 27, 0.95);
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }
    h1, h2 {
      margin: 0 0 10px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    h1 { font-size: 22px; }
    h2 { font-size: 15px; color: var(--muted); text-transform: uppercase; }
    p, li {
      color: var(--muted);
      line-height: 1.45;
      margin: 0;
      font-size: 14px;
    }
    .stack {
      display: grid;
      gap: 10px;
    }
    .button-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    button {
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      padding: 10px 12px;
      border-radius: 10px;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover {
      border-color: var(--accent);
    }
    button.danger:hover {
      border-color: var(--danger);
    }
    button.ok:hover {
      border-color: var(--ok);
    }
    textarea {
      width: 100%;
      min-height: 130px;
      resize: vertical;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #0d1319;
      color: var(--text);
      padding: 12px;
      font: 13px/1.45 "Consolas", "SFMono-Regular", monospace;
    }
    .status {
      font: 13px/1.5 "Consolas", "SFMono-Regular", monospace;
      color: var(--text);
      white-space: pre-wrap;
    }
    .inline-note {
      font-size: 13px;
      color: var(--muted);
    }
    .links {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .links a {
      color: var(--accent);
      text-decoration: none;
      font-size: 13px;
    }
    @media (max-width: 1100px) {
      .layout {
        grid-template-columns: 1fr;
      }
      .sidebar {
        border-left: 0;
        border-top: 1px solid var(--line);
      }
    }
  </style>
</head>
<body>
  <div class="layout">
    <div class="viewer">
      <div class="stage" id="stage">
        <img id="videoFeed" src="{{ video_feed_url }}" alt="Live video feed">
        <canvas id="overlay"></canvas>
      </div>
    </div>
    <aside class="sidebar">
      <div class="card stack">
        <h1>Polygon ROI Editor</h1>
        <p>Left click to add points. Drag an existing point to adjust it. Close the polygon when it has at least 3 points.</p>
        <div class="links">
          <a href="{{ video_feed_url }}" target="_blank" rel="noopener noreferrer">Open raw video</a>
        </div>
      </div>

      <div class="card stack">
        <h2>Actions</h2>
        <div class="button-grid">
          <button class="ok" id="closePolygonBtn">Close Polygon</button>
          <button id="undoPointBtn">Undo Point</button>
          <button id="clearDraftBtn">Clear Draft</button>
          <button id="removeZoneBtn">Remove Last Zone</button>
          <button id="resetBtn">Reset To Config</button>
          <button class="danger" id="clearAllBtn">Clear All</button>
        </div>
      </div>

      <div class="card stack">
        <h2>Export</h2>
        <textarea id="zonesOutput" spellcheck="false" readonly></textarea>
        <div class="button-grid">
          <button class="ok" id="copyRawBtn">Copy Raw Value</button>
          <button class="ok" id="copyEnvBtn">Copy Env Line</button>
        </div>
        <p class="inline-note">Use this as `COUNT_ZONES='...'` in your run command or `.env`.</p>
      </div>

      <div class="card stack">
        <h2>Status</h2>
        <div class="status" id="statusBox"></div>
      </div>
    </aside>
  </div>

  <script>
    const initialZones = {{ initial_count_zones|tojson }};
    const detectionRoi = {{ detection_roi|tojson }};

    const video = document.getElementById("videoFeed");
    const canvas = document.getElementById("overlay");
    const ctx = canvas.getContext("2d");
    const output = document.getElementById("zonesOutput");
    const statusBox = document.getElementById("statusBox");
    const storageKey = "vehicle_detection_count_zones";

    const palette = [
      "#00f5d4",
      "#ffd166",
      "#ff7b72",
      "#7cc6fe",
      "#c77dff",
      "#90f18d",
      "#ffa94d",
      "#f783ac",
    ];

    let zones = [];
    let draft = [];
    let hoverPoint = null;
    let dragging = null;
    let pendingAddPoint = null;
    let pointerMoved = false;

    function deepCloneZones(zoneList) {
      return (zoneList || []).map((zone) => zone.map((point) => ({ x: point[0], y: point[1] })));
    }

    function loadState() {
      const raw = window.localStorage.getItem(storageKey);
      if (!raw) {
        zones = deepCloneZones(initialZones);
        draft = [];
        return;
      }
      try {
        const parsed = JSON.parse(raw);
        zones = Array.isArray(parsed.zones) ? parsed.zones : deepCloneZones(initialZones);
        draft = Array.isArray(parsed.draft) ? parsed.draft : [];
      } catch (err) {
        zones = deepCloneZones(initialZones);
        draft = [];
      }
    }

    function saveState() {
      window.localStorage.setItem(storageKey, JSON.stringify({ zones, draft }));
    }

    function clamp01(value) {
      return Math.max(0, Math.min(1, value));
    }

    function syncCanvasSize() {
      const rect = video.getBoundingClientRect();
      const width = Math.max(1, Math.round(rect.width));
      const height = Math.max(1, Math.round(rect.height));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    }

    function normalizedFromEvent(event) {
      const rect = canvas.getBoundingClientRect();
      const x = clamp01((event.clientX - rect.left) / rect.width);
      const y = clamp01((event.clientY - rect.top) / rect.height);
      return { x, y };
    }

    function pixelFromNormalized(point) {
      return {
        x: point.x * canvas.width,
        y: point.y * canvas.height,
      };
    }

    function colorForIndex(index) {
      return palette[index % palette.length];
    }

    function colorWithAlpha(hex, alpha) {
      const value = hex.replace("#", "");
      const r = Number.parseInt(value.slice(0, 2), 16);
      const g = Number.parseInt(value.slice(2, 4), 16);
      const b = Number.parseInt(value.slice(4, 6), 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    function drawHandle(point, color, radius = 5) {
      const pixel = pixelFromNormalized(point);
      ctx.beginPath();
      ctx.arc(pixel.x, pixel.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(0, 0, 0, 0.7)";
      ctx.stroke();
    }

    function drawPolygon(points, color, label) {
      if (!points.length) {
        return;
      }
      const px = points.map(pixelFromNormalized);
      ctx.beginPath();
      ctx.moveTo(px[0].x, px[0].y);
      for (let i = 1; i < px.length; i += 1) {
        ctx.lineTo(px[i].x, px[i].y);
      }
      ctx.closePath();
      ctx.fillStyle = colorWithAlpha(color, 0.16);
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = color;
      ctx.stroke();

      points.forEach((point) => drawHandle(point, color));

      const minX = Math.min(...px.map((point) => point.x));
      const minY = Math.min(...px.map((point) => point.y));
      ctx.font = "bold 14px Segoe UI";
      ctx.fillStyle = color;
      ctx.fillText(label, minX + 6, Math.max(18, minY - 8));
    }

    function drawDraftLine() {
      if (!draft.length) {
        return;
      }
      const color = colorForIndex(zones.length);
      const px = draft.map(pixelFromNormalized);
      ctx.beginPath();
      ctx.moveTo(px[0].x, px[0].y);
      for (let i = 1; i < px.length; i += 1) {
        ctx.lineTo(px[i].x, px[i].y);
      }
      if (hoverPoint) {
        const hoverPx = pixelFromNormalized(hoverPoint);
        ctx.lineTo(hoverPx.x, hoverPx.y);
      }
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 6]);
      ctx.strokeStyle = color;
      ctx.stroke();
      ctx.setLineDash([]);
      draft.forEach((point) => drawHandle(point, color, 4));
    }

    function drawDetectionRoi() {
      const [x1, y1, x2, y2] = detectionRoi;
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(86, 212, 255, 0.95)";
      ctx.strokeRect(
        x1 * canvas.width,
        y1 * canvas.height,
        (x2 - x1) * canvas.width,
        (y2 - y1) * canvas.height,
      );
      ctx.font = "bold 13px Segoe UI";
      ctx.fillStyle = "rgba(86, 212, 255, 0.95)";
      ctx.fillText("DETECTION_ROI", x1 * canvas.width + 6, Math.max(18, y1 * canvas.height - 8));
    }

    function formatPoint(point) {
      return `${point.x.toFixed(4)},${point.y.toFixed(4)}`;
    }

    function serializeZones() {
      return zones
        .map((zone) => zone.map(formatPoint).join("|"))
        .join(";");
    }

    function updateOutput() {
      output.value = serializeZones();
      saveState();
      const cursorText = hoverPoint
        ? `cursor=${hoverPoint.x.toFixed(4)},${hoverPoint.y.toFixed(4)}`
        : "cursor=none";
      statusBox.textContent =
        `zones=${zones.length}\n` +
        `draft_points=${draft.length}\n` +
        `${cursorText}\n` +
        `detection_roi=${detectionRoi.join(",")}`;
    }

    function findNearestHandle(event) {
      const rect = canvas.getBoundingClientRect();
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;
      const threshold = 12;

      const collections = [
        ...zones.map((zone, zoneIndex) => ({ zone, zoneIndex, isDraft: false })),
        { zone: draft, zoneIndex: -1, isDraft: true },
      ];

      for (const collection of collections) {
        for (let pointIndex = 0; pointIndex < collection.zone.length; pointIndex += 1) {
          const point = collection.zone[pointIndex];
          const handlePx = pixelFromNormalized(point);
          const distance = Math.hypot(px - handlePx.x, py - handlePx.y);
          if (distance <= threshold) {
            return {
              isDraft: collection.isDraft,
              zoneIndex: collection.zoneIndex,
              pointIndex,
            };
          }
        }
      }
      return null;
    }

    function setHandlePoint(handle, point) {
      if (handle.isDraft) {
        draft[handle.pointIndex] = point;
        return;
      }
      zones[handle.zoneIndex][handle.pointIndex] = point;
    }

    function redraw() {
      syncCanvasSize();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawDetectionRoi();
      zones.forEach((zone, index) => drawPolygon(zone, colorForIndex(index), `Zone ${index + 1}`));
      drawDraftLine();
    }

    function closePolygon() {
      if (draft.length < 3) {
        window.alert("Polygon needs at least 3 points.");
        return;
      }
      zones.push(draft.map((point) => ({ ...point })));
      draft = [];
      updateOutput();
      redraw();
    }

    canvas.addEventListener("pointerdown", (event) => {
      const handle = findNearestHandle(event);
      pointerMoved = false;
      if (handle) {
        dragging = handle;
      } else {
        pendingAddPoint = normalizedFromEvent(event);
      }
    });

    canvas.addEventListener("pointermove", (event) => {
      hoverPoint = normalizedFromEvent(event);
      if (dragging) {
        setHandlePoint(dragging, hoverPoint);
        pointerMoved = true;
        updateOutput();
      }
      redraw();
    });

    canvas.addEventListener("pointerup", (event) => {
      if (dragging) {
        dragging = null;
        updateOutput();
        redraw();
        return;
      }
      if (pendingAddPoint && !pointerMoved) {
        draft.push(normalizedFromEvent(event));
        updateOutput();
        redraw();
      }
      pendingAddPoint = null;
    });

    canvas.addEventListener("pointerleave", () => {
      hoverPoint = null;
      if (dragging) {
        dragging = null;
        updateOutput();
      }
      redraw();
    });
    canvas.addEventListener("dblclick", () => {
      if (draft.length >= 3) {
        closePolygon();
      }
    });

    document.getElementById("closePolygonBtn").addEventListener("click", closePolygon);
    document.getElementById("undoPointBtn").addEventListener("click", () => {
      if (!draft.length) {
        return;
      }
      draft.pop();
      updateOutput();
      redraw();
    });
    document.getElementById("clearDraftBtn").addEventListener("click", () => {
      draft = [];
      updateOutput();
      redraw();
    });
    document.getElementById("removeZoneBtn").addEventListener("click", () => {
      if (!zones.length) {
        return;
      }
      zones.pop();
      updateOutput();
      redraw();
    });
    document.getElementById("resetBtn").addEventListener("click", () => {
      zones = deepCloneZones(initialZones);
      draft = [];
      updateOutput();
      redraw();
    });
    document.getElementById("clearAllBtn").addEventListener("click", () => {
      zones = [];
      draft = [];
      updateOutput();
      redraw();
    });
    document.getElementById("copyRawBtn").addEventListener("click", async () => {
      output.select();
      output.setSelectionRange(0, output.value.length);
      await navigator.clipboard.writeText(output.value);
    });
    document.getElementById("copyEnvBtn").addEventListener("click", async () => {
      const envValue = `COUNT_ZONES='${output.value}'`;
      output.value = output.value;
      await navigator.clipboard.writeText(envValue);
    });

    const resizeObserver = new ResizeObserver(() => redraw());
    resizeObserver.observe(video);
    window.addEventListener("resize", redraw);

    video.addEventListener("load", redraw);
    loadState();
    updateOutput();
    redraw();
    window.setInterval(redraw, 1000);
  </script>
</body>
</html>
"""

def generate_frames():
    frame_count = 0
    while True:
        frame = frame_queue.get()  # Get latest frame
        frame_count += 1
        if frame is not None and frame_count % 1 == 0:  # Encode every 3rd frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return (
        '<html><body style="font-family: sans-serif; background:#111; color:#eee; padding:24px;">'
        '<h2>Vehicle Detection</h2>'
        '<p><a style="color:#56d4ff" href="/video_feed">/video_feed</a></p>'
        '<p><a style="color:#56d4ff" href="/roi_editor">/roi_editor</a></p>'
        '</body></html>'
    )


@app.route('/roi_editor')
def roi_editor():
    from config.config import Config
    config = Config()
    return render_template_string(
        ROI_EDITOR_TEMPLATE,
        video_feed_url="/video_feed",
        initial_count_zones=[list(zone) for zone in config.COUNT_ZONES],
        detection_roi=list(config.DETECTION_ROI),
    )

def run_flask():
    from config.config import Config
    config = Config()
    app.run(host='0.0.0.0', port=config.STREAM_PORT, threaded=True, use_reloader=False)
