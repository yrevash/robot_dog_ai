const $ = (id) => document.getElementById(id);
const api = (path, opts = {}) => fetch(path, {
  headers: { "Content-Type": "application/json" }, ...opts,
});

const feed = $("feed");
const localcam = $("localcam");
const log = $("log");
const dot = $("dot");

let currentMode = "pi_camera";
let framesSocket = null;
let localStream = null;
let frameTimer = null;

// -------- status websocket --------
function connectStatus() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/status`);
  ws.onopen  = () => { dot.className = "dot on"; };
  ws.onclose = () => { dot.className = "dot off"; setTimeout(connectStatus, 1500); };
  ws.onmessage = (e) => applyState(JSON.parse(e.data));
}

let aiPending = false;

function applyState(s) {
  $("s-mode").textContent  = s.mode ?? "—";
  $("s-auth").textContent  = s.authorized ?? "—";
  $("s-gest").textContent  = s.last_gesture ?? "—";
  $("s-cmd").textContent   = s.last_command ?? "—";
  $("s-speed").textContent = s.speed;
  $("s-fps").textContent   = s.fps ?? 0;
  $("s-uart").textContent  = s.uart_connected ? "connected" : "mock";

  const aiBtn = $("ai-toggle");
  // Don't overwrite the transient "Starting…" / "Stopping…" label.
  if (!aiPending) {
    if (s.ai_enabled) {
      aiBtn.textContent = "Stop AI";
      aiBtn.classList.remove("primary"); aiBtn.classList.add("danger");
    } else {
      aiBtn.textContent = "Start AI";
      aiBtn.classList.remove("danger"); aiBtn.classList.add("primary");
    }
  }

  if (s.mode && s.mode !== currentMode) {
    currentMode = s.mode;
    syncModeButtons();
    applyModeUi();
  }
  if (Array.isArray(s.log)) {
    log.innerHTML = s.log.slice(-30).reverse().map(l => `<li>${l}</li>`).join("");
  }
}

// -------- mode switch --------
function syncModeButtons() {
  document.querySelectorAll("#mode-seg button").forEach(b => {
    b.classList.toggle("active", b.dataset.mode === currentMode);
  });
}

async function switchMode(mode) {
  const r = await api("/api/mode", { method: "POST", body: JSON.stringify({ mode }) });
  if (!r.ok) { alert("mode switch failed"); return; }
  currentMode = mode;
  syncModeButtons();
  applyModeUi();
}

function applyModeUi() {
  stopLaptopCapture();
  if (currentMode === "pi_camera") {
    feed.style.display = "block";
    localcam.style.display = "none";
  } else if (currentMode === "laptop_stream") {
    feed.style.display = "none";
    localcam.style.display = "block";
    startLaptopCapture(/*stream=*/true);
  } else if (currentMode === "laptop_client") {
    // Run-laptop-AI mode: the laptop_client.py script posts commands.
    // The UI still shows the Pi feed (annotated with whatever laptop pushes via /api/gesture).
    feed.style.display = "block";
    localcam.style.display = "none";
  }
}

// -------- laptop camera capture (mode: laptop_stream) --------
async function startLaptopCapture(stream) {
  try {
    localStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 }, audio: false });
  } catch (e) { alert("camera permission denied: " + e); return; }
  localcam.srcObject = localStream;
  await localcam.play();

  if (stream) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    framesSocket = new WebSocket(`${proto}://${location.host}/ws/frames`);
    await new Promise(r => framesSocket.addEventListener("open", r, { once: true }));

    const canvas = document.createElement("canvas");
    canvas.width = 640; canvas.height = 480;
    const ctx = canvas.getContext("2d");

    frameTimer = setInterval(() => {
      if (!framesSocket || framesSocket.readyState !== 1) return;
      ctx.drawImage(localcam, 0, 0, 640, 480);
      canvas.toBlob((blob) => {
        if (blob && framesSocket.readyState === 1) blob.arrayBuffer().then(b => framesSocket.send(b));
      }, "image/jpeg", 0.6);
    }, 100);  // ~10 fps upload
  }
}

function stopLaptopCapture() {
  if (frameTimer) { clearInterval(frameTimer); frameTimer = null; }
  if (framesSocket) { try { framesSocket.close(); } catch (_) {} framesSocket = null; }
  if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
}

// -------- wire buttons --------
document.querySelectorAll("#mode-seg button").forEach(b => {
  b.addEventListener("click", () => switchMode(b.dataset.mode));
});
document.querySelectorAll('[data-cmd]').forEach(b => {
  b.addEventListener("click", async () => {
    await api("/api/" + b.dataset.cmd, { method: "POST" });
  });
});
document.querySelectorAll('[data-speed]').forEach(b => {
  b.addEventListener("click", async () => {
    document.querySelectorAll('[data-speed]').forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    await api("/api/speed/" + b.dataset.speed, { method: "POST" });
  });
});
$("ai-toggle").addEventListener("click", async () => {
  if (aiPending) return;
  const btn = $("ai-toggle");
  const stopping = btn.textContent.includes("Stop");
  aiPending = true;
  btn.disabled = true;
  btn.textContent = stopping ? "Stopping…" : "Starting…";
  try {
    const r = await api(stopping ? "/api/ai/stop" : "/api/ai/start", { method: "POST" });
    if (!r.ok) {
      const err = await r.text();
      alert("AI " + (stopping ? "stop" : "start") + " failed: " + err);
    }
  } catch (e) {
    alert("AI toggle error: " + e);
  } finally {
    aiPending = false;
    btn.disabled = false;
    // The next WS status update will set the correct label via applyState().
  }
});
$("raw-send").addEventListener("click", async () => {
  const cmd = $("raw").value.trim();
  if (!cmd) return;
  await api("/api/command", { method: "POST", body: JSON.stringify({ cmd }) });
  $("raw").value = "";
});

// populate gesture buttons
api("/api/gestures").then(r => r.json()).then(({ gestures }) => {
  const row = $("gesture-row");
  gestures.forEach(g => {
    const btn = document.createElement("button");
    btn.textContent = g;
    btn.addEventListener("click", async () => {
      await api("/api/gesture", { method: "POST", body: JSON.stringify({ gesture: g }) });
    });
    row.appendChild(btn);
  });
});

connectStatus();
