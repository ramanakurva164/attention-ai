/**
 * Attention Tracker — Browser / MediaPipe JS
 * ============================================
 * All processing runs on-device via WebAssembly.
 * No video is sent to any server.
 */

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs";

// ─── Constants (mirrors Python version) ─────────────────────────────────────
const EAR_THRESHOLD       = 0.20;
const EAR_CONSEC          = 2;
const EYE_CLOSE_GRACE     = 0.5;   // seconds before sustained closure = inattentive
const YAW_LIMIT           = 25;    // degrees
const PITCH_LIMIT         = 20;
const GAZE_LIMIT          = 0.30;
const SCORE_DECAY         = 20;    // per second while distracted
const SCORE_RECOVER       = 10;    // per second while attentive
const LOW_ATTN_THRESHOLD  = 40;
const LOW_ATTN_DURATION   = 5;     // seconds before alert fires

// EAR landmark indices (MediaPipe 478-point face mesh)
const LEFT_EAR_IDS   = [362, 385, 387, 263, 373, 380];
const RIGHT_EAR_IDS  = [33,  160, 158, 133, 144, 153];

// Iris landmark indices
const LEFT_IRIS_IDS   = [474, 475, 476, 477];
const RIGHT_IRIS_IDS  = [469, 470, 471, 472];
const LEFT_EYE_CORNERS  = [362, 263];
const RIGHT_EYE_CORNERS = [33,  133];

// Colours for canvas drawing
const CLR = {
  green:  '#00e5a0',
  yellow: '#ffb800',
  red:    '#ff3d57',
  iris:   '#ff8c00',
  dim:    'rgba(255,255,255,0.15)',
  bg:     'rgba(13,13,20,0.72)',
};

// ─── Math helpers ────────────────────────────────────────────────────────────

function eyeAspectRatio(lm, ids, W, H) {
  const p = ids.map(i => [lm[i].x * W, lm[i].y * H]);
  const A = dist(p[1], p[5]);
  const B = dist(p[2], p[4]);
  const C = dist(p[0], p[3]);
  return (A + B) / (2 * C + 1e-6);
}

function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function irisOffset(lm, irisIds, cornerIds, W, H) {
  const irisPts = irisIds.map(i => [lm[i].x * W, lm[i].y * H]);
  const cx = irisPts.reduce((s, p) => s + p[0], 0) / irisPts.length;
  const lc = [lm[cornerIds[0]].x * W, lm[cornerIds[0]].y * H];
  const rc = [lm[cornerIds[1]].x * W, lm[cornerIds[1]].y * H];
  const ew = dist(lc, rc) + 1e-6;
  return (cx - (lc[0] + rc[0]) / 2) / (ew / 2);
}

// Extract Euler angles from the 4x4 face transformation matrix
// MediaPipe provides matrix in column-major order (16 values)
function matrixToEuler(m) {
  // Column-major 4×4 → rotation part
  const R = [
    [m[0], m[4], m[8]],
    [m[1], m[5], m[9]],
    [m[2], m[6], m[10]],
  ];
  const sy = Math.sqrt(R[0][0] ** 2 + R[1][0] ** 2);
  let pitch, yaw, roll;
  if (sy > 1e-6) {
    pitch = Math.atan2(R[2][1], R[2][2]);
    yaw   = Math.atan2(-R[2][0], sy);
    roll  = Math.atan2(R[1][0], R[0][0]);
  } else {
    pitch = Math.atan2(-R[1][2], R[1][1]);
    yaw   = Math.atan2(-R[2][0], sy);
    roll  = 0;
  }
  const r2d = 180 / Math.PI;
  return { pitch: pitch * r2d, yaw: yaw * r2d, roll: roll * r2d };
}

// ─── State ───────────────────────────────────────────────────────────────────
const state = {
  running:        false,
  score:          100,
  blinks:         0,
  distractions:   0,
  blinkCounter:   0,
  eyesClosedSince: null,
  lowAttnSince:   null,
  sessionStart:   null,
  sessionScores:  [],
  sessionLog:     [],   // { t, score, pitch, yaw, roll, earL, earR, gazeL, gazeR, status }
  lastTime:       0,
  fps:            0,
  fpsCounter:     0,
  fpsTimer:       0,
  // smoothing buffers
  pitchBuf: [], yawBuf: [], rollBuf: [],
  SMOOTH: 8,
  pitch: 0, yaw: 0, roll: 0,
  earL: 0, earR: 0,
  gazeL: 0, gazeR: 0,
  eyesClosed: false,
  pose: null,       // { pitch, yaw, roll }
  status: 'ATTENTIVE',
};

function smoothPush(buf, val, N) {
  buf.push(val);
  if (buf.length > N) buf.shift();
  return buf.reduce((s, v) => s + v, 0) / buf.length;
}

// ─── DOM refs ────────────────────────────────────────────────────────────────
const video     = document.getElementById('video');
const canvas    = document.getElementById('canvas');
const ctx       = canvas.getContext('2d');
const placeholder = document.getElementById('placeholder');
const fpschip   = document.getElementById('fps-chip');

const btnStart  = document.getElementById('btn-start');
const btnStop   = document.getElementById('btn-stop');
const btnCsv    = document.getElementById('btn-csv');

const alertOverlay = document.getElementById('alert-overlay');

// HUD elements
const scoreVal      = document.getElementById('score-value');
const scoreBar      = document.getElementById('score-bar');
const hudStatus     = document.getElementById('hud-status');
const statusChip    = document.getElementById('status-label');
const statusDot     = document.getElementById('status-dot');
const statAvg       = document.getElementById('stat-avg');
const statBlinks    = document.getElementById('stat-blinks');
const statDist      = document.getElementById('stat-dist');
const statTime      = document.getElementById('stat-time');
const poseP         = document.getElementById('pose-pitch');
const poseY         = document.getElementById('pose-yaw');
const poseR         = document.getElementById('pose-roll');
const posePBar      = document.getElementById('pose-pitch-bar');
const poseYBar      = document.getElementById('pose-yaw-bar');
const poseRBar      = document.getElementById('pose-roll-bar');
const gazeLabel     = document.getElementById('gaze-label');
const gazeCursor    = document.getElementById('gaze-cursor');
const eyeLeftIcon   = document.getElementById('eye-left-icon');
const eyeRightIcon  = document.getElementById('eye-right-icon');
const eyeStateLabel = document.getElementById('eye-state-label');
const earLeftEl     = document.getElementById('ear-left');
const earRightEl    = document.getElementById('ear-right');

// ─── MediaPipe setup ─────────────────────────────────────────────────────────
let landmarker = null;
let modelLoaded = false;

async function loadModel() {
  showLoadingRing('Loading AI model…');
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      },
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: true,
      runningMode: "VIDEO",
      numFaces: 1,
      minFaceDetectionConfidence: 0.6,
      minFacePresenceConfidence:  0.6,
      minTrackingConfidence:      0.5,
    });
    modelLoaded = true;
    removeLoadingRing();
    setStatus('ready', 'Ready');
  } catch (e) {
    console.error('Model load failed:', e);
    removeLoadingRing();
    setStatus('danger', 'Model load failed');
  }
}

// ─── Camera ──────────────────────────────────────────────────────────────────
async function startCamera() {
  if (!modelLoaded) { alert('Model still loading, please wait…'); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    placeholder.classList.add('fade-out');
    state.running = true;
    state.sessionStart = Date.now();
    state.score = 100;
    state.blinks = 0;
    state.distractions = 0;
    state.sessionScores = [];
    state.sessionLog = [];
    state.lastTime = performance.now();
    state.fpsTimer = performance.now();
    btnStart.classList.add('hidden');
    btnStop.classList.remove('hidden');
    setStatus('live', 'Tracking');
    requestAnimationFrame(loop);
  } catch (e) {
    alert('Camera access denied. Please allow camera permission and try again.');
    console.error(e);
  }
}

function stopCamera() {
  state.running = false;
  const tracks = video.srcObject?.getTracks() || [];
  tracks.forEach(t => t.stop());
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  placeholder.classList.remove('fade-out');
  btnStop.classList.add('hidden');
  btnStart.classList.remove('hidden');
  alertOverlay.classList.add('hidden');
  setStatus('ready', 'Stopped');
}

// ─── Main loop ───────────────────────────────────────────────────────────────
let animId = null;

function loop(now) {
  if (!state.running) return;
  animId = requestAnimationFrame(loop);

  // Resize canvas to match video
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
  }

  const dt = (now - state.lastTime) / 1000;
  state.lastTime = now;
  if (dt <= 0 || dt > 1) return; // skip bad frames

  // Run detection
  const result = landmarker.detectForVideo(video, now);
  processResult(result, dt, now);

  // Draw canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawCanvas(result);

  // FPS
  state.fpsCounter++;
  if (now - state.fpsTimer >= 1000) {
    state.fps = Math.round(state.fpsCounter * 1000 / (now - state.fpsTimer));
    state.fpsCounter = 0;
    state.fpsTimer = now;
    fpschip.textContent = state.fps + ' FPS';
  }

  // Update HUD ~15fps
  updateHUD(now);
}

// ─── Attention logic ─────────────────────────────────────────────────────────
function processResult(result, dt, now) {
  const lm = result.faceLandmarks?.[0];
  const W = canvas.width, H = canvas.height;

  if (!lm) {
    state.status = 'NO FACE';
    updateScore(true, dt);
    return;
  }

  // EAR
  const earL = eyeAspectRatio(lm, LEFT_EAR_IDS,  W, H);
  const earR = eyeAspectRatio(lm, RIGHT_EAR_IDS, W, H);
  const avgEar = (earL + earR) / 2;
  const eyesClosed = avgEar < EAR_THRESHOLD;

  state.earL = earL;
  state.earR = earR;
  state.eyesClosed = eyesClosed;

  if (eyesClosed) {
    state.blinkCounter++;
    if (!state.eyesClosedSince) state.eyesClosedSince = now / 1000;
  } else {
    if (state.blinkCounter >= EAR_CONSEC) state.blinks++;
    state.blinkCounter = 0;
    state.eyesClosedSince = null;
  }

  const eyesDistracted = eyesClosed &&
    state.eyesClosedSince !== null &&
    (now / 1000 - state.eyesClosedSince) > EYE_CLOSE_GRACE;

  // Gaze
  let gazeL = 0, gazeR = 0, gazeDist = false;
  if (!eyesClosed) {
    gazeL = irisOffset(lm, LEFT_IRIS_IDS,  LEFT_EYE_CORNERS,  W, H);
    gazeR = irisOffset(lm, RIGHT_IRIS_IDS, RIGHT_EYE_CORNERS, W, H);
    const avgGaze = (Math.abs(gazeL) + Math.abs(gazeR)) / 2;
    gazeDist = avgGaze > GAZE_LIMIT;
  }
  state.gazeL = gazeL;
  state.gazeR = gazeR;

  // Head pose from transformation matrix
  let headDist = false;
  const mat = result.facialTransformationMatrixes?.[0]?.data;
  if (mat) {
    const { pitch, yaw, roll } = matrixToEuler(mat);
    const sp = smoothPush(state.pitchBuf, pitch, state.SMOOTH);
    const sy = smoothPush(state.yawBuf,   yaw,   state.SMOOTH);
    const sr = smoothPush(state.rollBuf,  roll,  state.SMOOTH);
    state.pitch = sp; state.yaw = sy; state.roll = sr;
    if (Math.abs(sy) > YAW_LIMIT || Math.abs(sp) > PITCH_LIMIT) headDist = true;
  }

  // Reasons
  const reasons = [];
  if (eyesClosed && eyesDistracted) reasons.push('EYES CLOSED');
  if (headDist) {
    if (Math.abs(state.yaw)   > YAW_LIMIT)   reasons.push(`HEAD ${state.yaw < 0 ? 'LEFT' : 'RIGHT'}`);
    if (Math.abs(state.pitch) > PITCH_LIMIT)  reasons.push(`HEAD ${state.pitch > 0 ? 'DOWN' : 'UP'}`);
  }
  if (gazeDist) reasons.push(`GAZE ${(gazeL + gazeR) > 0 ? 'RIGHT' : 'LEFT'}`);

  const distracted = headDist || gazeDist || eyesDistracted;

  if (eyesClosed && !distracted) {
    state.status = 'EYES CLOSED';
  } else if (reasons.length) {
    state.status = reasons.join(' | ');
  } else {
    state.status = 'ATTENTIVE';
  }

  if (reasons.length) state.distractions++;

  updateScore(distracted, dt);

  // Log row
  state.sessionLog.push({
    t: new Date().toISOString(),
    score: state.score.toFixed(1),
    pitch: state.pitch.toFixed(1), yaw: state.yaw.toFixed(1), roll: state.roll.toFixed(1),
    earL: earL.toFixed(3), earR: earR.toFixed(3),
    gazeL: gazeL.toFixed(3), gazeR: gazeR.toFixed(3),
    status: state.status,
  });

  // Low-attention alert timer
  if (state.score < LOW_ATTN_THRESHOLD) {
    if (!state.lowAttnSince) state.lowAttnSince = now / 1000;
  } else {
    state.lowAttnSince = null;
  }
}

function updateScore(distracted, dt) {
  if (distracted) state.score -= SCORE_DECAY * dt;
  else            state.score += SCORE_RECOVER * dt;
  state.score = Math.max(0, Math.min(100, state.score));
  state.sessionScores.push(state.score);
}

// ─── Canvas drawing ──────────────────────────────────────────────────────────
function drawCanvas(result) {
  const lm = result.faceLandmarks?.[0];
  const W = canvas.width, H = canvas.height;

  if (!lm) {
    // No face banner
    ctx.save();
    ctx.fillStyle = 'rgba(200,30,50,0.18)';
    ctx.fillRect(0, H / 2 - 28, W, 56);
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 18px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('NO FACE DETECTED', W / 2, H / 2);
    ctx.restore();
    return;
  }

  // Eye bounding boxes
  drawEyeBox(lm, LEFT_EAR_IDS,  W, H, state.eyesClosed);
  drawEyeBox(lm, RIGHT_EAR_IDS, W, H, state.eyesClosed);

  // Iris circles (only when open)
  if (!state.eyesClosed) {
    drawIris(lm, LEFT_IRIS_IDS,  W, H);
    drawIris(lm, RIGHT_IRIS_IDS, W, H);
  }

  // Key dots: nose, chin, mouth corners
  [1, 152, 57, 287, 4].forEach(i => {
    const x = lm[i].x * W, y = lm[i].y * H;
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(200,200,200,0.5)';
    ctx.fill();
  });

  // Alert border on canvas when score is critically low
  const alertActive = state.lowAttnSince !== null &&
    (performance.now() / 1000 - state.lowAttnSince) >= LOW_ATTN_DURATION;

  if (alertActive) {
    const pulse = 0.45 + 0.3 * Math.sin(performance.now() / 400);
    ctx.save();
    ctx.strokeStyle = `rgba(255,61,87,${pulse})`;
    ctx.lineWidth = 22;
    ctx.strokeRect(0, 0, W, H);
    ctx.restore();
    alertOverlay.classList.remove('hidden');
  } else {
    alertOverlay.classList.add('hidden');
  }
}

function drawEyeBox(lm, ids, W, H, closed) {
  const pts = ids.map(i => [lm[i].x * W, lm[i].y * H]);
  ctx.save();
  ctx.strokeStyle = closed ? 'rgba(100,100,100,0.7)' : CLR.green;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

function drawIris(lm, ids, W, H) {
  const pts = ids.map(i => [lm[i].x * W, lm[i].y * H]);
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  const r  = Math.max(2, dist(pts[0], pts[2]) / 2);
  ctx.save();
  ctx.strokeStyle = CLR.iris;
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
  ctx.fillStyle = CLR.iris;
  ctx.beginPath(); ctx.arc(cx, cy, 2, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

// ─── HUD Update (DOM) ────────────────────────────────────────────────────────
let hudThrottle = 0;

function updateHUD(now) {
  if (now - hudThrottle < 50) return; // ~20fps DOM updates
  hudThrottle = now;

  const s = Math.round(state.score);

  // Score
  scoreVal.textContent = s;
  const pct = `${s}%`;
  scoreBar.style.width = pct;
  const col = s >= 70 ? '#00e5a0' : s >= 40 ? '#ffb800' : '#ff3d57';
  scoreBar.style.background = col;
  scoreVal.style.color = col;

  // Status
  const st = state.status;
  const isGood = st === 'ATTENTIVE';
  const isEye  = st === 'EYES CLOSED';
  hudStatus.textContent = st;
  hudStatus.className   = 'hud-status-label' + (isGood ? '' : isEye ? ' warn' : ' danger');

  // Status chip
  statusChip.textContent = state.running ? (st === 'ATTENTIVE' ? 'Attentive' : 'Distracted') : 'Ready';
  statusDot.className    = 'status-dot' + (state.running ? (isGood ? ' live' : isEye ? ' warn' : ' danger') : '');

  // Session stats
  const elapsed = state.sessionStart ? Math.round((Date.now() - state.sessionStart) / 1000) : 0;
  const avg = state.sessionScores.length
    ? Math.round(state.sessionScores.reduce((a, b) => a + b, 0) / state.sessionScores.length)
    : 0;
  statAvg.textContent    = avg + '%';
  statBlinks.textContent = state.blinks;
  statDist.textContent   = state.distractions;
  statTime.textContent   = elapsed >= 60
    ? `${Math.floor(elapsed/60)}m ${elapsed%60}s`
    : elapsed + 's';

  // Pose bars (map ±90° → 0–100% offset from centre)
  setPoseBar(posePBar, poseP, state.pitch, PITCH_LIMIT);
  setPoseBar(poseYBar, poseY, state.yaw,   YAW_LIMIT);
  setPoseBar(poseRBar, poseR, state.roll,  40);

  // Gaze
  const avgGaze = (state.gazeL + state.gazeR) / 2;
  const gazePos = Math.max(4, Math.min(96, 50 + avgGaze * 38));
  gazeCursor.style.left = gazePos + '%';
  const gazeOff = Math.abs(avgGaze) > GAZE_LIMIT;
  gazeCursor.classList.toggle('offcenter', gazeOff);
  gazeLabel.textContent = state.eyesClosed ? 'Suppressed' : gazeOff
    ? (avgGaze > 0 ? 'Looking RIGHT' : 'Looking LEFT')
    : 'Centre';
  gazeLabel.className = 'gaze-label' + (gazeOff ? ' offcenter' : '');

  // Eyes
  const ec = state.eyesClosed;
  eyeLeftIcon.className  = 'eye-icon' + (ec ? ' closed' : '');
  eyeRightIcon.className = 'eye-icon' + (ec ? ' closed' : '');
  eyeStateLabel.textContent = ec ? 'CLOSED' : 'OPEN';
  eyeStateLabel.className   = 'eye-state' + (ec ? ' closed' : '');
  earLeftEl.textContent  = state.earL.toFixed(3);
  earRightEl.textContent = state.earR.toFixed(3);
}

function setPoseBar(barEl, labelEl, val, limit) {
  const pct = Math.max(2, Math.min(96, 50 + (val / limit) * 45));
  barEl.style.left = pct + '%';
  const warn = Math.abs(val) > limit;
  barEl.classList.toggle('warn', warn);
  const sign = val >= 0 ? '+' : '';
  labelEl.textContent = sign + val.toFixed(1) + '°';
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
function setStatus(type, label) {
  statusDot.className = 'status-dot' + (type === 'ready' ? '' : ' ' + type);
  statusChip.textContent = label;
}

function showLoadingRing(msg) {
  const el = document.createElement('div');
  el.id = 'loading-ring';
  el.className = 'loading-ring';
  el.innerHTML = `<div class="spinner"></div><div class="loading-text">${msg}</div>`;
  document.getElementById('camera-wrapper').appendChild(el);
}
function removeLoadingRing() {
  document.getElementById('loading-ring')?.remove();
}

// ─── HTML Report Export ────────────────────────────────────────────────────────
function exportHTML() {
  if (!state.sessionLog.length) { alert('No data recorded yet.'); return; }
  
  const headers = ['Timestamp', 'Score', 'Pitch', 'Yaw', 'Roll', 'EAR L', 'EAR R', 'Gaze L', 'Gaze R', 'Status'];
  const tbody = state.sessionLog.map(r => `
    <tr>
      <td>${r.t.split('T')[1].slice(0, 8)}</td>
      <td class="score-${r.score >= 70 ? 'good' : r.score >= 40 ? 'warn' : 'bad'}">${r.score}%</td>
      <td>${r.pitch}°</td><td>${r.yaw}°</td><td>${r.roll}°</td>
      <td>${r.earL}</td><td>${r.earR}</td>
      <td>${r.gazeL}</td><td>${r.gazeR}</td>
      <td><span class="status ${r.status === 'ATTENTIVE' ? 'good' : r.status === 'EYES CLOSED' ? 'warn' : 'bad'}">${r.status}</span></td>
    </tr>
  `).join('');

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Attention Report - ${new Date().toLocaleString()}</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #0d0d14; color: #e8e8f0; padding: 20px; line-height: 1.5; margin: 0; }
    .container { max-width: 1000px; margin: 0 auto; background: #1a1a2e; border-radius: 12px; padding: 24px; box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
    h1 { color: #fff; margin-top: 0; font-size: 1.5rem; }
    p { color: #a0a0b0; font-size: 0.9rem; }
    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; margin-bottom: 30px; }
    .stat { background: #13131f; padding: 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); }
    .stat-val { font-size: 1.8rem; font-weight: bold; color: #fff; }
    .stat-key { font-size: 0.8rem; color: #a0a0b0; text-transform: uppercase; letter-spacing: 1px; }
    table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 0.85rem; }
    th { text-align: left; padding: 12px 8px; color: #a0a0b0; border-bottom: 2px solid rgba(255,255,255,0.1); white-space: nowrap; }
    td { padding: 10px 8px; border-bottom: 1px solid rgba(255,255,255,0.05); white-space: nowrap; }
    tr:hover td { background: rgba(255,255,255,0.02); }
    .score-good { color: #00e5a0; font-weight: bold; }
    .score-warn { color: #ffb800; font-weight: bold; }
    .score-bad  { color: #ff3d57; font-weight: bold; }
    .status { padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.75rem; letter-spacing: 0.5px; }
    .status.good { background: rgba(0,229,160,0.1); color: #00e5a0; border: 1px solid rgba(0,229,160,0.2); }
    .status.warn { background: rgba(255,184,0,0.1); color: #ffb800; border: 1px solid rgba(255,184,0,0.2); }
    .status.bad  { background: rgba(255,61,87,0.1); color: #ff3d57; border: 1px solid rgba(255,61,87,0.2); }
    @media (max-width: 768px) {
      body { padding: 10px; }
      .container { padding: 16px; }
      .table-wrapper { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Attention Report</h1>
    <p>Session started at ${new Date(state.sessionStart).toLocaleString()}</p>
    
    <div class="summary">
      <div class="stat"><div class="stat-val">${statAvg.textContent}</div><div class="stat-key">Average Score</div></div>
      <div class="stat"><div class="stat-val">${state.blinks}</div><div class="stat-key">Total Blinks</div></div>
      <div class="stat"><div class="stat-val">${state.distractions}</div><div class="stat-key">Distractions</div></div>
      <div class="stat"><div class="stat-val">${statTime.textContent}</div><div class="stat-key">Duration</div></div>
    </div>

    <div class="table-wrapper">
      <table>
        <thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>
        <tbody>${tbody}</tbody>
      </table>
    </div>
  </div>
</body>
</html>`;

  const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `attention_report_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.html`;
  a.click();
  URL.revokeObjectURL(url);
}

// ─── Event listeners ─────────────────────────────────────────────────────────
btnStart.addEventListener('click', startCamera);
btnStop.addEventListener('click',  stopCamera);
btnCsv.addEventListener('click',   exportCSV);

// ─── Boot ─────────────────────────────────────────────────────────────────────
loadModel();
