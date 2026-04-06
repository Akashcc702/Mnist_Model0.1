// ===== STATE =====
let model;
let drawCanvas, drawCtx;
let quizCanvas, quizCtx;
let drawing = false;
let activeCanvas = null;
let activeCtx = null;
let brushSize = 22;
let quizBrushSize = 22;

let undoStack = [];
let quizUndoStack = [];
const MAX_UNDO = 20;

let chart = null;
let predHistory = [];
let autoPredict = null;
let quizAutoPred = null;

// Quiz state
let quizTarget = null;
let quizActive = false;
let quizScore = { correct: 0, total: 0 };
let quizAnswered = false;
let quizLang = { digits: ['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine'] };

// Camera state
let cameraStream = null;

// ===== INIT =====
async function init() {
  try {
    model = await tf.loadLayersModel('./model/mnist_model.json');
    console.log('Model loaded ✅');
  } catch (e) {
    alert('Model failed to load ❌');
    return;
  }

  // Draw canvas
  drawCanvas = document.getElementById('sketchpad');
  drawCtx = drawCanvas.getContext('2d');
  initCtx(drawCtx);
  setupCanvasEvents(drawCanvas, drawCtx, undoStack, 'draw');

  // Quiz canvas
  quizCanvas = document.getElementById('quiz-canvas');
  quizCtx = quizCanvas.getContext('2d');
  initCtx(quizCtx);
  setupCanvasEvents(quizCanvas, quizCtx, quizUndoStack, 'quiz');

  speechSynthesis.onvoiceschanged = () => {};
}

function initCtx(ctx) {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}

// ===== CANVAS EVENTS =====
function setupCanvasEvents(canvas, ctx, stack, mode) {
  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    if (e.touches && e.touches.length > 0) {
      return {
        x: (e.touches[0].clientX - rect.left) * sx,
        y: (e.touches[0].clientY - rect.top) * sy
      };
    }
    return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy };
  }

  function onStart(e) {
    e.preventDefault();
    drawing = true;
    activeCanvas = canvas;
    activeCtx = ctx;

    // Save state for undo
    if (stack.length >= MAX_UNDO) stack.shift();
    stack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));

    // Hide hint
    if (mode === 'draw') {
      const hint = document.getElementById('canvas-hint');
      if (hint) hint.classList.add('hidden');
    }

    if (mode === 'draw') {
      document.getElementById('undo_btn').disabled = stack.length === 0;
    } else {
      document.getElementById('quiz_undo_btn').disabled = stack.length === 0;
    }

    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  }

  function onMove(e) {
    e.preventDefault();
    if (!drawing || activeCanvas !== canvas) return;
    const pos = getPos(e);
    const size = mode === 'draw' ? brushSize : quizBrushSize;
    ctx.lineWidth = size;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#fff';
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  }

  function onEnd(e) {
    e.preventDefault();
    drawing = false;
    ctx.beginPath();

    if (mode === 'draw') {
      clearTimeout(autoPredict);
      autoPredict = setTimeout(predict, 700);
    } else if (mode === 'quiz' && quizActive && !quizAnswered) {
      clearTimeout(quizAutoPred);
      quizAutoPred = setTimeout(checkQuizAnswer, 700);
    }
  }

  canvas.addEventListener('mousedown', onStart);
  canvas.addEventListener('mousemove', onMove);
  canvas.addEventListener('mouseup', onEnd);
  canvas.addEventListener('mouseleave', onEnd);
  canvas.addEventListener('touchstart', onStart, { passive: false });
  canvas.addEventListener('touchmove', onMove, { passive: false });
  canvas.addEventListener('touchend', onEnd, { passive: false });
}

// ===== BRUSH =====
function updateBrush(val) {
  brushSize = parseInt(val);
  document.getElementById('brush_display').textContent = val;
}

function updateQuizBrush(val) {
  quizBrushSize = parseInt(val);
}

// ===== UNDO =====
function undo() {
  if (undoStack.length === 0) return;
  const data = undoStack.pop();
  drawCtx.putImageData(data, 0, 0);
  document.getElementById('undo_btn').disabled = undoStack.length === 0;
  clearTimeout(autoPredict);
  autoPredict = setTimeout(predict, 700);
}

function undoQuiz() {
  if (quizUndoStack.length === 0) return;
  const data = quizUndoStack.pop();
  quizCtx.putImageData(data, 0, 0);
  document.getElementById('quiz_undo_btn').disabled = quizUndoStack.length === 0;
}

// ===== CLEAR =====
function clearCanvas() {
  clearTimeout(autoPredict);
  undoStack = [];
  initCtx(drawCtx);
  document.getElementById('canvas-hint').classList.remove('hidden');
  document.getElementById('undo_btn').disabled = true;
  document.getElementById('result').textContent = '-';
  document.getElementById('confidence').textContent = 'Confidence: -';
  if (chart) { chart.data.datasets[0].data = [0,0,0,0,0,0,0,0,0,0]; chart.update('none'); }
}

function clearQuizCanvas() {
  quizUndoStack = [];
  initCtx(quizCtx);
  document.getElementById('quiz_undo_btn').disabled = true;
}

// ===== PREPROCESS (Improved: bounding box → center → pad → 28×28) =====
function preprocessCanvas(sourceCanvas) {
  const W = sourceCanvas.width;
  const H = sourceCanvas.height;
  const ctx = sourceCanvas.getContext('2d');
  const data = ctx.getImageData(0, 0, W, H).data;

  let minX = W, minY = H, maxX = 0, maxY = 0, hasPixels = false;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      if (data[i] > 50 || data[i+1] > 50 || data[i+2] > 50) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        hasPixels = true;
      }
    }
  }

  if (!hasPixels) return tf.zeros([1, 28, 28, 1]);

  const bboxW = maxX - minX + 1;
  const bboxH = maxY - minY + 1;
  const size = Math.max(bboxW, bboxH);
  const offX = minX - Math.floor((size - bboxW) / 2);
  const offY = minY - Math.floor((size - bboxH) / 2);
  const pad = Math.floor(size * 0.25);
  const finalSize = size + pad * 2;

  const tmp = document.createElement('canvas');
  tmp.width = finalSize; tmp.height = finalSize;
  const tCtx = tmp.getContext('2d');
  tCtx.fillStyle = '#000';
  tCtx.fillRect(0, 0, finalSize, finalSize);
  tCtx.drawImage(sourceCanvas, offX, offY, size, size, pad, pad, size, size);

  const mn = document.createElement('canvas');
  mn.width = 28; mn.height = 28;
  mn.getContext('2d').drawImage(tmp, 0, 0, 28, 28);

  const pixels = mn.getContext('2d').getImageData(0, 0, 28, 28).data;
  const input = [];
  for (let i = 0; i < pixels.length; i += 4) input.push(pixels[i] / 255.0);
  return tf.tensor(input).reshape([1, 28, 28, 1]);
}

// ===== PREDICT (Draw tab) =====
async function predict() {
  if (!model) return;
  const tensor = preprocessCanvas(drawCanvas);
  const pred = model.predict(tensor);
  const probs = Array.from(pred.dataSync());
  const max = Math.max(...probs);
  const digit = probs.indexOf(max);
  tf.dispose([tensor, pred]);

  document.getElementById('result').textContent = digit;
  const confPct = (max * 100).toFixed(1);
  document.getElementById('confidence').textContent = `Confidence: ${confPct}%`;

  updateChart(probs);
  speak(digit, false);
  addToHistory(digit, confPct, null);
}

// ===== CHART =====
function updateChart(probs) {
  const maxIdx = probs.indexOf(Math.max(...probs));
  const colors = probs.map((_, i) =>
    i === maxIdx ? 'rgba(79,70,229,0.85)' : 'rgba(79,70,229,0.25)'
  );
  const canvasEl = document.getElementById('myChart');
  if (!canvasEl) return;

  if (chart) {
    chart.data.datasets[0].data = probs;
    chart.data.datasets[0].backgroundColor = colors;
    chart.update('none');
    return;
  }
  chart = new Chart(canvasEl.getContext('2d'), {
    type: 'bar',
    data: {
      labels: ['0','1','2','3','4','5','6','7','8','9'],
      datasets: [{
        data: probs,
        backgroundColor: colors,
        borderRadius: 4,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        y: {
          beginAtZero: true, max: 1,
          ticks: { callback: v => (v*100).toFixed(0)+'%', font: { size: 10 } },
          grid: { color: 'rgba(0,0,0,0.05)' }
        },
        x: { ticks: { font: { size: 11 } }, grid: { display: false } }
      }
    }
  });
}

// ===== SPEECH =====
function speak(digit, isKannadaQuiz) {
  const toggle = document.getElementById('voice_toggle');
  if (toggle && !toggle.checked) return;

  const lang = document.getElementById('language_select').value;
  let text, langCode;

  if (lang === 'kn') {
    const knDigits = ['ಶೂನ್ಯ','ಒಂದು','ಎರಡು','ಮೂರು','ನಾಲ್ಕು','ಐದು','ಆರು','ಏಳು','ಎಂಟು','ಒಂಬತ್ತು'];
    text = 'ನೀವು ಬರೆದ ಸಂಖ್ಯೆ ' + knDigits[digit];
    langCode = 'kn-IN';
  } else {
    text = 'The number is ' + digit;
    langCode = 'en-US';
  }

  const msg = new SpeechSynthesisUtterance(text);
  msg.lang = langCode;
  msg.rate = 0.9;
  const voices = speechSynthesis.getVoices();
  const voice = voices.find(v => v.lang === langCode) || voices.find(v => v.lang.includes('en'));
  if (voice) msg.voice = voice;
  speechSynthesis.cancel();
  speechSynthesis.speak(msg);
}

// ===== TAB SWITCHING =====
function switchTab(tab, btn) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
  btn.classList.add('active');

  if (tab !== 'camera' && cameraStream) stopCamera();
}

// ===== HISTORY =====
function addToHistory(digit, conf, isCorrect) {
  predHistory.unshift({ digit, conf, isCorrect, time: new Date() });
  if (predHistory.length > 20) predHistory.pop();
  renderHistory();
}

function renderHistory() {
  const list = document.getElementById('history-list');
  const total = document.getElementById('hist-total');
  const accEl = document.getElementById('hist-accuracy');
  const accVal = document.getElementById('hist-acc-val');

  total.textContent = predHistory.length + ' prediction' + (predHistory.length !== 1 ? 's' : '');

  const quizItems = predHistory.filter(h => h.isCorrect !== null);
  if (quizItems.length > 0) {
    const correct = quizItems.filter(h => h.isCorrect).length;
    accEl.classList.remove('hidden');
    accVal.textContent = ((correct / quizItems.length) * 100).toFixed(0) + '%';
  }

  if (predHistory.length === 0) {
    list.innerHTML = '<div class="history-empty">No predictions yet. Start drawing!</div>';
    return;
  }

  list.innerHTML = predHistory.map(h => {
    const cls = h.isCorrect === null ? '' : (h.isCorrect ? 'correct' : 'wrong');
    const badge = h.isCorrect === null ? '' : (h.isCorrect ? '✅' : '❌');
    return `<div class="history-item ${cls}">
      <div class="history-digit">${h.digit}</div>
      <div class="history-conf">${h.conf}%</div>
      <div class="history-badge">${badge}</div>
    </div>`;
  }).join('');
}

function clearHistory() {
  predHistory = [];
  document.getElementById('hist-accuracy').classList.add('hidden');
  renderHistory();
}

// ===== QUIZ =====
function startQuiz() {
  quizActive = true;
  quizAnswered = false;
  document.getElementById('quiz_start_btn').disabled = true;
  document.getElementById('quiz_check_btn').disabled = false;
  document.getElementById('quiz_next_btn').disabled = true;
  document.getElementById('quiz-feedback').style.display = 'none';
  clearQuizCanvas();
  newQuizRound();
}

function newQuizRound() {
  quizTarget = Math.floor(Math.random() * 10);
  quizAnswered = false;
  clearQuizCanvas();
  document.getElementById('quiz_check_btn').disabled = false;
  document.getElementById('quiz_next_btn').disabled = true;
  document.getElementById('quiz-feedback').style.display = 'none';

  const lang = document.getElementById('language_select').value;
  const knDigits = ['ಶೂನ್ಯ','ಒಂದು','ಎರಡು','ಮೂರು','ನಾಲ್ಕು','ಐದು','ಆರು','ಏಳು','ಎಂಟು','ಒಂಬತ್ತು'];
  const label = lang === 'kn'
    ? `ಈ ಸಂಖ್ಯೆ ಬರೆಯಿರಿ:`
    : `Draw this digit:`;

  document.getElementById('quiz-prompt').innerHTML = `
    <div class="quiz-prompt-text">${label}</div>
    <span class="quiz-prompt-number">${quizTarget}</span>
  `;

  // Speak the prompt
  const toggle = document.getElementById('voice_toggle');
  if (toggle && toggle.checked) {
    const knDigitNames = ['ಶೂನ್ಯ','ಒಂದು','ಎರಡು','ಮೂರು','ನಾಲ್ಕು','ಐದು','ಆರು','ಏಳು','ಎಂಟು','ಒಂಬತ್ತು'];
    const promptText = lang === 'kn'
      ? knDigitNames[quizTarget] + ' ಬರೆಯಿರಿ'
      : 'Draw ' + quizTarget;
    const msg = new SpeechSynthesisUtterance(promptText);
    msg.lang = lang === 'kn' ? 'kn-IN' : 'en-US';
    msg.rate = 0.8;
    speechSynthesis.cancel();
    setTimeout(() => speechSynthesis.speak(msg), 200);
  }
}

async function checkQuizAnswer() {
  if (!model || quizAnswered) return;
  quizAnswered = true;
  document.getElementById('quiz_check_btn').disabled = true;

  const tensor = preprocessCanvas(quizCanvas);
  const pred = model.predict(tensor);
  const probs = Array.from(pred.dataSync());
  const max = Math.max(...probs);
  const digit = probs.indexOf(max);
  tf.dispose([tensor, pred]);

  const correct = digit === quizTarget;
  quizScore.total++;
  if (correct) quizScore.correct++;

  updateQuizScore();

  const lang = document.getElementById('language_select').value;
  const fb = document.getElementById('quiz-feedback');
  fb.style.display = 'block';
  if (correct) {
    fb.className = 'quiz-feedback correct';
    fb.textContent = lang === 'kn'
      ? `✅ ಸರಿ! "${digit}" ಎಂದು predict ಮಾಡಿತು`
      : `✅ Correct! Predicted "${digit}"`;
  } else {
    fb.className = 'quiz-feedback wrong';
    fb.textContent = lang === 'kn'
      ? `❌ ತಪ್ಪು! "${digit}" ಎಂದು predict ಮಾಡಿತು — ಆದರೆ "${quizTarget}" ಆಗಿರಬೇಕಿತ್ತು`
      : `❌ Wrong! Predicted "${digit}" — expected "${quizTarget}"`;
  }

  document.getElementById('quiz_next_btn').disabled = false;
  addToHistory(digit, (max * 100).toFixed(1), correct);
  speak(digit, true);
}

function nextQuizRound() {
  if (!quizActive) return;
  document.getElementById('quiz_next_btn').disabled = true;
  newQuizRound();
}

function updateQuizScore() {
  document.getElementById('quiz-correct').textContent = quizScore.correct;
  document.getElementById('quiz-total').textContent = quizScore.total;
  const acc = quizScore.total > 0
    ? ((quizScore.correct / quizScore.total) * 100).toFixed(0) + '%'
    : '0%';
  document.getElementById('quiz-accuracy').textContent = acc;
}

function resetQuiz() {
  quizActive = false;
  quizTarget = null;
  quizScore = { correct: 0, total: 0 };
  quizAnswered = false;
  clearQuizCanvas();
  updateQuizScore();
  document.getElementById('quiz_start_btn').disabled = false;
  document.getElementById('quiz_check_btn').disabled = true;
  document.getElementById('quiz_next_btn').disabled = true;
  document.getElementById('quiz-feedback').style.display = 'none';
  document.getElementById('quiz-prompt').innerHTML =
    '<div class="quiz-prompt-text">Press Start to play!</div>';
}

// ===== CAMERA =====
async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 640 } }
    });
    const video = document.getElementById('camera-video');
    video.srcObject = cameraStream;

    document.getElementById('start_camera_btn').disabled = true;
    document.getElementById('capture_btn').disabled = false;
    document.getElementById('stop_camera_btn').disabled = false;
  } catch (err) {
    alert('Camera not available: ' + err.message + '\n\nNote: Camera requires HTTPS or localhost.');
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  const video = document.getElementById('camera-video');
  video.srcObject = null;
  document.getElementById('start_camera_btn').disabled = false;
  document.getElementById('capture_btn').disabled = true;
  document.getElementById('stop_camera_btn').disabled = true;
}

async function captureAndPredict() {
  if (!model || !cameraStream) return;

  const video   = document.getElementById('camera-video');
  const camCanvas = document.getElementById('camera-canvas');
  const camCtx  = camCanvas.getContext('2d');
  const threshold = parseInt(document.getElementById('threshold_slider').value);

  // ── Step 1: Capture only the guide-box crop (centre square 60% of shorter dim) ──
  const vw = video.videoWidth  || 300;
  const vh = video.videoHeight || 300;
  const cropSize = Math.floor(Math.min(vw, vh) * 0.6);
  const cx = Math.floor((vw - cropSize) / 2);
  const cy = Math.floor((vh - cropSize) / 2);

  camCanvas.width  = cropSize;
  camCanvas.height = cropSize;
  camCtx.drawImage(video, cx, cy, cropSize, cropSize, 0, 0, cropSize, cropSize);

  // ── Step 2: Show "captured" preview ──
  const prevCap = document.getElementById('preview-captured');
  prevCap.getContext('2d').drawImage(camCanvas, 0, 0, 84, 84);

  // ── Step 3: Grayscale + Adaptive threshold + Invert ──
  const raw = camCtx.getImageData(0, 0, cropSize, cropSize);
  const d   = raw.data;

  // Convert to grayscale
  const gray = new Float32Array(cropSize * cropSize);
  for (let i = 0; i < d.length; i += 4) {
    gray[i >> 2] = 0.299 * d[i] + 0.587 * d[i+1] + 0.114 * d[i+2];
  }

  // Compute local mean with a simple box blur (radius = 15% of cropSize)
  const radius = Math.max(10, Math.floor(cropSize * 0.15));
  const localMean = computeBoxBlur(gray, cropSize, cropSize, radius);

  // Binary threshold: pixel is "ink" if it is darker than local mean minus offset
  const offset = threshold - 100; // centred at 0; user slides +/-100
  for (let j = 0; j < gray.length; j++) {
    // ink (dark) → white (255), paper (light) → black (0)
    const isInk = gray[j] < localMean[j] - offset;
    const val   = isInk ? 255 : 0;
    const base  = j * 4;
    d[base] = d[base+1] = d[base+2] = val;
    d[base+3] = 255;
  }
  camCtx.putImageData(raw, 0, 0);

  // ── Step 4: Show "processed" preview ──
  const prevProc = document.getElementById('preview-processed');
  prevProc.getContext('2d').drawImage(camCanvas, 0, 0, 84, 84);

  // ── Step 5: Bounding-box crop → centre → pad → 28×28 ──
  const mnistCanvas = preprocessCameraCanvas(camCanvas);

  // Show model-input preview
  const prevMn = document.getElementById('preview-mnist');
  prevMn.getContext('2d').drawImage(mnistCanvas, 0, 0, 84, 84);

  // ── Step 6: Predict ──
  const pixels = mnistCanvas.getContext('2d').getImageData(0, 0, 28, 28).data;
  const input  = [];
  for (let i = 0; i < pixels.length; i += 4) input.push(pixels[i] / 255.0);

  const tensor = tf.tensor(input).reshape([1, 28, 28, 1]);
  const pred   = model.predict(tensor);
  const probs  = Array.from(pred.dataSync());
  const max    = Math.max(...probs);
  const digit  = probs.indexOf(max);
  tf.dispose([tensor, pred]);

  const confPct = (max * 100).toFixed(1);
  document.getElementById('camera-result').textContent    = digit;
  document.getElementById('camera-confidence').textContent = `Confidence: ${confPct}%`;

  speak(digit, false);
  addToHistory(digit, confPct, null);
}

// ── Box blur helper (separable, O(n)) ──
function computeBoxBlur(gray, W, H, r) {
  const out = new Float32Array(gray.length);
  const tmp = new Float32Array(gray.length);

  // Horizontal pass
  for (let y = 0; y < H; y++) {
    let sum = 0, count = 0;
    for (let x = 0; x < W; x++) {
      sum   += gray[y * W + x];
      count += 1;
      if (x > r) { sum -= gray[y * W + (x - r - 1)]; count--; }
      tmp[y * W + x] = sum / count;
    }
  }

  // Vertical pass
  for (let x = 0; x < W; x++) {
    let sum = 0, count = 0;
    for (let y = 0; y < H; y++) {
      sum   += tmp[y * W + x];
      count += 1;
      if (y > r) { sum -= tmp[(y - r - 1) * W + x]; count--; }
      out[y * W + x] = sum / count;
    }
  }
  return out;
}

// ── Preprocess binary camera canvas → 28×28 MNIST canvas ──
function preprocessCameraCanvas(src) {
  const W = src.width, H = src.height;
  const d = src.getContext('2d').getImageData(0, 0, W, H).data;

  // Find bounding box of white (ink) pixels
  let minX = W, minY = H, maxX = 0, maxY = 0, any = false;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (d[(y * W + x) * 4] > 128) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        any = true;
      }
    }
  }

  const mn = document.createElement('canvas');
  mn.width = mn.height = 28;
  const mCtx = mn.getContext('2d');
  mCtx.fillStyle = '#000';
  mCtx.fillRect(0, 0, 28, 28);

  if (!any) return mn; // blank

  const bw = maxX - minX + 1, bh = maxY - minY + 1;
  const sz = Math.max(bw, bh);
  const ox = minX - Math.floor((sz - bw) / 2);
  const oy = minY - Math.floor((sz - bh) / 2);
  const pad = Math.floor(sz * 0.25);
  const fs  = sz + pad * 2;

  const tmp = document.createElement('canvas');
  tmp.width = tmp.height = fs;
  const tCtx = tmp.getContext('2d');
  tCtx.fillStyle = '#000';
  tCtx.fillRect(0, 0, fs, fs);
  tCtx.drawImage(src, ox, oy, sz, sz, pad, pad, sz, sz);

  mCtx.drawImage(tmp, 0, 0, 28, 28);
  return mn;
}

// ===== THEME =====
function toggleTheme() {
  document.body.classList.toggle('dark');
  const btn = document.querySelector('.theme-btn');
  btn.textContent = document.body.classList.contains('dark') ? '☀️' : '🌙';
}
