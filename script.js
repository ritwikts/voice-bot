// frontend/main.js
// Persistent listening + VAD-based barge-in + WS streaming + toggle mic behaviour

// ===================== UI Elements =====================
const micBtn = document.getElementById('micBtn');
const micIcon = document.getElementById('micIcon');
const userText = document.getElementById('userText');
const botText = document.getElementById('botText');

// ========== State ==========
let running = false;            // whether assistant is in persistent mode
let recognition = null;
let aiSpeaking = false;
let bargeInEnabled = false;     // becomes true a short time after TTS begins
let audio = null;               // audio playback element for server TTS
let vadRunning = false;
let vadSession = null;
let vadAudioCtx = null;
let vadSource = null;
let vadProcessor = null;
let ws = null;
let wsReady = false;
let currentReqId = null;        // id of current streaming generation
let partialBuffer = {};         // incremental text buffer per id

// Config
const USE_WEBSOCKET = true;
const WS_URL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';

// === helpers ===
function log(...args) { console.log('[voice-assistant]', ...args); }
function uuidv4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random()*16|0, v = c === 'x' ? r : (r&0x3|0x8);
    return v.toString(16);
  });
}

// ========== WebSocket (streaming) ==========
function createWebSocket() {
  if (!USE_WEBSOCKET) return;
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    wsReady = true;
    log('WS open');
  };
  ws.onclose = () => {
    wsReady = false;
    log('WS closed — reconnecting in 2s');
    setTimeout(createWebSocket, 2000);
  };
  ws.onerror = (e) => {
    wsReady = false;
    console.error('WS error', e);
    try { ws.close(); } catch {}
  };
  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      handleWSMessage(msg);
    } catch (e) {
      console.warn('WS non-json message', evt.data);
    }
  };
}

function handleWSMessage(msg) {
  const { type, id, text, error } = msg;
  if (!id) return;

  if (type === 'partial') {
    partialBuffer[id] = (partialBuffer[id] || '') + (text || '');
    // show partial text
    botText.textContent = partialBuffer[id];
  } else if (type === 'final') {
    const finalText = text || partialBuffer[id] || '';
    delete partialBuffer[id];
    // botText.textContent = finalText;
    // Play final audio via /speak endpoint (keeps behavior consistent)
    // start TTS playback (this will set aiSpeaking and enable barge-in)
    playAnswerAudio(finalText);
  } else if (type === 'error') {
    botText.textContent = 'Error: ' + (error || 'Unknown error');
    log('Stream error for', id, error);
  }
}

// ========== SpeechRecognition setup ==========
function setupRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    botText.textContent = 'Speech recognition not supported in this browser.';
    micBtn.disabled = true;
    return;
  }

  recognition = new SpeechRecognition();
  recognition.lang = 'en-IN';
  recognition.continuous = true;      // continuous capture while running
  recognition.interimResults = false; // only final results
  recognition.maxAlternatives = 1;

  recognition.onstart = () => { micIcon.textContent = '⏺️'; log('Recognition started'); };
  recognition.onerror = (e) => { console.error('Recognition error', e); };

  recognition.onresult = async (ev) => {
    // Grab the last result (final)
    const last = ev.results[ev.results.length - 1];
    const transcript = last[0].transcript.trim();
    if (!transcript) return;

    log('Recognized:', transcript);
    userText.textContent = transcript;

    // If assistant was speaking and recognition is able to capture new speech
    // we still want to cancel current generation + TTS so we can respond immediately.
    if (aiSpeaking) {
      // This will stop audio playback and tell the server to cancel generation (if any)
      stopSpeaking(false); // do not auto-restart recognition because it's already running
      // small short delay to ensure TTS fully stopped before sending the new question
      setTimeout(() => handleQuestion(transcript), 40);
    } else {
      // Normal flow: send question
      await handleQuestion(transcript);
    }
  };

  recognition.onend = () => {
    // If still running, restart recognition automatically (keeps continuous flow)
    micIcon.textContent = running ? '🎙️' : '🎤';
    if (running && !aiSpeaking) {
      try { recognition.start(); } catch (e) { log('Recognition restart failed', e); }
    }
  };

  try { recognition.start(); } catch (e) { log('Recognition start failed', e); }
}

// ========== Silero VAD setup ==========
async function loadSileroModel() {
  log('Loading Silero VAD model...');
  // "ort" (onnxruntime-web) must already be included in the page.
  // Path "./silero_vad.onnx" should point to your model file.
  vadSession = await ort.InferenceSession.create("./silero_vad.onnx");
  log('Silero VAD loaded');
  return vadSession;
}

async function startVAD(session) {
  if (!session) throw new Error('VAD session required');

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
  });

  vadAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
  vadSource = vadAudioCtx.createMediaStreamSource(stream);

  // createScriptProcessor buffer size 4096 for slightly more stable VAD; fallback to 512 if unavailable
  const bufferSize = (vadAudioCtx.createScriptProcessor) ? 4096 : 512;
  vadProcessor = vadAudioCtx.createScriptProcessor(bufferSize, 1, 1);

  vadSource.connect(vadProcessor);
  vadProcessor.connect(vadAudioCtx.destination);

  let ring = [];
  vadRunning = true;

  vadProcessor.onaudioprocess = async (e) => {
    if (!vadRunning) return;

    const inputData = e.inputBuffer.getChannelData(0);
    // copy to ring buffer
    for (let i = 0; i < inputData.length; i++) ring.push(inputData[i]);

    // Check every ~0.1 - 0.2s worth of samples (assuming 16kHz -> 1600 samples ~0.1s)
    if (ring.length > 1600) {
      // Create a Float32Array for the model
      try {
        const inputTensor = new ort.Tensor('float32', Float32Array.from(ring), [1, ring.length]);
        const results = await session.run({ input: inputTensor });
        const speechProb = results.output.data[0];

        // Thresholds: lower when idling; higher when we want robust detection during AI audio
        const threshold = aiSpeaking ? 0.8 : 0.65;

        // If AI is speaking and bargeInEnabled (dead zone passed), and we detect user speech -> barge in
        if (aiSpeaking && bargeInEnabled && speechProb > threshold) {
          log('VAD detected user speech during AI audio (prob=' + speechProb + '), triggering barge-in');
          // stop speaking and immediately restart recognition to capture the user's question
          stopSpeaking(true); // stop TTS and start listening
        }

        // If AI is not speaking and VAD detects speech, ensure recognition is active (it usually is)
        // We don't send question directly from VAD — SpeechRecognition handles transcription.
      } catch (err) {
        console.warn('VAD run error', err);
      } finally {
        ring = [];
      }
    }
  };

  log('VAD started');
}

// Stop VAD and free audio nodes
function stopVAD() {
  try {
    vadRunning = false;
    if (vadProcessor) { vadProcessor.disconnect(); vadProcessor.onaudioprocess = null; vadProcessor = null; }
    if (vadSource) { vadSource.disconnect(); vadSource = null; }
    if (vadAudioCtx) { try { vadAudioCtx.close(); } catch {} vadAudioCtx = null; }
    vadSession = null;
    log('VAD stopped');
  } catch (e) { console.error('Error stopping VAD', e); }
}

// ========== AI Speech control (TTS + barge-in) ==========
function onAISpeechStart() {
  aiSpeaking = true;
  // stop recognition while TTS plays to avoid recognition capturing TTS audio;
  // VAD continues to run so it can detect user voice and signal barge-in.
  try { recognition && recognition.stop(); } catch (e) {}
  bargeInEnabled = false;
  // small dead-zone to avoid accidental short noises being considered barge-in
  setTimeout(() => { if (aiSpeaking) bargeInEnabled = true; }, 350);
  log('AI speaking started; barge-in will enable after dead-zone');
}

function onAISpeechEnd() {
  aiSpeaking = false;
  bargeInEnabled = false;
  // restart recognition so further user speech is captured
  try { recognition && recognition.start(); } catch (e) { log('Recognition restart after AI end failed', e); }
  log('AI speaking ended');
}

function stopSpeaking(startListening = false) {
  // Cancel any playing audio
  aiSpeaking = false;
  bargeInEnabled = false;
  if (window.speechSynthesis) window.speechSynthesis.cancel();
  if (audio) {
    try { audio.pause(); } catch (e) {}
    try { audio.src = ''; } catch (e) {}
    audio = null;
  }

  // Notify server to cancel streaming generation if active
  if (wsReady && currentReqId) {
    try { ws.send(JSON.stringify({ type: 'cancel', id: currentReqId })); } catch (e) { log('WS cancel send failed', e); }
    currentReqId = null;
  }

  if (startListening) {
    try { recognition && recognition.start(); } catch (e) { log('Recognition start failed after stopSpeaking', e); }
  }
}

// ========== TTS playback (final audio) ==========
async function playAnswerAudio(answerText) {
  // start TTS lifecycle
  onAISpeechStart();

  try {
    const speakResp = await fetch('/speak', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: answerText })
    });

    if (speakResp.status === 204) {
      log('TTs returned 204 — using browser fallback');
      fallbackSpeak(answerText, onAISpeechEnd);
      return;
    }

    if (!speakResp.ok) throw new Error('TTS fetch failed');

    const audioBlob = await speakResp.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    audio = new Audio(audioUrl);
    audio.onended = () => {
      try { URL.revokeObjectURL(audioUrl); } catch (e) {}
      audio = null;
      onAISpeechEnd();
    };
    audio.onerror = (e) => {
      console.warn('Audio playback error', e);
      audio = null;
      onAISpeechEnd();
    };
    await audio.play();
  } catch (err) {
    console.warn('TTS error, using fallbackSpeak', err);
    fallbackSpeak(answerText, onAISpeechEnd);
  }
}

function fallbackSpeak(txt, cb) {
  if (!window.speechSynthesis) { if (cb) cb(); return; }
  const u = new SpeechSynthesisUtterance(txt);
  u.lang = 'en-IN';
  u.rate = 1;
  u.onend = () => { if (cb) cb(); };
  speechSynthesis.speak(u);
}

// ========== Handle Question flow ==========
async function handleQuestion(questionText) {
  if (!questionText || !questionText.trim()) return;
  botText.textContent = 'Thinking...';
  // Use WS streaming if available
  if (USE_WEBSOCKET && wsReady) {
    const id = uuidv4();
    currentReqId = id;
    partialBuffer[id] = '';
    try {
      ws.send(JSON.stringify({ type: 'query', id, question: questionText }));
    } catch (e) {
      log('WS send failed, falling back to fetch', e);
      await handleQuestionViaFetch(questionText);
    }
  } else {
    await handleQuestionViaFetch(questionText);
  }
}

async function handleQuestionViaFetch(questionText) {
  try {
    botText.textContent = 'Thinking...';
    const qResp = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: questionText })
    });
    if (!qResp.ok) throw new Error('Query failed');
    const qData = await qResp.json();
    const answer = qData.answer || 'No answer returned.';
    botText.textContent = answer;
    await playAnswerAudio(answer);
  } catch (err) {
    console.error('Error handling question via fetch', err);
    botText.textContent = 'Error contacting assistant.';
  }
}

// ========== Start / Stop assistant (mic button) ==========
async function startAssistant() {
  if (running) return;
  running = true;
  micIcon.textContent = '🟢';
  botText.textContent = 'Listening...';

  // Start recognition
  setupRecognition();

  // Load VAD and start
  try {
    const session = await loadSileroModel();
    await startVAD(session);
  } catch (err) {
    console.warn('VAD failed to start', err);
  }

  // Create WS connection
  if (USE_WEBSOCKET) createWebSocket();

  log('Assistant started (persistent listening)');
}

function stopAssistant() {
  if (!running) return;
  running = false;
  micIcon.textContent = '🎤';
  botText.textContent = 'Stopped';

  // Stop recognition
  try {
    if (recognition) {
      recognition.onresult = null;
      recognition.onend = null;
      recognition.onerror = null;
      try { recognition.stop(); } catch (e) {}
      recognition = null;
    }
  } catch (e) { log('Error stopping recognition', e); }

  // Stop VAD
  stopVAD();

  // Stop any playing audio and cancel generation
  stopSpeaking(false);

  // Optionally close ws
  try { if (ws) { ws.close(); ws = null; wsReady = false; } } catch (e) {}

  log('Assistant stopped');
}

// ========== Mic button click handler ==========
micBtn.addEventListener('click', async () => {
  if (!running) {
    await startAssistant();
  } else {
    // user clicked to stop assistant entirely
    stopAssistant();
  }
});

// ========== Initialize (do not auto-start) ==========
log('Voice assistant script loaded. Click mic to start persistent listening.');