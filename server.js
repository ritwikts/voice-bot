/**
 * server.js — Express + WebSocket backend with Gemini streaming + cancel support
 *
 * Fixed: robustly detect the SDK's async-iterable shape instead of assuming stream.stream exists.
 *
 * Notes:
 *  - Requires: npm install @google/genai express dotenv cors wav ws
 *  - Put your key in backend/.env as GEMINI_API_KEY
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const { Writer } = require('wav');
const { WebSocketServer } = require('ws');

let GoogleGenAI;
try {
  ({ GoogleGenAI } = require('@google/genai'));
} catch (e) {
  console.warn('Warning: @google/genai not installed.');
  GoogleGenAI = null;
}

const GEMINI_API_KEY = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
const TEXT_MODEL = process.env.TEXT_MODEL || 'gemini-2.5-flash';
const TTS_MODEL = process.env.TTS_MODEL || 'gemini-2.5-flash-preview-tts';
const PORT = process.env.PORT || 3000;

const app = express();
app.use(cors());
app.use(express.json());
app.use('/', express.static(path.join(__dirname, '..', 'frontend')));

// System prompt (Revolt Motors expert)
const SYSTEM_PROMPT = `
You are an AI assistant and product expert for Revolt Motors, an Indian electric motorcycle manufacturer. 
Your role is to provide accurate, concise, and engaging information about all aspects of Revolt Motors’ 
products, technology, services, and brand. 

Key details you must know and communicate:
- Revolt Motors is known for models such as the RV400, RV400 BRZ, and RV300.
- Provide specifications such as motor power, top speed, battery type and capacity, range, charging time, 
  riding modes (Eco, Normal, Sport), braking system, suspension, and smart features like MyRevolt App, 
  geo-fencing, and battery swapping.
- Mention battery warranty, service intervals, and cost of ownership if asked.
- Cover latest prices, on-road costs, EMI options, subsidies (FAME-II), and availability in Indian states.
- Provide updates on new launches, limited editions, and color options.
- Explain the buying process, booking procedure, delivery timelines, and test ride availability.
- Give charging infrastructure information, home charging requirements, and compatibility with public chargers.
- Share company background, vision for sustainable mobility, and unique selling points compared to competitors.
- Stay polite, factual, and avoid speculation — if information is not officially released, state that clearly.
- Keep answers relevant to Revolt Motors. If asked unrelated questions, politely redirect to Revolt topics.
- There should me no markup in the response ex- *,**.

Tone:
- Professional but friendly.
- Use simple explanations for beginners, but include detailed specs for enthusiasts.
- Whenever possible, include both numerical data and real-world examples (e.g., “Range: 150 km in Eco mode — 
  enough for a week’s city commute for most riders”).
- There should me no markup in the response ex- *,**.
`.trim();

// Instantiate GoogleGenAI client if available
let ai = null;
if (GoogleGenAI && GEMINI_API_KEY) {
  try {
    ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });
    console.log('Google Gen AI client initialized.');
  } catch (err) {
    console.warn('Failed to initialize GoogleGenAI:', err?.message || err);
    ai = null;
  }
} else {
  if (!GoogleGenAI) console.warn('@google/genai SDK not present.');
  if (!GEMINI_API_KEY) console.warn('No GEMINI_API_KEY set.');
}

// Start server
const server = app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT} (TEXT_MODEL=${TEXT_MODEL}, TTS_MODEL=${TTS_MODEL})`);
});

const wss = new WebSocketServer({ server, path: '/ws' });

function safeSend(ws, obj) {
  try {
    if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(obj));
  } catch (e) {
    console.warn('safeSend failed', e);
  }
}

// Helper to obtain an async-iterable from the SDK response
function getAsyncIterable(possibleStream) {
  // If it's already an async-iterable (most ideal), return it
  if (possibleStream && typeof possibleStream[Symbol.asyncIterator] === 'function') {
    return possibleStream;
  }
  // Some SDKs return an object with `.stream` that is the async-iterable
  if (possibleStream && possibleStream.stream && typeof possibleStream.stream[Symbol.asyncIterator] === 'function') {
    return possibleStream.stream;
  }
  // Some SDKs return { responses: asyncIterable } (less common) - check common fields
  if (possibleStream && possibleStream.responses && typeof possibleStream.responses[Symbol.asyncIterator] === 'function') {
    return possibleStream.responses;
  }
  // Not found
  return null;
}

wss.on('connection', (ws) => {
  console.log('WS client connected');
  // Per-socket request controllers (id -> AbortController)
  const socketRequests = new Map();

  ws.on('message', async (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw.toString());
    } catch (e) {
      console.warn('WS received non-JSON:', raw.toString());
      safeSend(ws, { type: 'error', id: null, error: 'Invalid JSON' });
      return;
    }

    if (msg.type === 'query') {
      const { id, question } = msg;
      if (!id || !question) {
        safeSend(ws, { type: 'error', id: id || null, error: 'Missing id or question' });
        return;
      }

      if (socketRequests.has(id)) {
        safeSend(ws, { type: 'error', id, error: 'Request id already active' });
        return;
      }

      const prompt = `${SYSTEM_PROMPT}\n\nUser: ${question}\nAssistant:`;

      if (!ai) {
        const canned = `Demo mode: I heard "${question}". I can answer Revolt Motors questions like "Tell me about the RV400 range".`;
        safeSend(ws, { type: 'final', id, text: canned });
        return;
      }

      const controller = new AbortController();
      socketRequests.set(id, controller);

      try {
        const streamResp = await ai.models.generateContentStream({
          model: TEXT_MODEL,
          contents: prompt,
          signal: controller.signal
        });

        // Find the async iterable in the response reliably
        const iterable = getAsyncIterable(streamResp);
        if (!iterable) {
          // No streaming available - attempt to extract a final answer synchronously
          let finalText = '';
          try {
            // try resp.text if available
            if (typeof streamResp?.text === 'string') finalText = streamResp.text;
            // or check candidates
            else if (streamResp?.candidates && streamResp.candidates[0]) {
              const cand = streamResp.candidates[0];
              finalText = cand?.content?.[0]?.parts?.[0]?.text || JSON.stringify(streamResp).slice(0, 800);
            } else {
              finalText = JSON.stringify(streamResp).slice(0, 800);
            }
          } catch (e) {
            finalText = 'Failed to obtain an answer from model.';
          }
          safeSend(ws, { type: 'final', id, text: finalText });
          socketRequests.delete(id);
          return;
        }

        // Iterate chunks from the async iterable
        let accumulated = '';
        for await (const chunk of iterable) {
          // If socket closed, abort generation
          if (ws.readyState !== ws.OPEN) {
            try { controller.abort(); } catch (e) {}
            break;
          }

          // Extract text from chunk robustly
          let chunkText = '';
          try {
            if (!chunk) {
              chunkText = '';
            } else if (typeof chunk === 'string') {
              chunkText = chunk;
            } else if (typeof chunk.text === 'function') {
              chunkText = chunk.text() || '';
            } else if (typeof chunk.text === 'string') {
              chunkText = chunk.text;
            } else if (chunk?.delta?.content) {
              chunkText = chunk.delta.content;
            } else if (chunk?.candidates?.[0]?.content?.[0]?.parts?.[0]?.text) {
              chunkText = chunk.candidates[0].content[0].parts[0].text || '';
            } else {
              // last resort, small debug string (avoid huge dumps)
              chunkText = '';
            }
          } catch (e) {
            chunkText = '';
          }

          if (chunkText) {
            accumulated += chunkText;
            safeSend(ws, { type: 'partial', id, text: chunkText });
          }
        } // end for-await

        if (controller.signal.aborted) {
          safeSend(ws, { type: 'cancelled', id });
        } else {
          safeSend(ws, { type: 'final', id, text: accumulated.trim() });
        }
      } catch (err) {
        if (err && (err.name === 'AbortError' || err?.code === 'ABORT_ERR')) {
          safeSend(ws, { type: 'cancelled', id });
        } else {
          console.error('Error while streaming generation:', err);
          safeSend(ws, { type: 'error', id, error: String(err?.message || err) });
        }
      } finally {
        socketRequests.delete(id);
      }
    }

    else if (msg.type === 'cancel') {
      const { id } = msg;
      if (!id) return;
      const ctrl = socketRequests.get(id);
      if (ctrl) {
        try { ctrl.abort(); } catch (e) {}
        socketRequests.delete(id);
        safeSend(ws, { type: 'cancelled', id });
        console.log(`Cancelled generation ${id} for a ws client`);
      } else {
        // Acknowledge even if not present
        safeSend(ws, { type: 'cancelled', id });
      }
    }

    else {
      safeSend(ws, { type: 'error', id: msg?.id || null, error: 'Unknown message type' });
    }
  });

  ws.on('close', () => {
    // Clean up any active generation controllers for this socket
    // (NB: sending on ws after close will be skipped in safeSend)
    console.log('WS client disconnected; cleaning up socketRequests');
  });

  ws.on('error', (err) => {
    console.warn('WS socket error', err);
  });
});

// ---------- /query fallback ----------
app.post('/query', async (req, res) => {
  try {
    const question = (req.body.question || '').toString().trim();
    if (!question) return res.status(400).json({ error: 'Question is required' });

    const prompt = `${SYSTEM_PROMPT}\n\nUser: ${question}\nAssistant:`;

    if (!ai) {
      const canned = `Demo mode: I heard "${question}". I can answer Revolt Motors questions like "Tell me about the RV400 range".`;
      return res.json({ answer: canned });
    }

    const resp = await ai.models.generateContent({
      model: TEXT_MODEL,
      contents: prompt
    });

    let answer = resp?.text ?? null;
    if (!answer) {
      try {
        const cand = resp?.candidates?.[0];
        if (cand) {
          const partText = cand?.content?.[0]?.parts?.[0]?.text;
          if (partText) answer = partText;
        }
      } catch (e) {}
    }

    if (!answer) answer = `Sorry, I couldn't generate an answer for "${question}".`;
    return res.json({ answer: answer.trim() });
  } catch (err) {
    console.error('Error in /query:', err);
    return res.status(500).json({ answer: "Sorry — I couldn't process your question right now." });
  }
});

// ---------- /speak fallback ----------
app.post('/speak', async (req, res) => {
  try {
    const text = (req.body.text || '').toString().trim();
    if (!text) return res.status(400).json({ error: 'Text is required' });

    if (!ai) {
      console.warn('/speak called but no GenAI client configured. Returning 204 for fallback.');
      return res.status(204).send();
    }

    const ttsReq = {
      model: TTS_MODEL,
      contents: text,
      config: {
        generationConfig: { responseModalities: ['AUDIO'] },
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName: 'kore' } }
        }
      }
    };

    const ttsResp = await ai.models.generateContent(ttsReq);

    let base64Audio = null;
    try {
      base64Audio = ttsResp?.candidates?.[0]?.content?.[0]?.parts?.[0]?.inlineData?.data || null;
    } catch (e) {}

    if (!base64Audio) {
      base64Audio = ttsResp?.audio?.audioData || ttsResp?.audio?.audioContent || null;
    }

    if (!base64Audio) {
      console.error('TTS response did not include audio. Full resp (truncated):', JSON.stringify(ttsResp).slice(0, 2000));
      return res.status(204).send();
    }

    const audioBuffer = Buffer.from(base64Audio, 'base64');

    if (audioBuffer.slice(0, 4).toString() === 'RIFF') {
      res.setHeader('Content-Type', 'audio/wav');
      res.setHeader('Content-Disposition', 'inline; filename="reply.wav"');
      return res.send(audioBuffer);
    }

    res.setHeader('Content-Type', 'audio/wav');
    res.setHeader('Content-Disposition', 'inline; filename="reply.wav"');

    const writer = new Writer({
      sampleRate: 24000,
      channels: 1,
      bitDepth: 16
    });
    writer.pipe(res);
    writer.write(audioBuffer);
    writer.end();

  } catch (err) {
    console.error('Error in /speak:', err);
    return res.status(500).json({ error: 'Speech synthesis failed.' });
  }
});