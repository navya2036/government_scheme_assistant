'use strict';

/**
 * Express gateway for the RAG service.
 *
 * Layer position:
 *   React (future) -> this Express app -> Python FastAPI -> generation.py
 *
 * This process never touches FAISS, embeddings, SQLite, or any model. It only
 * validates input, forwards the question to the Python API, and normalises
 * errors for the browser.
 */

const express = require('express');
const cors = require('cors');

const DEFAULT_PYTHON_API_URL = 'http://localhost:8000';
// Must exceed the Python side's worst case: retrieval + reranking plus the LLM
// call with its own retries.  Override with PYTHON_API_TIMEOUT_MS.
const DEFAULT_TIMEOUT_MS = 180000;
const DEFAULT_HEALTH_TIMEOUT_MS = 5000;
const MAX_QUESTION_LENGTH = 1000;
const MAX_BODY_BYTES = '16kb';

function positiveInt(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function readConfig(env = process.env) {
  return {
    pythonApiUrl: (env.PYTHON_API_URL || DEFAULT_PYTHON_API_URL).replace(/\/+$/, ''),
    timeoutMs: positiveInt(env.PYTHON_API_TIMEOUT_MS, DEFAULT_TIMEOUT_MS),
    healthTimeoutMs: positiveInt(env.PYTHON_HEALTH_TIMEOUT_MS, DEFAULT_HEALTH_TIMEOUT_MS),
    corsOrigin: env.CORS_ORIGIN || '*',
    port: positiveInt(env.PORT, 3000),
    host: env.HOST || '0.0.0.0',
  };
}

function validateQuestion(body) {
  if (body === null || typeof body !== 'object' || Array.isArray(body)) {
    return { ok: false, message: 'Request body must be a JSON object with a "question" field.' };
  }
  const { question } = body;
  if (typeof question !== 'string') {
    return { ok: false, message: '"question" must be a string.' };
  }
  const trimmed = question.trim();
  if (trimmed.length === 0) {
    return { ok: false, message: '"question" must not be empty or whitespace.' };
  }
  if (trimmed.length > MAX_QUESTION_LENGTH) {
    return {
      ok: false,
      message: `"question" must be at most ${MAX_QUESTION_LENGTH} characters.`,
    };
  }
  return { ok: true, question: trimmed };
}

async function fetchJson(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const raw = await response.text();
    let payload = null;
    try {
      payload = raw ? JSON.parse(raw) : null;
    } catch {
      payload = null;
    }
    return { status: response.status, payload, raw };
  } finally {
    clearTimeout(timer);
  }
}

function errorBody(code, message) {
  return { error: { code, message } };
}

/** Copy only known-safe upstream error fields; never reflect arbitrary content. */
function sanitizeUpstreamError(payload, fallbackCode) {
  const upstream = payload && typeof payload === 'object' ? payload.error : null;
  const code =
    upstream && typeof upstream.code === 'string' ? upstream.code : fallbackCode;
  const message =
    upstream && typeof upstream.message === 'string'
      ? upstream.message
      : 'The answer service reported an error.';
  return errorBody(code, message);
}

function createApp(config = readConfig()) {
  const app = express();

  app.disable('x-powered-by');
  app.use(cors({ origin: config.corsOrigin }));
  app.use(express.json({ limit: MAX_BODY_BYTES }));

  app.get('/api/health', async (req, res) => {
    try {
      const upstream = await fetchJson(
        `${config.pythonApiUrl}/health`,
        { method: 'GET', headers: { Accept: 'application/json' } },
        config.healthTimeoutMs,
      );
      const healthy = upstream.status >= 200 && upstream.status < 300 && upstream.payload;
      return res.status(healthy ? 200 : 503).json({
        status: healthy ? 'ok' : 'degraded',
        python_api: upstream.payload,
      });
    } catch (error) {
      return res.status(503).json({
        status: 'degraded',
        python_api: null,
        ...errorBody('python_api_unavailable', 'The answer service is not reachable.'),
      });
    }
  });

  app.post('/api/answer', async (req, res) => {
    const validation = validateQuestion(req.body);
    if (!validation.ok) {
      return res.status(400).json(errorBody('invalid_question', validation.message));
    }

    let upstream;
    try {
      upstream = await fetchJson(
        `${config.pythonApiUrl}/answer`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
          body: JSON.stringify({ question: validation.question }),
        },
        config.timeoutMs,
      );
    } catch (error) {
      if (error && error.name === 'AbortError') {
        return res.status(504).json(
          errorBody('python_api_timeout', 'The answer service took too long to respond.'),
        );
      }
      return res.status(503).json(
        errorBody('python_api_unavailable', 'The answer service is not reachable.'),
      );
    }

    if (upstream.payload === null) {
      return res.status(502).json(
        errorBody('python_api_invalid_response', 'The answer service returned an unreadable response.'),
      );
    }

    if (upstream.status >= 200 && upstream.status < 300) {
      return res.status(200).json(upstream.payload);
    }

    if (upstream.status >= 400 && upstream.status < 500) {
      return res
        .status(upstream.status)
        .json(sanitizeUpstreamError(upstream.payload, 'invalid_request'));
    }

    return res
      .status(502)
      .json(sanitizeUpstreamError(upstream.payload, 'python_api_error'));
  });

  app.use((req, res) => {
    res.status(404).json(errorBody('not_found', 'Unknown endpoint.'));
  });

  // Body-parser and other middleware errors.
  app.use((err, req, res, next) => {
    if (err && (err.type === 'entity.parse.failed' || err instanceof SyntaxError)) {
      return res.status(400).json(errorBody('invalid_json', 'Request body must be valid JSON.'));
    }
    if (err && err.type === 'entity.too.large') {
      return res.status(413).json(errorBody('payload_too_large', 'Request body is too large.'));
    }
    return res.status(500).json(errorBody('internal_error', 'Unexpected server error.'));
  });

  return app;
}

function start() {
  const config = readConfig();
  const app = createApp(config);
  return app.listen(config.port, config.host, () => {
    console.log(
      `Express gateway listening on http://${config.host}:${config.port} -> ${config.pythonApiUrl}`,
    );
  });
}

if (require.main === module) {
  start();
}

module.exports = { createApp, readConfig, validateQuestion, MAX_QUESTION_LENGTH };
