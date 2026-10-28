'use strict';

/**
 * Express gateway tests. The Python API is replaced with a local stub server,
 * so these tests need no RAG models, no API key, and no network.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');

const { createApp, validateQuestion } = require('../src/server');

const CANNED = {
  answer: 'Old age pension schemes provide periodic financial assistance.',
  citations: [
    {
      chunk_id: 'chunk-1',
      scheme_name: 'Old Age Assistance',
      source_file: 'oaa.pdf',
      page_start: 1,
      page_end: 1,
      label: '[Source: oaa.pdf, p. 1]',
    },
  ],
  citation_labels: ['[Source: oaa.pdf, p. 1]'],
  retrieved_chunk_ids: ['chunk-1', 'chunk-2'],
  retrieved_chunks: [
    {
      chunk_id: 'chunk-1',
      scheme_name: 'Old Age Assistance',
      source_file: 'oaa.pdf',
      page_start: 1,
      page_end: 1,
      final_rank: 1,
      faiss_similarity: 0.89,
      reranker_score: 7.5,
    },
  ],
  insufficient_context: false,
  model: 'gpt-4.1-mini',
  candidate_k: 30,
  final_k: 5,
};

function startStub(handler) {
  return new Promise((resolve) => {
    const requests = [];
    const server = http.createServer((req, res) => {
      let body = '';
      req.on('data', (chunk) => {
        body += chunk;
      });
      req.on('end', () => {
        let parsed = null;
        try {
          parsed = body ? JSON.parse(body) : null;
        } catch {
          parsed = body;
        }
        requests.push({ method: req.method, url: req.url, body: parsed });
        handler(req, res, requests);
      });
    });
    server.listen(0, '127.0.0.1', () => {
      resolve({
        url: `http://127.0.0.1:${server.address().port}`,
        requests,
        close: () => new Promise((done) => server.close(done)),
      });
    });
  });
}

function startApp(config) {
  return new Promise((resolve) => {
    const app = createApp(config);
    const server = app.listen(0, '127.0.0.1', () => {
      resolve({
        url: `http://127.0.0.1:${server.address().port}`,
        close: () => new Promise((done) => server.close(done)),
      });
    });
  });
}

function jsonHandler(status, payload) {
  return (req, res) => {
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(payload));
  };
}

test('valid question is forwarded and the response is preserved', async (t) => {
  const stub = await startStub(jsonHandler(200, CANNED));
  const app = await startApp({ pythonApiUrl: stub.url, timeoutMs: 5000 });
  t.after(async () => {
    await app.close();
    await stub.close();
  });

  const response = await fetch(`${app.url}/api/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: '  What is old age pension?  ' }),
  });
  const body = await response.json();

  assert.equal(response.status, 200);
  assert.deepEqual(body, CANNED);
  assert.equal(stub.requests.length, 1);
  assert.equal(stub.requests[0].method, 'POST');
  assert.equal(stub.requests[0].url, '/answer');
  assert.deepEqual(stub.requests[0].body, { question: 'What is old age pension?' });
});

test('empty question is rejected before reaching Python', async (t) => {
  const stub = await startStub(jsonHandler(200, CANNED));
  const app = await startApp({ pythonApiUrl: stub.url, timeoutMs: 5000 });
  t.after(async () => {
    await app.close();
    await stub.close();
  });

  for (const question of ['', '   ', null, 42]) {
    const response = await fetch(`${app.url}/api/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const body = await response.json();
    assert.equal(response.status, 400);
    assert.equal(body.error.code, 'invalid_question');
  }

  const missing = await fetch(`${app.url}/api/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
  assert.equal(missing.status, 400);
  assert.equal(stub.requests.length, 0);
});

test('over-long question is rejected', async (t) => {
  const stub = await startStub(jsonHandler(200, CANNED));
  const app = await startApp({ pythonApiUrl: stub.url, timeoutMs: 5000 });
  t.after(async () => {
    await app.close();
    await stub.close();
  });

  const response = await fetch(`${app.url}/api/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: 'x'.repeat(1001) }),
  });
  assert.equal(response.status, 400);
});

test('malformed JSON body returns 400', async (t) => {
  const stub = await startStub(jsonHandler(200, CANNED));
  const app = await startApp({ pythonApiUrl: stub.url, timeoutMs: 5000 });
  t.after(async () => {
    await app.close();
    await stub.close();
  });

  const response = await fetch(`${app.url}/api/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{"question": "unterminated',
  });
  assert.equal(response.status, 400);
  const body = await response.json();
  assert.equal(body.error.code, 'invalid_json');
  assert.equal(stub.requests.length, 0);
});

test('unavailable Python service produces a clear error', async (t) => {
  const app = await startApp({ pythonApiUrl: 'http://127.0.0.1:1', timeoutMs: 2000 });
  t.after(() => app.close());

  const response = await fetch(`${app.url}/api/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: 'old age pension' }),
  });
  const body = await response.json();
  assert.equal(response.status, 503);
  assert.equal(body.error.code, 'python_api_unavailable');
});

test('slow Python service produces a timeout error', async (t) => {
  const stub = await startStub(() => {
    /* never respond */
  });
  const app = await startApp({ pythonApiUrl: stub.url, timeoutMs: 150 });
  t.after(async () => {
    await app.close();
    await stub.close();
  });

  const response = await fetch(`${app.url}/api/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: 'old age pension' }),
  });
  const body = await response.json();
  assert.equal(response.status, 504);
  assert.equal(body.error.code, 'python_api_timeout');
});

test('upstream 4xx/503 semantics pass through', async (t) => {
  const notConfigured = {
    error: { code: 'llm_not_configured', message: 'LLM_API_KEY is missing.' },
  };
  const stub = await startStub(jsonHandler(503, notConfigured));
  const app = await startApp({ pythonApiUrl: stub.url, timeoutMs: 5000 });
  t.after(async () => {
    await app.close();
    await stub.close();
  });

  const response = await fetch(`${app.url}/api/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: 'old age pension' }),
  });
  // Python 5xx is normalised to 502, preserving the machine-readable code.
  assert.equal(response.status, 502);
  const body = await response.json();
  assert.equal(body.error.code, 'llm_not_configured');
});

test('upstream 422 validation error passes through', async (t) => {
  const stub = await startStub(
    jsonHandler(422, { error: { code: 'invalid_question', message: 'bad input' } }),
  );
  const app = await startApp({ pythonApiUrl: stub.url, timeoutMs: 5000 });
  t.after(async () => {
    await app.close();
    await stub.close();
  });

  const response = await fetch(`${app.url}/api/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: 'old age pension' }),
  });
  assert.equal(response.status, 422);
  const body = await response.json();
  assert.equal(body.error.code, 'invalid_question');
});

test('health reflects Python availability', async (t) => {
  const healthy = await startStub(jsonHandler(200, { status: 'ok', llm: { configured: true } }));
  const app = await startApp({ pythonApiUrl: healthy.url, healthTimeoutMs: 2000 });
  t.after(async () => {
    await app.close();
    await healthy.close();
  });

  const response = await fetch(`${app.url}/api/health`);
  const body = await response.json();
  assert.equal(response.status, 200);
  assert.equal(body.status, 'ok');
  assert.equal(body.python_api.status, 'ok');
});

test('health reports degraded when Python is down', async (t) => {
  const app = await startApp({ pythonApiUrl: 'http://127.0.0.1:1', healthTimeoutMs: 1000 });
  t.after(() => app.close());

  const response = await fetch(`${app.url}/api/health`);
  const body = await response.json();
  assert.equal(response.status, 503);
  assert.equal(body.status, 'degraded');
  assert.equal(body.error.code, 'python_api_unavailable');
});

test('CORS is enabled for the future React client', async (t) => {
  const stub = await startStub(jsonHandler(200, CANNED));
  const app = await startApp({ pythonApiUrl: stub.url, timeoutMs: 5000, corsOrigin: '*' });
  t.after(async () => {
    await app.close();
    await stub.close();
  });

  const response = await fetch(`${app.url}/api/health`, {
    headers: { Origin: 'http://localhost:5173' },
  });
  assert.equal(response.headers.get('access-control-allow-origin'), '*');
});

test('unknown route returns a structured 404', async (t) => {
  const stub = await startStub(jsonHandler(200, CANNED));
  const app = await startApp({ pythonApiUrl: stub.url, timeoutMs: 5000 });
  t.after(async () => {
    await app.close();
    await stub.close();
  });

  const response = await fetch(`${app.url}/api/nope`);
  const body = await response.json();
  assert.equal(response.status, 404);
  assert.equal(body.error.code, 'not_found');
});

test('responses never contain secret material', async (t) => {
  const stub = await startStub(jsonHandler(200, CANNED));
  const app = await startApp({ pythonApiUrl: stub.url, timeoutMs: 5000 });
  t.after(async () => {
    await app.close();
    await stub.close();
  });

  const response = await fetch(`${app.url}/api/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: 'old age pension' }),
  });
  const raw = await response.text();
  assert.doesNotMatch(raw, /sk-[A-Za-z0-9]/);
  assert.doesNotMatch(raw, /LLM_API_KEY/);
});

test('question validation helper rejects non-object bodies', () => {
  assert.equal(validateQuestion(null).ok, false);
  assert.equal(validateQuestion([]).ok, false);
  assert.equal(validateQuestion('text').ok, false);
  assert.equal(validateQuestion({ question: 'ok' }).ok, true);
});
