import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiError, askQuestion, getApiBaseUrl } from '../api/client';

const PAYLOAD = {
  answer: 'Old age pension schemes provide periodic financial assistance.',
  citations: [],
  citation_labels: [],
  retrieved_chunk_ids: ['chunk-1'],
  retrieved_chunks: [],
  insufficient_context: false,
  model: 'gpt-4.1-mini',
  candidate_k: 30,
  final_k: 5,
};

function jsonResponse(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  };
}

function stubFetch(implementation) {
  const mock = vi.fn(implementation);
  vi.stubGlobal('fetch', mock);
  return mock;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('getApiBaseUrl', () => {
  it('defaults to the local Node gateway', () => {
    expect(getApiBaseUrl()).toBe('http://localhost:3000');
  });

  it('uses VITE_API_BASE_URL and strips trailing slashes', () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.test/');
    expect(getApiBaseUrl()).toBe('https://api.example.test');
  });
});

describe('askQuestion', () => {
  it('posts a trimmed question to /api/answer', async () => {
    const fetchMock = stubFetch(async () => jsonResponse(PAYLOAD));

    const result = await askQuestion('  What is old age pension?  ');

    expect(result).toEqual(PAYLOAD);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe('http://localhost:3000/api/answer');
    expect(options.method).toBe('POST');
    expect(options.headers['Content-Type']).toBe('application/json');
    expect(JSON.parse(options.body)).toEqual({
      question: 'What is old age pension?',
    });
  });

  it('honours the configured backend URL', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://127.0.0.1:4000');
    const fetchMock = stubFetch(async () => jsonResponse(PAYLOAD));

    await askQuestion('question');

    expect(fetchMock.mock.calls[0][0]).toBe('http://127.0.0.1:4000/api/answer');
  });

  it('rejects an empty question without calling the network', async () => {
    const fetchMock = stubFetch(async () => jsonResponse(PAYLOAD));

    await expect(askQuestion('   ')).rejects.toMatchObject({
      code: 'empty_question',
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('maps backend error bodies to ApiError', async () => {
    stubFetch(async () =>
      jsonResponse(
        {
          error: {
            code: 'llm_not_configured',
            message: 'The answer service is not configured.',
          },
        },
        502,
      ),
    );

    await expect(askQuestion('question')).rejects.toMatchObject({
      name: 'ApiError',
      code: 'llm_not_configured',
      status: 502,
      message: 'The answer service is not configured.',
    });
  });

  it('falls back to a status message for unreadable error bodies', async () => {
    stubFetch(async () => ({
      ok: false,
      status: 500,
      json: async () => {
        throw new Error('not json');
      },
    }));

    const error = await askQuestion('question').catch((caught) => caught);

    expect(error).toBeInstanceOf(ApiError);
    expect(error.code).toBe('request_failed');
    expect(error.message).toMatch(/HTTP 500/);
  });

  it('reports network failures with a helpful message', async () => {
    stubFetch(async () => {
      throw new TypeError('fetch failed');
    });

    await expect(askQuestion('question')).rejects.toMatchObject({
      code: 'network_error',
      message: expect.stringMatching(/could not reach the assistant service/i),
    });
  });

  it('rejects a malformed success payload', async () => {
    stubFetch(async () => jsonResponse({ citations: [] }));

    await expect(askQuestion('question')).rejects.toMatchObject({
      code: 'invalid_response',
    });
  });

  it('passes abort errors through untouched', async () => {
    stubFetch(async () => {
      const abort = new Error('aborted');
      abort.name = 'AbortError';
      throw abort;
    });

    await expect(askQuestion('question')).rejects.toMatchObject({
      name: 'AbortError',
    });
  });
});
