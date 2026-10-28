/**
 * The only module that talks to the backend.
 *
 * The browser calls the Node/Express gateway, never the Python service and
 * never anything model-related.  The base URL comes from `VITE_API_BASE_URL`
 * so no component hardcodes it, and only public values are read — a Vite
 * client bundle must never contain provider credentials or backend secrets.
 */

const DEFAULT_BASE_URL = 'http://localhost:3000';

export class ApiError extends Error {
  constructor(message, { status = 0, code = 'request_failed' } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

export function getApiBaseUrl() {
  const configured =
    typeof import.meta !== 'undefined' && import.meta.env
      ? import.meta.env.VITE_API_BASE_URL
      : undefined;
  return (configured || DEFAULT_BASE_URL).trim().replace(/\/+$/, '');
}

/**
 * Ask the gateway a question.
 *
 * Resolves with the gateway's JSON payload (the contract is owned by the
 * backend), or throws an {@link ApiError} with a human-readable message.
 * Abort errors are re-thrown untouched so callers can ignore cancellations.
 */
export async function askQuestion(question, { signal } = {}) {
  const trimmed = typeof question === 'string' ? question.trim() : '';
  if (!trimmed) {
    throw new ApiError('Please enter a question before asking.', {
      code: 'empty_question',
    });
  }

  let response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ question: trimmed }),
      signal,
    });
  } catch (error) {
    if (error && error.name === 'AbortError') {
      throw error;
    }
    throw new ApiError(
      'Could not reach the assistant service. Check that the API is running, then try again.',
      { code: 'network_error' },
    );
  }

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const upstream = payload && typeof payload === 'object' ? payload.error : null;
    const message =
      upstream && typeof upstream.message === 'string' && upstream.message.trim()
        ? upstream.message
        : `The assistant service returned an error (HTTP ${response.status}).`;
    const code =
      upstream && typeof upstream.code === 'string' ? upstream.code : 'request_failed';
    throw new ApiError(message, { status: response.status, code });
  }

  if (!payload || typeof payload.answer !== 'string') {
    throw new ApiError(
      'The assistant returned an unexpected response. Please try again.',
      { status: response.status, code: 'invalid_response' },
    );
  }

  return payload;
}
