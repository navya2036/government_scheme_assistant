import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import App from '../App';

let fetchMock;
let recognitionInstances;

class MockSpeechRecognition {
  constructor() {
    this.start = vi.fn(() => {
      this.onstart?.();
    });
    this.stop = vi.fn(() => {
      this.onend?.();
    });
    this.abort = vi.fn(() => {
      this.onend?.();
    });
    recognitionInstances.push(this);
  }
}

function installRecognition() {
  recognitionInstances = [];
  vi.stubGlobal('SpeechRecognition', MockSpeechRecognition);
}

function resultEvent(transcript, { isFinal = true, resultIndex = 0 } = {}) {
  const result = [{ transcript, confidence: 0.95 }];
  result.isFinal = isFinal;
  return { resultIndex, results: [result] };
}

function renderApp() {
  const user = userEvent.setup();
  render(<App />);
  return {
    user,
    input: screen.getByLabelText(/your question/i),
    mic: () =>
      screen.getByRole('button', {
        name: /start voice input|stop voice input|voice input is not supported/i,
      }),
  };
}

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('voice input', () => {
  it('starts listening and shows a clear listening state', async () => {
    installRecognition();
    const { user, mic } = renderApp();

    await user.click(mic());

    expect(recognitionInstances).toHaveLength(1);
    expect(recognitionInstances[0].start).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('status')).toHaveTextContent(/listening/i);
    expect(
      screen.getByRole('button', { name: /stop voice input/i }),
    ).toHaveAttribute('aria-pressed', 'true');
    expect(
      screen.getByRole('button', { name: /stop voice input/i }),
    ).toHaveTextContent(/listening/i);
  });

  it('puts a final transcript into the question without submitting it', async () => {
    installRecognition();
    const { user, input, mic } = renderApp();

    await user.click(mic());
    act(() => {
      recognitionInstances[0].onresult(resultEvent('old age pension'));
    });

    expect(input).toHaveValue('old age pension');
    expect(fetchMock).not.toHaveBeenCalled();
    expect(screen.getByRole('status')).toHaveTextContent(/listening/i);
  });

  it('appends final speech to text the user already typed', async () => {
    installRecognition();
    const { user, input, mic } = renderApp();

    await user.type(input, 'Tell me about');
    await user.click(mic());

    act(() => {
      recognitionInstances[0].onresult(resultEvent('old age pension'));
    });
    act(() => {
      recognitionInstances[0].onresult(resultEvent('eligibility'));
    });

    expect(input).toHaveValue('Tell me about old age pension eligibility');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('never inserts interim or empty transcripts', async () => {
    installRecognition();
    const { user, input, mic } = renderApp();

    await user.click(mic());

    act(() => {
      recognitionInstances[0].onresult(
        resultEvent('old age', { isFinal: false }),
      );
    });
    expect(input).toHaveValue('');
    expect(screen.getByRole('status')).toHaveTextContent(/old age/);

    act(() => {
      recognitionInstances[0].onresult(resultEvent('   '));
    });
    expect(input).toHaveValue('');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('stops listening when the microphone button is clicked again', async () => {
    installRecognition();
    const { user, input, mic } = renderApp();

    await user.click(mic());
    const recognition = recognitionInstances[0];

    await user.click(mic());

    expect(recognition.stop).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /start voice input/i }),
    ).toHaveAttribute('aria-pressed', 'false');
    expect(input).toHaveValue('');
  });

  it('explains when microphone permission is denied', async () => {
    installRecognition();
    const { user, mic } = renderApp();

    await user.click(mic());
    act(() => {
      recognitionInstances[0].onerror({ error: 'not-allowed' });
    });

    expect(screen.getByRole('alert')).toHaveTextContent(
      /microphone access was denied/i,
    );
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /start voice input/i }),
    ).toBeEnabled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('handles other recognition errors without submitting', async () => {
    installRecognition();
    const { user, mic } = renderApp();

    await user.click(mic());
    act(() => {
      recognitionInstances[0].onerror({ error: 'no-speech' });
    });

    expect(screen.getByRole('alert')).toHaveTextContent(
      /no speech was detected/i,
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('supports the prefixed webkitSpeechRecognition API', async () => {
    recognitionInstances = [];
    vi.stubGlobal('SpeechRecognition', undefined);
    vi.stubGlobal('webkitSpeechRecognition', MockSpeechRecognition);
    const { user, mic } = renderApp();

    await user.click(mic());

    expect(recognitionInstances).toHaveLength(1);
    expect(recognitionInstances[0].start).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('status')).toHaveTextContent(/listening/i);
  });

  it('disables voice input and explains when the browser does not support it', () => {
    vi.stubGlobal('SpeechRecognition', undefined);
    vi.stubGlobal('webkitSpeechRecognition', undefined);
    renderApp();

    expect(
      screen.getByRole('button', {
        name: /voice input is not supported/i,
      }),
    ).toBeDisabled();
    expect(
      screen.getByText(/voice input is not supported in this browser/i),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/your question/i)).toBeEnabled();
  });

  it('stops listening when the question is cleared', async () => {
    installRecognition();
    const { user, input, mic } = renderApp();

    await user.click(mic());
    const recognition = recognitionInstances[0];
    await user.click(screen.getByRole('button', { name: /^clear$/i }));

    expect(recognition.stop).toHaveBeenCalledTimes(1);
    expect(input).toHaveValue('');
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });
});
