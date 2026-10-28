import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import App from '../App';

const ANSWER =
  'Old age pension schemes provide periodic financial assistance to eligible senior citizens.';

const SUCCESS = {
  answer: ANSWER,
  citations: [
    {
      chunk_id: 'chunk-1',
      scheme_name: 'Indira Gandhi National Old Age Pension Scheme',
      source_file: 'ignoaps.pdf',
      page_start: 1,
      page_end: 2,
      label: '[Source: ignoaps.pdf, pp. 1-2]',
    },
    {
      chunk_id: 'chunk-2',
      scheme_name: 'State Old Age Pension',
      source_file: 'state_oap.pdf',
      page_start: 4,
      page_end: 4,
      label: '[Source: state_oap.pdf, p. 4]',
    },
  ],
  citation_labels: ['[Source: ignoaps.pdf, pp. 1-2]', '[Source: state_oap.pdf, p. 4]'],
  retrieved_chunk_ids: ['chunk-1', 'chunk-2', 'chunk-3', 'chunk-4', 'chunk-5'],
  retrieved_chunks: [],
  insufficient_context: false,
  model: 'gpt-4.1-mini',
  candidate_k: 30,
  final_k: 5,
};

function okResponse(body = SUCCESS) {
  return { ok: true, status: 200, json: async () => body };
}

function errorResponse(body, status) {
  return { ok: false, status, json: async () => body };
}

let fetchMock;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

function renderApp() {
  const user = userEvent.setup();
  render(<App />);
  return {
    user,
    input: screen.getByLabelText(/your question/i),
    ask: () => screen.getByRole('button', { name: /^ask$/i }),
    clear: () => screen.getByRole('button', { name: /^clear$/i }),
  };
}

async function ask(user, input, question) {
  await user.type(input, question);
  await user.click(screen.getByRole('button', { name: /^ask$/i }));
}

describe('government scheme assistant UI', () => {
  it('renders the header, question input, and controls', () => {
    renderApp();

    expect(
      screen.getByRole('heading', {
        level: 1,
        name: /government scheme assistant/i,
      }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/your question/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^ask$/i })).toBeEnabled();
    expect(screen.getByRole('button', { name: /^clear$/i })).toBeInTheDocument();
  });

  it('submits the question to the Node gateway', async () => {
    fetchMock.mockResolvedValue(okResponse());
    const { user, input } = renderApp();

    await ask(user, input, 'What is the old age pension?');

    await screen.findByText(ANSWER);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe('http://localhost:3000/api/answer');
    expect(options.method).toBe('POST');
    expect(JSON.parse(options.body)).toEqual({
      question: 'What is the old age pension?',
    });
  });

  it('shows a loading state and disables Ask until the answer arrives', async () => {
    let resolveRequest;
    fetchMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveRequest = resolve;
        }),
    );
    const { user, input } = renderApp();

    await ask(user, input, 'What is the old age pension?');

    const pending = screen.getByRole('button', { name: /asking/i });
    expect(pending).toBeDisabled();
    expect(pending).toHaveAttribute('aria-busy', 'true');
    expect(screen.getByRole('status')).toHaveTextContent(
      /searching the scheme documents/i,
    );

    await act(async () => {
      resolveRequest(okResponse());
    });

    expect(await screen.findByText(ANSWER)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^ask$/i })).toBeEnabled();
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });

  it('displays the answer with source cards from the API metadata', async () => {
    fetchMock.mockResolvedValue(okResponse());
    const { user, input } = renderApp();

    await ask(user, input, 'What is the old age pension?');
    await screen.findByText(ANSWER);

    expect(
      screen.getByRole('heading', { name: /sources \(2\)/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByText('Indira Gandhi National Old Age Pension Scheme'),
    ).toBeInTheDocument();
    expect(screen.getByText('ignoaps.pdf')).toBeInTheDocument();
    expect(screen.getByText('1–2')).toBeInTheDocument();
    expect(screen.getByText('State Old Age Pension')).toBeInTheDocument();
    expect(screen.getByText('state_oap.pdf')).toBeInTheDocument();
    expect(screen.getByText('4')).toBeInTheDocument();
    expect(
      screen.getByText('[Source: ignoaps.pdf, pp. 1-2]'),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/grounded in 5 retrieved passages/i),
    ).toBeInTheDocument();
    expect(screen.queryByText(/gpt-4\.1-mini/)).not.toBeInTheDocument();
    expect(screen.queryByText(/model:/i)).not.toBeInTheDocument();
  });

  it('warns when the documents do not fully cover the question', async () => {
    fetchMock.mockResolvedValue(
      okResponse({ ...SUCCESS, insufficient_context: true }),
    );
    const { user, input } = renderApp();

    await ask(user, input, 'A very obscure question');
    await screen.findByText(ANSWER);

    expect(screen.getByRole('note')).toHaveTextContent(/may not fully cover/i);
  });

  it('shows a useful message when the API returns an error', async () => {
    fetchMock.mockResolvedValue(
      errorResponse(
        {
          error: {
            code: 'llm_not_configured',
            message:
              'The answer service is not configured (LLM_API_KEY is missing or invalid).',
          },
        },
        502,
      ),
    );
    const { user, input } = renderApp();

    await ask(user, input, 'What is the old age pension?');

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/not configured/i);
    expect(alert).toHaveTextContent(/llm_not_configured/);
    expect(screen.getByRole('button', { name: /^ask$/i })).toBeEnabled();
  });

  it('explains network failures and offers a retry', async () => {
    fetchMock.mockRejectedValue(new TypeError('fetch failed'));
    const { user, input } = renderApp();

    await ask(user, input, 'What is the old age pension?');

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/could not reach the assistant service/i);
    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
  });

  it('prevents empty and whitespace-only submissions', async () => {
    const { user, input } = renderApp();

    await user.click(screen.getByRole('button', { name: /^ask$/i }));
    expect(screen.getByRole('alert')).toHaveTextContent(/please enter a question/i);
    expect(fetchMock).not.toHaveBeenCalled();

    await user.type(input, '   ');
    await user.click(screen.getByRole('button', { name: /^ask$/i }));

    expect(screen.getByRole('alert')).toHaveTextContent(/please enter a question/i);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('submits with Enter', async () => {
    fetchMock.mockResolvedValue(okResponse());
    const { user, input } = renderApp();

    await user.type(input, 'old age pension{Enter}');

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      question: 'old age pension',
    });
    await screen.findByText(ANSWER);
  });

  it('submits with Ctrl+Enter', async () => {
    fetchMock.mockResolvedValue(okResponse());
    const { user, input } = renderApp();

    await user.type(input, 'scholarship{Control>}{Enter}{/Control}');

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      question: 'scholarship',
    });
  });

  it('inserts a newline on Shift+Enter instead of submitting', async () => {
    const { user, input } = renderApp();

    await user.type(input, 'first line{Shift>}{Enter}{/Shift}second line');

    expect(fetchMock).not.toHaveBeenCalled();
    expect(input).toHaveValue('first line\nsecond line');
  });

  it('clears the question, answer, and sources', async () => {
    fetchMock.mockResolvedValue(okResponse());
    const { user, input, clear } = renderApp();

    await ask(user, input, 'What is the old age pension?');
    await screen.findByText(ANSWER);

    await user.click(clear());

    expect(input).toHaveValue('');
    expect(screen.queryByText(ANSWER)).not.toBeInTheDocument();
    expect(screen.queryByText(/^sources/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('dismisses an error without clearing the question', async () => {
    fetchMock.mockRejectedValue(new TypeError('fetch failed'));
    const { user, input } = renderApp();

    await ask(user, input, 'What is the old age pension?');
    await screen.findByRole('alert');

    await user.click(screen.getByRole('button', { name: /dismiss/i }));

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(input).toHaveValue('What is the old age pension?');
  });
});
