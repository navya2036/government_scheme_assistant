import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import AnswerPanel from '../components/AnswerPanel';

const ANSWER = [
  '### Old Age Pension',
  '- Benefit: **₹2 lakh** per year.',
  '- Apply at the local office.',
].join('\n');

const RESULT = {
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
  ],
  citation_labels: ['[Source: ignoaps.pdf, pp. 1-2]'],
  retrieved_chunk_ids: ['chunk-1', 'chunk-2'],
  retrieved_chunks: [],
  insufficient_context: false,
  model: 'provider/model-x',
  candidate_k: 30,
  final_k: 5,
};

let synth;

class MockSpeechSynthesisUtterance {
  constructor(text) {
    this.text = text;
  }
}

function renderPanel(overrides = {}) {
  const user = userEvent.setup();
  render(<AnswerPanel result={{ ...RESULT, ...overrides }} />);
  return {
    user,
    read: () => screen.getByRole('button', { name: /read answer/i }),
    stop: () => screen.getByRole('button', { name: /^stop$/i }),
  };
}

beforeEach(() => {
  synth = { speak: vi.fn(), cancel: vi.fn() };
  vi.stubGlobal('speechSynthesis', synth);
  vi.stubGlobal('SpeechSynthesisUtterance', MockSpeechSynthesisUtterance);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('voice output', () => {
  it('does not speak automatically', () => {
    renderPanel();

    expect(synth.speak).not.toHaveBeenCalled();
    expect(synth.cancel).not.toHaveBeenCalled();
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });

  it('reads only the generated answer when Read Answer is clicked', async () => {
    const { user, read, stop } = renderPanel();

    await user.click(read());

    expect(synth.cancel).toHaveBeenCalledTimes(1);
    expect(synth.speak).toHaveBeenCalledTimes(1);
    const utterance = synth.speak.mock.calls[0][0];
    expect(utterance.text).toContain('Old Age Pension');
    expect(utterance.text).toContain('Benefit: ₹2 lakh per year.');
    expect(utterance.text).toContain('Apply at the local office.');
    expect(utterance.text).not.toContain('ignoaps.pdf');
    expect(utterance.text).not.toContain(
      'Indira Gandhi National Old Age Pension Scheme',
    );
    expect(utterance.text).not.toContain('[Source:');
    expect(screen.getByRole('status')).toHaveTextContent(
      /reading the answer aloud/i,
    );
    expect(stop()).toBeEnabled();
  });

  it('cancels existing speech before starting a new utterance', async () => {
    const { user, read } = renderPanel();

    await user.click(read());
    await user.click(read());

    expect(synth.cancel).toHaveBeenCalledTimes(2);
    expect(synth.speak).toHaveBeenCalledTimes(2);
    expect(synth.cancel.mock.invocationCallOrder[1]).toBeLessThan(
      synth.speak.mock.invocationCallOrder[1],
    );
    expect(synth.speak.mock.calls[0][0]).not.toBe(
      synth.speak.mock.calls[1][0],
    );
  });

  it('stops speaking and cancels the utterance', async () => {
    const { user, read, stop } = renderPanel();

    await user.click(read());
    await user.click(stop());

    expect(synth.cancel).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(stop()).toBeDisabled();
  });

  it('returns to idle when the utterance ends', async () => {
    const { user, read, stop } = renderPanel();

    await user.click(read());
    const utterance = synth.speak.mock.calls[0][0];
    act(() => utterance.onend());

    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(stop()).toBeDisabled();
  });

  it('reports a speech synthesis failure', async () => {
    const { user, read } = renderPanel();

    await user.click(read());
    const utterance = synth.speak.mock.calls[0][0];
    act(() => utterance.onerror({ error: 'synthesis-failed' }));

    expect(screen.getByRole('alert')).toHaveTextContent(
      /could not be read aloud/i,
    );
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });

  it('disables Read Answer when speech synthesis is unavailable', () => {
    vi.stubGlobal('speechSynthesis', undefined);
    vi.stubGlobal('SpeechSynthesisUtterance', undefined);
    renderPanel();

    expect(screen.getByRole('button', { name: /read answer/i })).toBeDisabled();
    expect(
      screen.getByText(/reading answers aloud is not supported/i),
    ).toBeInTheDocument();
  });

  it('disables Read Answer for an empty answer', () => {
    renderPanel({ answer: '   ' });

    expect(screen.getByRole('button', { name: /read answer/i })).toBeDisabled();
    expect(
      screen.getByText(/returned an empty answer/i),
    ).toBeInTheDocument();
  });
});
