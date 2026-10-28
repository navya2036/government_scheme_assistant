import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import AnswerPanel from '../components/AnswerPanel';

const RESULT = {
  answer: '',
  citations: [],
  citation_labels: [],
  retrieved_chunk_ids: [],
  retrieved_chunks: [],
  insufficient_context: false,
  model: 'provider/model-x',
  candidate_k: 30,
  final_k: 5,
};

const CITATION = {
  chunk_id: 'chunk-42',
  scheme_name: 'Indira Gandhi National Old Age Pension Scheme',
  source_file: 'ignoaps.pdf',
  page_start: 7,
  page_end: 9,
  label: '[Source: ignoaps.pdf, pp. 7-9]',
};

function renderPanel(overrides = {}) {
  return render(<AnswerPanel result={{ ...RESULT, ...overrides }} />);
}

describe('AnswerPanel formatting', () => {
  it('renders markdown headings as separate readable sections', () => {
    const { container } = renderPanel({
      answer: [
        '### Indira Gandhi National Old Age Pension Scheme',
        'Provides a monthly pension to eligible senior citizens.',
        '',
        '### State Old Age Pension',
        'Tops up the central scheme for state residents.',
      ].join('\n'),
    });

    expect(
      screen.getByRole('heading', {
        level: 3,
        name: 'Indira Gandhi National Old Age Pension Scheme',
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('heading', { level: 3, name: 'State Old Age Pension' }),
    ).toBeInTheDocument();
    expect(
      screen.getByText('Provides a monthly pension to eligible senior citizens.'),
    ).toBeInTheDocument();
    expect(container.querySelectorAll('.answer-section')).toHaveLength(2);
    expect(container.textContent).not.toContain('###');
  });

  it('treats bold scheme lines as headings and bullet details as a list', () => {
    renderPanel({
      answer: [
        '**Pradhan Mantri Jeevan Jyoti Bima Yojana**',
        '- Eligibility: members of self-help groups',
        '- **Benefit:** life cover of ₹2 lakh',
      ].join('\n'),
    });

    expect(
      screen.getByRole('heading', {
        level: 3,
        name: 'Pradhan Mantri Jeevan Jyoti Bima Yojana',
      }),
    ).toBeInTheDocument();

    const list = screen.getByRole('list');
    const items = screen.getAllByRole('listitem');
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveTextContent('Eligibility: members of self-help groups');
    expect(items[1]).toHaveTextContent('Benefit: life cover of ₹2 lakh');
    expect(screen.getByText('Benefit:', { selector: 'strong' })).toBeInTheDocument();
    expect(list.textContent).not.toContain('**');
    expect(list.textContent).not.toContain('- Eligibility');
  });

  it('converts numbered lists to bullet points in the original order', () => {
    const { container } = renderPanel({
      answer: [
        '1. First detail',
        '2. Second detail',
        '3. Third detail',
      ].join('\n'),
    });

    const items = screen.getAllByRole('listitem');
    expect(items.map((item) => item.textContent)).toEqual([
      'First detail',
      'Second detail',
      'Third detail',
    ]);
    expect(container.querySelectorAll('ol')).toHaveLength(0);
    expect(screen.queryByText('1. First detail')).not.toBeInTheDocument();
  });

  it('keeps separate short paragraphs and joins wrapped lines', () => {
    const { container } = renderPanel({
      answer: [
        'First paragraph starts here.',
        'It continues on the next line.',
        '',
        'Second paragraph.',
      ].join('\n'),
    });

    const paragraphs = [...container.querySelectorAll('.answer-text p')].map(
      (node) => node.textContent,
    );
    expect(paragraphs).toEqual([
      'First paragraph starts here. It continues on the next line.',
      'Second paragraph.',
    ]);
  });

  it('renders inline emphasis without altering the factual text', () => {
    renderPanel({
      answer: 'The benefit is **₹2 lakh** per year for eligible families.',
    });

    expect(
      screen.getByText('₹2 lakh', { selector: 'strong' }),
    ).toBeInTheDocument();
    const paragraph = screen.getByText(/The benefit is/);
    expect(paragraph).toHaveTextContent(
      'The benefit is ₹2 lakh per year for eligible families.',
    );
  });

  it('renders deeper markdown headings as sub-headings', () => {
    renderPanel({ answer: '#### Application process\nVisit the local office.' });

    expect(
      screen.getByRole('heading', { level: 4, name: 'Application process' }),
    ).toBeInTheDocument();
  });

  it('drops the model name but keeps the grounded-source indicator', () => {
    renderPanel({
      answer: 'A short answer.',
      retrieved_chunk_ids: ['a', 'b', 'c'],
    });

    expect(screen.queryByText(/provider\/model-x/)).not.toBeInTheDocument();
    expect(screen.queryByText(/model:/i)).not.toBeInTheDocument();
    expect(
      screen.getByText(/grounded in 3 retrieved passages/i),
    ).toBeInTheDocument();
  });

  it('keeps the citation cards and their source metadata below the answer', () => {
    renderPanel({
      answer: '### Scheme\nSee the source below.',
      citations: [CITATION],
      citation_labels: [CITATION.label],
      retrieved_chunk_ids: ['chunk-42', 'chunk-43'],
    });

    expect(
      screen.getByRole('heading', { name: /sources \(1\)/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByText('Indira Gandhi National Old Age Pension Scheme'),
    ).toBeInTheDocument();
    expect(screen.getByText('ignoaps.pdf')).toBeInTheDocument();
    expect(screen.getByText('7–9')).toBeInTheDocument();
    expect(
      screen.getByText('[Source: ignoaps.pdf, pp. 7-9]'),
    ).toBeInTheDocument();
  });

  it('shows the insufficient-context notice above the answer', () => {
    renderPanel({
      answer: 'The documents do not contain enough information.',
      insufficient_context: true,
    });

    expect(screen.getByRole('note')).toHaveTextContent(/may not fully cover/i);
  });

  it('falls back cleanly for empty answers', () => {
    renderPanel({ answer: '   \n  ' });

    expect(
      screen.getByText('The assistant returned an empty answer.'),
    ).toBeInTheDocument();
  });

  it('never injects HTML contained in the answer', () => {
    const malicious = '<script>window.__pwned = true</script>';
    const { container } = renderPanel({ answer: malicious });

    expect(container.querySelector('script')).toBeNull();
    expect(container.textContent).toContain(malicious);
  });
});
