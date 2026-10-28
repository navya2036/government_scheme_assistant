import { answerToSpeechText, useSpeechSynthesis } from '../voice/speech';
import CitationCard from './CitationCard';

const HEADING_RE = /^(#{1,6})\s+(.+?)\s*#*\s*$/;
const BOLD_HEADING_RE = /^(?:(\d+)[.)]\s+)?\*\*(.+?)\*\*\s*:?\s*$/;
const BULLET_RE = /^[-*•]\s+(.+?)\s*$/;
const NUMBERED_RE = /^(\d+)[.)]\s+(.+?)\s*$/;
const INLINE_RE = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;

/**
 * Render the small amount of inline markdown an LLM commonly emits without
 * using HTML injection. Only the supplied text becomes nodes; nothing is added.
 */
function renderInline(text) {
  return String(text)
    .split(INLINE_RE)
    .filter((part) => part !== '')
    .map((part, index) => {
      if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
        return <strong key={index}>{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
        return <code key={index}>{part.slice(1, -1)}</code>;
      }
      if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
        return <em key={index}>{part.slice(1, -1)}</em>;
      }
      return part;
    });
}

function headingText(line) {
  const markdown = line.match(HEADING_RE);
  if (markdown) {
    return { level: markdown[1].length, text: markdown[2] };
  }
  const bold = line.match(BOLD_HEADING_RE);
  if (bold) {
    return { level: 3, text: bold[2] };
  }
  return null;
}

/**
 * Parse the answer into headings, paragraphs, and lists.
 *
 * The backend's factual text is never rewritten: markdown markers are removed,
 * consecutive plain lines are joined into one paragraph, and list markers are
 * stripped while item order and wording stay exactly as returned.
 */
function parseAnswer(answer) {
  const blocks = [];
  let paragraph = [];
  let listItems = null;

  function flushParagraph() {
    if (paragraph.length > 0) {
      blocks.push({ type: 'paragraph', text: paragraph.join(' ').trim() });
      paragraph = [];
    }
  }

  function flushList() {
    if (listItems && listItems.length > 0) {
      blocks.push({ type: 'list', items: listItems });
    }
    listItems = null;
  }

  const lines = String(answer ?? '')
    .replace(/\r\n?/g, '\n')
    .split('\n');

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = headingText(line);
    if (heading) {
      flushParagraph();
      flushList();
      blocks.push({ type: 'heading', ...heading });
      continue;
    }

    const bullet = line.match(BULLET_RE);
    if (bullet) {
      flushParagraph();
      if (!listItems) {
        listItems = [];
      }
      listItems.push(bullet[1]);
      continue;
    }

    const numbered = line.match(NUMBERED_RE);
    if (numbered) {
      flushParagraph();
      if (!listItems) {
        listItems = [];
      }
      listItems.push(numbered[2]);
      continue;
    }

    flushList();
    paragraph.push(line);
  }

  flushParagraph();
  flushList();
  return blocks;
}

/** Group blocks under their headings so each scheme reads as one section. */
function buildSections(blocks) {
  const sections = [];
  let current = { heading: null, blocks: [] };

  for (const block of blocks) {
    if (block.type === 'heading') {
      if (current.heading || current.blocks.length > 0) {
        sections.push(current);
      }
      current = { heading: block, blocks: [] };
    } else {
      current.blocks.push(block);
    }
  }

  if (current.heading || current.blocks.length > 0) {
    sections.push(current);
  }

  return sections;
}

function AnswerBody({ answer }) {
  const sections = buildSections(parseAnswer(answer));

  if (sections.length === 0) {
    return <p>The assistant returned an empty answer.</p>;
  }

  return sections.map((section, sectionIndex) => (
    <section className="answer-section" key={`section-${sectionIndex}`}>
      {section.heading ? (
        section.heading.level <= 3 ? (
          <h3 className="answer-heading">{renderInline(section.heading.text)}</h3>
        ) : (
          <h4 className="answer-heading">{renderInline(section.heading.text)}</h4>
        )
      ) : null}
      {section.blocks.map((block, blockIndex) => {
        const key = `block-${sectionIndex}-${blockIndex}`;
        if (block.type === 'list') {
          return (
            <ul className="answer-list" key={key}>
              {block.items.map((item, itemIndex) => (
                <li key={`${key}-${itemIndex}`}>{renderInline(item)}</li>
              ))}
            </ul>
          );
        }
        return <p key={key}>{renderInline(block.text)}</p>;
      })}
    </section>
  ));
}

/** Rendered answer plus the source cards supplied by the backend. */
export default function AnswerPanel({ result }) {
  const citations = Array.isArray(result.citations) ? result.citations : [];
  const retrievedCount = Array.isArray(result.retrieved_chunk_ids)
    ? result.retrieved_chunk_ids.length
    : 0;
  const speech = useSpeechSynthesis();
  const hasAnswer =
    typeof result.answer === 'string' && result.answer.trim().length > 0;

  return (
    <section className="answer-panel" aria-label="Answer">
      <div className="answer-header">
        <h2 className="section-title">Answer</h2>
        <div
          className="answer-actions"
          role="group"
          aria-label="Answer audio controls"
        >
          <button
            type="button"
            className="button button-secondary"
            onClick={() => speech.speak(answerToSpeechText(result.answer))}
            disabled={!speech.supported || !hasAnswer}
            title="Read only the generated answer aloud"
          >
            Read Answer
          </button>
          <button
            type="button"
            className="button button-secondary"
            onClick={speech.stop}
            disabled={!speech.speaking}
          >
            Stop
          </button>
        </div>
      </div>

      {speech.speaking ? (
        <p className="voice-feedback" role="status">
          Reading the answer aloud…
        </p>
      ) : null}
      {speech.error ? (
        <p className="voice-feedback is-error" role="alert">
          {speech.error}
        </p>
      ) : null}
      {!speech.supported ? (
        <p className="voice-feedback">
          Reading answers aloud is not supported in this browser.
        </p>
      ) : null}

      {result.insufficient_context ? (
        <p className="notice" role="note">
          The scheme documents may not fully cover this question. Use the answer
          below with caution and check the sources.
        </p>
      ) : null}

      <div className="answer-text" aria-live="polite">
        <AnswerBody answer={result.answer} />
      </div>

      <p className="answer-meta">
        Grounded in {retrievedCount} retrieved passage
        {retrievedCount === 1 ? '' : 's'}
      </p>

      <section className="citations" aria-labelledby="sources-heading">
        <h3 className="section-subtitle" id="sources-heading">
          Sources ({citations.length})
        </h3>
        {citations.length > 0 ? (
          <ul className="citation-list">
            {citations.map((citation, index) => (
              <CitationCard
                key={citation.chunk_id || `${citation.source_file}-${index}`}
                citation={citation}
              />
            ))}
          </ul>
        ) : (
          <p className="muted">No sources were cited for this answer.</p>
        )}
      </section>
    </section>
  );
}
