function formatPages(start, end) {
  const first = Number.isFinite(start) ? start : null;
  const last = Number.isFinite(end) ? end : null;
  if (first === null && last === null) {
    return 'Not listed';
  }
  if (first !== null && last !== null && first !== last) {
    return `${first}–${last}`;
  }
  return String(first ?? last);
}

/** One source card built only from metadata returned by the API. */
export default function CitationCard({ citation }) {
  const scheme = citation.scheme_name || 'Unknown scheme';
  const source = citation.source_file || 'Unknown document';

  return (
    <li className="citation-card">
      <p className="citation-scheme">{scheme}</p>
      <dl className="citation-meta">
        <div>
          <dt>Source PDF</dt>
          <dd>{source}</dd>
        </div>
        <div>
          <dt>Pages</dt>
          <dd>{formatPages(citation.page_start, citation.page_end)}</dd>
        </div>
      </dl>
      {citation.label ? <p className="citation-label">{citation.label}</p> : null}
    </li>
  );
}
