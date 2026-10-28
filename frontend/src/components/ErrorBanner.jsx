/** Human-readable failure panel for API, network, and validation errors. */
export default function ErrorBanner({ error, onRetry, onDismiss }) {
  const message =
    error && typeof error.message === 'string' && error.message
      ? error.message
      : 'Something went wrong while contacting the assistant.';
  const code = error && typeof error.code === 'string' ? error.code : '';

  return (
    <div className="error-banner" role="alert">
      <div className="error-copy">
        <p className="error-title">Could not get an answer</p>
        <p className="error-message">{message}</p>
        {code ? <p className="error-code">Error code: {code}</p> : null}
      </div>
      <div className="error-actions">
        {onRetry ? (
          <button type="button" className="button button-secondary" onClick={onRetry}>
            Try again
          </button>
        ) : null}
        {onDismiss ? (
          <button type="button" className="button button-ghost" onClick={onDismiss}>
            Dismiss
          </button>
        ) : null}
      </div>
    </div>
  );
}
