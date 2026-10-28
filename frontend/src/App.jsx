import { useCallback, useEffect, useRef, useState } from 'react';

import { askQuestion } from './api/client';
import AnswerPanel from './components/AnswerPanel';
import ErrorBanner from './components/ErrorBanner';
import QuestionForm from './components/QuestionForm';

function LoadingPanel() {
  return (
    <div className="loading-panel" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <div>
        <p className="loading-title">Searching the scheme documents…</p>
        <p className="muted">
          Retrieving passages, reranking them, and drafting a grounded answer.
          This usually takes a few seconds.
        </p>
      </div>
    </div>
  );
}

export default function App() {
  const [question, setQuestion] = useState('');
  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [validation, setValidation] = useState('');
  const abortRef = useRef(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const submit = useCallback(
    async (event) => {
      if (event && typeof event.preventDefault === 'function') {
        event.preventDefault();
      }
      if (status === 'loading') {
        return;
      }

      const trimmed = question.trim();
      if (!trimmed) {
        setValidation('Please enter a question about a government scheme.');
        setResult(null);
        setError(null);
        setStatus('idle');
        return;
      }

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setValidation('');
      setError(null);
      setResult(null);
      setStatus('loading');

      try {
        const response = await askQuestion(trimmed, { signal: controller.signal });
        if (abortRef.current !== controller) {
          return;
        }
        setResult(response);
        setStatus('success');
      } catch (requestError) {
        if (requestError && requestError.name === 'AbortError') {
          return;
        }
        if (abortRef.current !== controller) {
          return;
        }
        setError(requestError);
        setStatus('error');
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
        }
      }
    },
    [question, status],
  );

  const clear = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setQuestion('');
    setResult(null);
    setError(null);
    setValidation('');
    setStatus('idle');
  }, []);

  const dismissError = useCallback(() => {
    setError(null);
    setStatus('idle');
  }, []);

  function handleQuestionChange(value) {
    setQuestion(value);
    if (validation) {
      setValidation('');
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <p className="eyebrow">Official scheme documents · grounded answers</p>
        <h1>Government Scheme Assistant</h1>
        <p className="subtitle">
          Ask about pensions, scholarships, subsidies, and other welfare schemes.
          Every answer is drawn from the indexed scheme PDFs and shows its
          sources.
        </p>
      </header>

      <main className="app-main">
        <QuestionForm
          value={question}
          onChange={handleQuestionChange}
          onSubmit={submit}
          onClear={clear}
          loading={status === 'loading'}
        />

        {validation ? (
          <p className="validation-message" role="alert">
            {validation}
          </p>
        ) : null}

        {status === 'loading' ? <LoadingPanel /> : null}

        {status === 'error' && error ? (
          <ErrorBanner
            error={error}
            onRetry={() => submit()}
            onDismiss={dismissError}
          />
        ) : null}

        {status === 'success' && result ? <AnswerPanel result={result} /> : null}
      </main>

      <footer className="app-footer">
        <p>
          Answers are generated from scheme documents and may not cover every
          case. Verify details with the official scheme guidelines before
          applying.
        </p>
      </footer>
    </div>
  );
}
