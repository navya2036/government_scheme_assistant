import { useCallback, useEffect, useId, useRef } from 'react';

import { useSpeechRecognition } from '../voice/speech';

const MAX_LENGTH = 1000;

function MicrophoneIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      aria-hidden="true"
      focusable="false"
    >
      <path
        d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Z"
        fill="currentColor"
      />
      <path
        d="M5 11a7 7 0 0 0 14 0M12 18v3m-3 0h6"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

/**
 * Question input. Enter asks, Shift+Enter adds a newline, Ctrl/Cmd+Enter also
 * asks. The Ask button is disabled while a request is in flight.
 *
 * A microphone button adds browser speech-to-text as an optional input method.
 * Only final transcripts are appended to the question; nothing is auto-submitted.
 */
export default function QuestionForm({
  value,
  onChange,
  onSubmit,
  onClear,
  loading,
}) {
  const inputId = useId();
  const hintId = useId();
  const remaining = MAX_LENGTH - value.length;
  const questionRef = useRef(value);

  useEffect(() => {
    questionRef.current = value;
  }, [value]);

  const appendTranscript = useCallback(
    (transcript) => {
      const current = questionRef.current.trim();
      const next = current ? `${current} ${transcript}` : transcript;
      const limited = next.slice(0, MAX_LENGTH);
      questionRef.current = limited;
      onChange(limited);
    },
    [onChange],
  );

  const {
    supported: voiceSupported,
    listening,
    interimTranscript,
    error: voiceError,
    toggle: toggleListening,
    stop: stopListening,
  } = useSpeechRecognition({ onFinalTranscript: appendTranscript });

  function handleInputChange(event) {
    questionRef.current = event.target.value;
    onChange(event.target.value);
  }

  function handleSubmit(event) {
    stopListening();
    onSubmit(event);
  }

  function handleKeyDown(event) {
    if (event.key !== 'Enter' || event.nativeEvent.isComposing) {
      return;
    }
    if (event.shiftKey) {
      return;
    }
    event.preventDefault();
    if (!loading) {
      handleSubmit(event);
    }
  }

  function handleClear() {
    stopListening();
    onClear();
  }

  const microphoneLabel = voiceSupported
    ? listening
      ? 'Microphone: stop voice input'
      : 'Microphone: start voice input'
    : 'Microphone: voice input is not supported in this browser';

  return (
    <form className="question-form" onSubmit={handleSubmit} noValidate>
      <label className="field-label" htmlFor={inputId}>
        Your question
      </label>
      <textarea
        id={inputId}
        className="question-input"
        name="question"
        rows={3}
        value={value}
        maxLength={MAX_LENGTH}
        placeholder="e.g. What financial assistance is available for old age pension?"
        aria-describedby={hintId}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        autoFocus
      />
      <div className="form-footer">
        <p className="hint" id={hintId}>
          <kbd>Enter</kbd> to ask · <kbd>Shift</kbd> + <kbd>Enter</kbd> for a new
          line · {remaining} characters left
        </p>
        <div className="form-actions">
          <button
            type="button"
            className={`button button-secondary voice-button${listening ? ' is-listening' : ''}`}
            onClick={toggleListening}
            disabled={!voiceSupported}
            aria-label={microphoneLabel}
            aria-pressed={listening}
            title={
              voiceSupported
                ? listening
                  ? 'Stop listening'
                  : 'Ask your question by speaking'
                : 'Voice input is not supported in this browser'
            }
          >
            <MicrophoneIcon />
            <span>
              {voiceSupported ? (listening ? 'Listening…' : 'Speak') : 'Voice'}
            </span>
          </button>
          <button
            type="button"
            className="button button-secondary"
            onClick={handleClear}
          >
            Clear
          </button>
          <button
            type="submit"
            className="button button-primary"
            disabled={loading}
            aria-busy={loading}
          >
            {loading ? 'Asking…' : 'Ask'}
          </button>
        </div>
      </div>

      <div className="voice-feedback">
        {listening ? (
          <p className="voice-status" role="status">
            <span className="listening-dot" aria-hidden="true" />
            Listening…
            {interimTranscript ? ` ${interimTranscript}` : ''}
          </p>
        ) : null}
        {voiceError ? (
          <p className="voice-error" role="alert">
            {voiceError}
          </p>
        ) : null}
        {!voiceSupported ? (
          <p className="voice-note">
            Voice input is not supported in this browser. You can still type
            your question.
          </p>
        ) : null}
      </div>
    </form>
  );
}
