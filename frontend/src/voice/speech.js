import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Browser-only speech helpers.
 *
 * Speech-to-text uses the Web Speech `SpeechRecognition` API and text-to-speech
 * uses `window.speechSynthesis`. Nothing here calls a remote voice service, and
 * every API is feature-detected so unsupported browsers keep working normally.
 */

const RECOGNITION_ERROR_MESSAGES = {
  'not-allowed':
    'Microphone access was denied. Allow microphone access in your browser and try again.',
  'service-not-allowed':
    'Microphone access was denied. Allow microphone access in your browser and try again.',
  'audio-capture':
    'No microphone was found. Check your microphone or type your question instead.',
  'no-speech': 'No speech was detected. Try again and speak clearly.',
  network:
    'Speech recognition could not connect. Check your connection or type your question instead.',
  'language-not-supported':
    'Voice input does not support this language. You can type your question instead.',
};

const RECOGNITION_FALLBACK_ERROR =
  'Voice input failed. Please try again or type your question.';

const SYNTHESIS_ERROR =
  'The answer could not be read aloud. Please try again.';

export function getSpeechRecognitionConstructor() {
  if (typeof window === 'undefined') {
    return null;
  }
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

export function speechRecognitionSupported() {
  return Boolean(getSpeechRecognitionConstructor());
}

export function speechSynthesisSupported() {
  return (
    typeof window !== 'undefined' &&
    Boolean(window.speechSynthesis) &&
    typeof window.SpeechSynthesisUtterance === 'function'
  );
}

export function recognitionErrorMessage(code) {
  return RECOGNITION_ERROR_MESSAGES[code] || RECOGNITION_FALLBACK_ERROR;
}

function speechLanguage() {
  if (typeof navigator !== 'undefined' && navigator.language) {
    return navigator.language;
  }
  return 'en-IN';
}

/**
 * Turn the generated answer into plain prose for the speech synthesiser.
 * Markdown markers are formatting only, so they are removed while every word
 * of the answer is kept. Citations and source metadata are never passed here.
 */
export function answerToSpeechText(answer) {
  return String(answer ?? '')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/^\s*[-*•]\s+/gm, '')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Speech-to-text state machine.
 *
 * Only final results are handed to `onFinalTranscript`; interim results are
 * exposed for the "Listening..." indicator and never inserted into the input.
 */
export function useSpeechRecognition({ onFinalTranscript } = {}) {
  const supported = speechRecognitionSupported();
  const [listening, setListening] = useState(false);
  const [interimTranscript, setInterimTranscript] = useState('');
  const [error, setError] = useState('');
  const recognitionRef = useRef(null);
  const onFinalRef = useRef(onFinalTranscript);

  useEffect(() => {
    onFinalRef.current = onFinalTranscript;
  }, [onFinalTranscript]);

  const stop = useCallback(() => {
    const recognition = recognitionRef.current;
    recognitionRef.current = null;
    setListening(false);
    setInterimTranscript('');
    if (recognition) {
      try {
        recognition.stop();
      } catch {
        // The browser may already have stopped it.
      }
    }
  }, []);

  useEffect(
    () => () => {
      const recognition = recognitionRef.current;
      recognitionRef.current = null;
      if (recognition) {
        try {
          recognition.abort();
        } catch {
          // Nothing else to clean up if the browser already ended it.
        }
      }
    },
    [],
  );

  const start = useCallback(() => {
    if (!supported || recognitionRef.current) {
      return;
    }

    const Recognition = getSpeechRecognitionConstructor();
    if (!Recognition) {
      setError(RECOGNITION_FALLBACK_ERROR);
      return;
    }

    let recognition;
    try {
      recognition = new Recognition();
    } catch {
      setError(RECOGNITION_FALLBACK_ERROR);
      return;
    }

    recognition.lang = speechLanguage();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setError('');
      setInterimTranscript('');
      setListening(true);
    };

    recognition.onresult = (event) => {
      const results = event?.results;
      if (!results) {
        return;
      }
      const startIndex = Number.isInteger(event.resultIndex)
        ? event.resultIndex
        : 0;
      let interim = '';

      for (let index = startIndex; index < results.length; index += 1) {
        const result = results[index];
        const transcript = result?.[0]?.transcript ?? '';
        if (result?.isFinal) {
          const finalText = transcript.trim();
          if (finalText) {
            onFinalRef.current?.(finalText);
          }
        } else {
          interim += transcript;
        }
      }

      setInterimTranscript(interim.trim());
    };

    recognition.onerror = (event) => {
      const code = event?.error ?? 'unknown';
      if (code === 'aborted') {
        return;
      }
      setError(recognitionErrorMessage(code));
      setListening(false);
      setInterimTranscript('');
    };

    recognition.onend = () => {
      if (recognitionRef.current === recognition) {
        recognitionRef.current = null;
      }
      setListening(false);
      setInterimTranscript('');
    };

    recognitionRef.current = recognition;
    setError('');
    setInterimTranscript('');
    setListening(true);

    try {
      recognition.start();
    } catch {
      recognitionRef.current = null;
      setListening(false);
      setError(RECOGNITION_FALLBACK_ERROR);
    }
  }, [supported]);

  const toggle = useCallback(() => {
    if (recognitionRef.current) {
      stop();
    } else {
      start();
    }
  }, [start, stop]);

  return {
    supported,
    listening,
    interimTranscript,
    error,
    start,
    stop,
    toggle,
  };
}

/**
 * Text-to-speech state for one answer.
 *
 * `speak` always cancels any current utterance before queueing a new one.
 * Playback never starts automatically.
 */
export function useSpeechSynthesis() {
  const supported = speechSynthesisSupported();
  const [speaking, setSpeaking] = useState(false);
  const [error, setError] = useState('');
  const utteranceRef = useRef(null);

  const stop = useCallback(() => {
    utteranceRef.current = null;
    if (
      typeof window !== 'undefined' &&
      window.speechSynthesis &&
      typeof window.speechSynthesis.cancel === 'function'
    ) {
      window.speechSynthesis.cancel();
    }
    setSpeaking(false);
  }, []);

  useEffect(
    () => () => {
      utteranceRef.current = null;
      if (
        typeof window !== 'undefined' &&
        window.speechSynthesis &&
        typeof window.speechSynthesis.cancel === 'function'
      ) {
        window.speechSynthesis.cancel();
      }
    },
    [],
  );

  const speak = useCallback(
    (text) => {
      const spokenText = String(text ?? '').trim();
      if (!supported || !spokenText) {
        return false;
      }

      const synth = window.speechSynthesis;

      // Cancel any existing speech before a new utterance is created.
      utteranceRef.current = null;
      if (typeof synth.cancel === 'function') {
        synth.cancel();
      }

      let utterance;
      try {
        utterance = new window.SpeechSynthesisUtterance(spokenText);
      } catch {
        setSpeaking(false);
        setError(SYNTHESIS_ERROR);
        return false;
      }

      utterance.lang = speechLanguage();
      utterance.onend = () => {
        if (utteranceRef.current === utterance) {
          utteranceRef.current = null;
          setSpeaking(false);
        }
      };
      utterance.onerror = (event) => {
        if (utteranceRef.current !== utterance) {
          return;
        }
        utteranceRef.current = null;
        setSpeaking(false);
        if (event?.error && event.error !== 'canceled' && event.error !== 'interrupted') {
          setError(SYNTHESIS_ERROR);
        }
      };

      utteranceRef.current = utterance;
      setError('');
      setSpeaking(true);
      synth.speak(utterance);
      return true;
    },
    [supported],
  );

  return { supported, speaking, error, speak, stop };
}
