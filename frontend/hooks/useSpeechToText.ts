import { useEffect, useRef, useState } from 'react';

type SpeechResultHandler = (text: string) => void;

export const useSpeechToText = (onFinalText: SpeechResultHandler) => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [hasSupport, setHasSupport] = useState(false);
  const recognitionRef = useRef<any>(null);
  const onFinalTextRef = useRef(onFinalText);
  const committedTextRef = useRef('');
  const lastEmittedTextRef = useRef('');
  const latestTranscriptRef = useRef('');

  useEffect(() => {
    onFinalTextRef.current = onFinalText;
  }, [onFinalText]);

  useEffect(() => {
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setHasSupport(false);
      return;
    }
    setHasSupport(true);

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = 'en-US';
    recognition.interimResults = true;

    recognition.onresult = (event: any) => {
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const piece = result?.[0]?.transcript ?? '';
        if (!piece) continue;
        if (result.isFinal) {
          committedTextRef.current = `${committedTextRef.current} ${piece}`.trim();
        } else {
          interim = `${interim} ${piece}`.trim();
        }
      }
      const full = `${committedTextRef.current} ${interim}`.trim();
      setTranscript(full);
      latestTranscriptRef.current = full;

      // If browser flags final here, send immediately.
      if (event.results[event.resultIndex]?.isFinal) {
        const finalText = committedTextRef.current.trim();
        if (finalText && finalText !== lastEmittedTextRef.current) {
          lastEmittedTextRef.current = finalText;
          setIsListening(false);
          onFinalTextRef.current(finalText);
        }
      }
    };

    recognition.onerror = () => {
      setIsListening(false);
    };
    recognition.onend = () => {
      // Some browsers end on silence without firing a useful final result.
      // Ensure we still auto-submit when speaking stops.
      const finalText = committedTextRef.current.trim() || latestTranscriptRef.current.trim();
      if (finalText && finalText !== lastEmittedTextRef.current) {
        lastEmittedTextRef.current = finalText;
        onFinalTextRef.current(finalText);
      }
      setIsListening(false);
    };

    recognitionRef.current = recognition;
    return () => {
      try {
        recognition.onresult = null;
        recognition.onerror = null;
        recognition.onend = null;
        recognition.stop();
      } catch {
        // ignore
      }
    };
  }, []);

  const startListening = () => {
    if (!recognitionRef.current || isListening) return;
    try {
      committedTextRef.current = '';
      lastEmittedTextRef.current = '';
      latestTranscriptRef.current = '';
      setTranscript('');
      recognitionRef.current.start();
      setIsListening(true);
    } catch {
      // ignore duplicate starts
    }
  };

  return {
    isListening,
    transcript,
    startListening,
    hasSupport,
  };
};
