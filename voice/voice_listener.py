"""
voice_listener.py — Offline Voice Listener
=============================================
Captures audio from the microphone and performs offline speech recognition
using Vosk. Runs in a background thread so it doesn't block the main loop.

Requirements:
    pip install vosk sounddevice
    Download a Vosk model (e.g. vosk-model-small-en-us-0.15) and place it
    in a folder called 'vosk-model' in the project root, or set VOSK_MODEL_PATH.

If Vosk is unavailable, falls back to a text-input stub for testing.
"""

import os
import sys
import json
import queue
import threading


# ---------- configuration ----------

VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "vosk-model")
SAMPLE_RATE = 16000  # 16 kHz for speech recognition


class VoiceListener:
    """
    Listens for voice commands using offline ASR (Vosk).
    Runs a background thread that feeds recognized text into a queue.
    """

    def __init__(self, model_path=None):
        """
        Initialize the voice listener.
        Args:
            model_path: path to Vosk model directory (optional)
        """
        self.model_path = model_path or VOSK_MODEL_PATH
        self.text_queue = queue.Queue()  # recognized text appears here
        self._running = False
        self._thread = None
        self._vosk_available = False

        # Try to import vosk
        try:
            import vosk
            import sounddevice as sd
            self._vosk = vosk
            self._sd = sd
            self._vosk_available = True
        except ImportError:
            print("[VoiceListener] Vosk/sounddevice not installed.")
            print("  Install with: pip install vosk sounddevice")
            print("  Falling back to text-input mode.\n")

    def start(self):
        """Start listening in a background thread."""
        if self._running:
            return

        self._running = True

        if self._vosk_available and os.path.isdir(self.model_path):
            self._thread = threading.Thread(
                target=self._listen_vosk, daemon=True
            )
        else:
            if self._vosk_available:
                print(f"[VoiceListener] Model not found at '{self.model_path}'")
                print("  Download from: https://alphacephei.com/vosk/models")
            print("[VoiceListener] Using text-input fallback mode.")
            self._thread = threading.Thread(
                target=self._listen_text_fallback, daemon=True
            )

        self._thread.start()
        print("[VoiceListener] 🎙️  Listening started.")

    def stop(self):
        """Stop the listener thread."""
        self._running = False
        print("[VoiceListener] Listening stopped.")

    def get_command(self):
        """
        Get the next recognized voice command (non-blocking).
        Returns:
            recognized text string, or None if queue is empty
        """
        try:
            return self.text_queue.get_nowait()
        except queue.Empty:
            return None

    # ---------- Vosk listening loop ----------

    def _listen_vosk(self):
        """Background thread: capture audio and run Vosk recognition."""
        model = self._vosk.Model(self.model_path)
        recognizer = self._vosk.KaldiRecognizer(model, SAMPLE_RATE)

        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"[VoiceListener] Audio status: {status}",
                      file=sys.stderr)
            audio_queue.put(bytes(indata))

        try:
            with self._sd.RawInputStream(
                samplerate=SAMPLE_RATE, blocksize=4000,
                dtype='int16', channels=1,
                callback=audio_callback
            ):
                while self._running:
                    data = audio_queue.get()
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").strip()
                        if text:
                            print(f"[VoiceListener] Heard: \"{text}\"")
                            self.text_queue.put(text)
        except Exception as e:
            print(f"[VoiceListener] Audio error: {e}")
            print("  Switching to text-input fallback.")
            self._listen_text_fallback()

    # ---------- text-input fallback ----------

    def _listen_text_fallback(self):
        """
        Fallback: read text commands from stdin.
        Useful for testing without a microphone or Vosk model.
        """
        while self._running:
            try:
                text = input("[Voice Fallback] Type a command > ").strip()
                if text:
                    self.text_queue.put(text)
            except (EOFError, KeyboardInterrupt):
                break

    @property
    def is_running(self):
        """Return True if the listener is active."""
        return self._running
