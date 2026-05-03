import pyttsx3
import threading
import sys

def speak_text_thread(text):
    """Speak text in a fresh thread to avoid pyttsx3 hanging on Windows."""
    if sys.platform == 'win32':
        try:
            import pythoncom  # type: ignore
            pythoncom.CoInitialize()
        except Exception as e:
            print(f"[Speech] pythoncom error: {e}")

    try:
        engine = pyttsx3.init()
        engine.setProperty('volume', 1.0)
        engine.setProperty('rate', 150)
        print(f"[Speech] Speaking: {text}")
        engine.say(text)
        engine.runAndWait()
        print(f"[Speech] Finished: {text}")
    except Exception as e:
        print(f"[Speech] Error: {e}")


def speak_text(text):
    """Non-blocking speech call."""
    threading.Thread(target=speak_text_thread, args=(text,), daemon=True).start()
