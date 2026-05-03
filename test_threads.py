import pyttsx3
import threading
import time
import sys

def speak_text_thread(text):
    if sys.platform == 'win32':
        import pythoncom
        pythoncom.CoInitialize()
    try:
        engine = pyttsx3.init()
        engine.setProperty('volume', 1.0)
        engine.setProperty('rate', 150)
        print(f"Speaking: {text}")
        engine.say(f"Letter {text}")
        engine.runAndWait()
        print(f"Done: {text}")
    except Exception as e:
        print(f"Error: {e}")

def speak_text(text):
    t = threading.Thread(target=speak_text_thread, args=(text,), daemon=True)
    t.start()
    return t

t1 = speak_text("A")
t1.join()
time.sleep(1)
t2 = speak_text("B")
t2.join()
time.sleep(1)
t3 = speak_text("C")
t3.join()
