import pyttsx3
import threading
import pythoncom
import time
import queue

speech_queue = queue.Queue()

def speech_worker():
    import pythoncom
    pythoncom.CoInitialize()
    try:
        engine = pyttsx3.init()
        engine.setProperty('volume',1.0)
        engine.setProperty('rate', 150)
        while True:
            text = speech_queue.get()
            if text is None:
              break
            print(f"Speaking: {text}")
            engine.say(text)
            engine.runAndWait()
            print(f"Done speaking: {text}")
            speech_queue.task_done()
    except Exception as e:
        print(f"Speech initialization error: {e}")

t = threading.Thread(target=speech_worker, daemon=True)
t.start()

speech_queue.put("A")
time.sleep(2)
speech_queue.put("B")
time.sleep(2)
speech_queue.put(None)
t.join()
