import pyttsx3
import threading
import pythoncom
import time

def f():
    pythoncom.CoInitialize()
    try:
        e = pyttsx3.init()
        e.say('hello')
        e.runAndWait()
        print('success')
    except Exception as ex:
        print('error', ex)

t = threading.Thread(target=f)
t.start()
t.join()
