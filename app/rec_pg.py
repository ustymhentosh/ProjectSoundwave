import time
import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import soundfile as sf
import threading
from tkinter import filedialog as fd
import subprocess
import pygame
import time
class RecPage(ttk.Frame):
    def __init__(self, master, arguments):
        super().__init__(master)
        root = ttk.Frame(master=self)
        self.model = arguments

        def check_file():
            file_name = fd.askopenfilename()
            countdown_label.config(text="and the answer is..")
            root.update()
            answ["text"] = self.model.test(file_name)

        def record():
            record_button.config(state=tk.DISABLED)

            duration = 5
            fs = 44100
            countdown_interval = 1

            recording_thread = threading.Thread(
                target=start_recording, args=(duration, fs)
            )
            recording_thread.start()

            for i in range(duration, 0, -countdown_interval):
                countdown_label.config(text=f"🎙Recording {i} seconds left")
                master.update()
                master.after(1000)

            countdown_label.config(text="and the answer is..")

        def start_recording(duration, fs):
            # recording = sd.rec(int(duration * fs), samplerate=fs)
            # sd.wait()  # Wait for recording to finish
            #
            # filename = "recorded_sound.wav"
            # sf.write(filename, recording, fs)
            # print("Recording saved as:", filename)
            answ["text"] = ""
            root.update()

            java_command = ["java", "AudioRecorder"]
            subprocess.run(java_command, check=True)

            # Notify the main GUI thread that recording is finished
            master.event_generate("<<RecordingFinished>>", when="tail")

        def enable_record_button(event):
            record_button.config(state=tk.NORMAL)
            answ["text"] = "🥁🥁🥁"
            root.update()
            answer = self.model.test("recorded_sound.wav")
            pygame.mixer.init()
            pygame.mixer.music.load("drumroll.wav")
            pygame.mixer.music.play()
            time.sleep(2.1)
            answ["text"] = answer

        ttk.Style().configure("primary.TButton", font=("Calibri", 12, "bold"))
        ttk.Style().configure("info.TButton", font=("Calibri", 12, "bold"))
        record_button = ttk.Button(
            root, text="Record", style="primary.TButton", command=record
        )
        record_button.pack(pady=(10, 10), ipadx=10)

        use_file = ttk.Button(
            root, text="Use File", style="info.TButton", command=check_file
        )
        use_file.pack(pady=(10, 10), ipadx=8)

        countdown_label = ttk.Label(root, text="", font="Calibri 12")
        countdown_label.pack(pady=(10, 10))

        master.bind("<<RecordingFinished>>", enable_record_button)
        answ = ttk.Label(root, text="None", font="Calibri 16")
        answ.pack(pady=(10, 10))

        root.pack()
