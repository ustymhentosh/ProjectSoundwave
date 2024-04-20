import tkinter as tk
from ttkbootstrap import ttk
from tkinter import filedialog as fd
from back import UsLeVoModel
import threading
import webbrowser


class LoadPage(ttk.Frame):
    def __init__(self, master, arguments=None):
        super().__init__(master)
        root = ttk.Frame(master=self)
        self.file_names = None
        self.ins_tr = None
        self.m = None

        def callback_1():
            file_names = fd.askopenfilenames()
            self.file_names = file_names
            num_counter["text"] = f"Loaded .mp3 files: {len(self.file_names)}"
            if bool(self.ins_tr) and len(self.file_names):
                train_button["state"] = tk.NORMAL

        def callback_2():
            file_name = fd.askopenfilename()
            self.ins_tr = file_name
            csv_counter["text"] = f"Loaded .csv file: {bool(self.ins_tr)}"
            if bool(self.ins_tr) and len(self.file_names):
                train_button["state"] = tk.NORMAL
            root.update()

        def start_taining():
            recording_thread = threading.Thread(target=train_model, args=())
            recording_thread.start()

        def callback_3():
            if bool(self.ins_tr) and len(self.file_names):
                wail_lbl.pack()
                start_taining()

        def show_start_button(nothing):
            wail_lbl.pack_forget()
            start_btn.pack(pady=(10, 0))

        def train_model():
            self.m = UsLeVoModel()
            self.m.train(self.file_names, self.ins_tr)
            root.event_generate("<<TrainingFinished>>", when="tail")

        co_label = ttk.Label(
            root, text="Load .mp3 recordings and .csv file", font="Calibri 14 bold"
        )
        co_label.pack()

        instruction = ttk.Button(
            root,
            style="link",
            text="Instruction",
            command=lambda: webbrowser.open(
                "https://share.note.sx/1d549wx0#4v5qSatVivfwzhRkjw4acNj5HQDkOqv4MHEOcm3kpgA"
            ),
        )
        instruction.pack()

        ttk.Style().configure("primary.TButton", font=("Calibri", 12, "bold"))
        ttk.Style().configure("secondary.TButton", font=("Calibri", 12, "bold"))
        ttk.Style().configure("success.TButton", font=("Calibri", 12, "bold"))
        load_mp3 = ttk.Button(
            root, text="Load .mp3s", style="primary.TButton", command=callback_1
        )
        load_mp3.pack(padx=10, pady=10)

        num_counter = ttk.Label(master=root, text="Loaded .mp3 files: 0")
        num_counter.pack(padx=10, pady=(0, 10))

        load_csv = ttk.Button(
            root, text="Load .csv", style="primary.TButton", command=callback_2
        )
        load_csv.pack(padx=10, pady=10)

        csv_counter = ttk.Label(master=root, text="Loaded .csv file: False")
        csv_counter.pack(padx=10, pady=(0, 10))

        train_button = ttk.Button(
            root, text="Train", style="secondary.TButton", command=callback_3
        )
        train_button.pack(padx=10, pady=10)
        train_button["state"] = tk.DISABLED

        wail_lbl = ttk.Label(
            master=root, text="🕖 please wait 🕖", font="Calibri 12 bold"
        )

        start_btn = ttk.Button(
            root,
            text="Start Recognition",
            style="success.TButton",
            command=lambda: master.switch_frame("RecPage", arguments=self.m),
        )

        root.bind("<<TrainingFinished>>", show_start_button)
        root.pack()
