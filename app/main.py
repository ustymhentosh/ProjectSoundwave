import tkinter as tk
from ttkbootstrap import Style
from rec_pg import RecPage
from laod_pg import LoadPage
import os

if os.name == "nt":
    from ctypes import windll

    windll.shcore.SetProcessDpiAwareness(1)


class SoundRecorderApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Speaker Classifier")
        self.geometry("500x500")
        self.style = Style(theme="cyborg")
        self._frame = None
        self.switch_frame("LoadPage")

    def switch_frame(self, frame_class, arguments=None):
        """Destroys current frame and replaces it with a new one."""
        if frame_class == "RecPage":
            frame_class = RecPage
        elif frame_class == "LoadPage":
            frame_class = LoadPage

        new_frame = frame_class(self, arguments)

        if self._frame is not None:
            for child in self._frame.winfo_children():
                child.destroy()
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()


def main():
    app = SoundRecorderApp()
    app.mainloop()


if __name__ == "__main__":
    main()
