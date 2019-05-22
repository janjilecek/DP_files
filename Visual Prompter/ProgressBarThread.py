# author: Jan Jilecek
import time
from PyQt5.QtCore import QThread, pyqtSignal


class ProgressBarThread(QThread):
    signal_progress = pyqtSignal(int)
    signal_toggle_lights = pyqtSignal(bool)
    signal_initiate_movie = pyqtSignal()
    signal_start_new_stage = pyqtSignal(int)

    def __init__(self):
        QThread.__init__(self)
        self.remaining = 5
        self.stage = 1

    def set_stage(self, stage):
        self.stage = stage

    def set_remaining(self, _remaining):
        self.remaining = _remaining

    def countdown(self):
        while self.remaining >= 0:
            self.signal_progress.emit(self.remaining)
            self.remaining -= 1
            time.sleep(1)

    def run(self):
        self.countdown()
        self.signal_toggle_lights.emit(False)
        self.signal_start_new_stage.emit(self.stage + 1)  # Initiate Next Stage
