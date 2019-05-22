# author: Jan Jilecek
from PyQt5.QtCore import QThread, pyqtSignal
from pylsl import StreamInlet, resolve_stream


class EEG(QThread):
    sample_signal = pyqtSignal(float, list, list)

    def __init__(self):
        QThread.__init__(self)
        self.eeg = None
        self.aux = None

    def run(self):
        print("looking for an EEG stream...")
        eeg_stream = resolve_stream('type', 'EEG')
        print("looking for an AUX (accelerometer) stream...")
        accelero_stream = resolve_stream('type', 'AUX')
        # read data from eeg stream
        self.eeg = StreamInlet(eeg_stream[0])
        # and one for aux data
        self.aux = StreamInlet(accelero_stream[0])
        while True:
            self.get_sample()

    def get_sample(self):
        eeg_sample, eeg_time = self.eeg.pull_sample()
        aux_sample, aux_time = self.aux.pull_sample()
        # print(eeg_time, eeg_sample, aux_sample)
        self.sample_signal.emit(eeg_time, eeg_sample, aux_sample)
