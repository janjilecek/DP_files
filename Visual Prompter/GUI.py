# author: Jan Jilecek

from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QFrame
from PyQt5.QtCore import QT_VERSION_STR

import Prompter


# TODO material theme (supported since Qt 5.7?) https://doc.qt.io/qt-5.11/qtquickcontrols2-material.html
class MainGUI(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        print("Qt version:", QT_VERSION_STR)
        self.title = "Visual Prompter v1.0"

        # init
        super(MainGUI, self).__init__(parent, QtCore.Qt.Window)
        with open('VisualPrompter_v0.02.ui') as f:
            uic.loadUi(f, self)

        self.setWindowTitle(self.title)

        self.progressBar.setValue(0)
        self.redFrame.setStyleSheet("QFrame { background-color: red; margin: 20px; }")
        self.greenFrame.setStyleSheet("QFrame { background-color: green; margin: 20px; }")
        self.greenFrame.setVisible(False)
        self.redFrame.setVisible(False)
        self.label.setStyleSheet("font: 36pt;")  # controls size of text in the main frame

        # signals and threads
        self.startButton.clicked.connect(self.start_button_clicked)
        self.sound.clicked.connect(self.sounds_toggle)
        self.stopButton.clicked.connect(self.stop_button_clicked)
        self.prompter_thread = Prompter.Prompter()  # Main Prompter thread
        self.prompter_thread.signal_label_text.connect(self.change_label_text)
        self.prompter_thread.signal_start_duration.connect(self.change_start_duration)
        self.prompter_thread.signal_toggle_lights.connect(self.toggle_lights)
        self.prompter_thread.signal_initiate_movie.connect(self.initiate_movie)
        self.prompter_thread.signal_eyes_open.connect(self.eyes_open)

    def initiate_movie(self, path):
        animation = QMovie(path)
        self.label.setScaledContents(
            True)  # TODO set to true / set fixed window height / create unified gif dimensions (square)
        self.label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.label.setMovie(animation)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        animation.start()

    def toggle_lights(self, red_first=True):
        self.redFrame.setVisible(red_first)
        self.greenFrame.setVisible(not red_first)

    def change_start_duration(self, duration, original_duration):
        self.seconds.display(duration)
        try:
            ratio = (100 / original_duration) * (original_duration - duration)
            self.progressBar.setValue(ratio)
        except ZeroDivisionError:
            self.progressBar.setValue(0)

    def change_label_text(self, text):
        self.label.setText(text)

    def start_button_clicked(self):
        self.prompter_thread.start()

    def eyes_open(self, opened):
        if opened:
            self.redFrame_2.setStyleSheet("background-image: url(./challenges/open.png); background-repeat: no-repeat;")
        else:
            self.redFrame_2.setStyleSheet("background-image: url(./challenges/closed.png); background-repeat: "
                                          "no-repeat;")

    def sounds_toggle(self):
        self.prompter_thread.change_sounds(self.sound.checkState())

    def stop_button_clicked(self):
        print("Prompter Quitting..")
        # TODO safe thread quit


def exec():
    app = QtWidgets.QApplication([])
    window = MainGUI()
    window.show()
    window.initiate_movie("./begin.png")
    app.exec_()
