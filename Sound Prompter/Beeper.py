# author: Jan Jilecek

from PyQt5.QtCore import QThread, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import pygame


class Beeper(QThread):
    def __init__(self):
        QThread.__init__(self)
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.StreamPlayback)
        pygame.init()
        pygame.mixer.music.load("audio/end_2.mp3")

    def set_start_sound(self):
        pygame.mixer.music.load("audio/start.mp3")

    def set_end_sound(self):
        pygame.mixer.music.load("audio/end_2.mp3")

    def set_stage_sound(self, name):
        pygame.mixer.music.load(name)

    def run(self):
        print("started beeper")

        """local = QUrl.fromLocalFile('beep-10.mp3')
        media = QMediaContent(local)
        self.mediaPlayer.setMedia(media)
        self.mediaPlayer.play()"""

        pygame.mixer.music.play()
