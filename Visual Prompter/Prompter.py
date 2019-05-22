# author: Jan Jilecek
import json
import random
import time
from collections import defaultdict

from PyQt5.QtCore import QThread, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

import Camera
import Challenge
import EEG
import ProgressBarThread


class Prompter(QThread):
    signal_label_text = pyqtSignal('QString')
    signal_start_duration = pyqtSignal(int, int)
    signal_toggle_lights = pyqtSignal(bool)
    signal_initiate_movie = pyqtSignal('QString')
    signal_sounds_clicked = pyqtSignal(int)
    signal_eyes_open = pyqtSignal(bool)

    def __init__(self):
        QThread.__init__(self)
        self.current_challenge = None
        self.challenges = defaultdict(list)
        self.progress = ProgressBarThread.ProgressBarThread()
        self.progress.signal_progress.connect(self.change_progress)
        self.progress.signal_toggle_lights.connect(self.prompter_toggle_lights)
        self.progress.signal_start_new_stage.connect(self.start_new_stage)
        self.camera_thread = Camera.WebcamRecorder()  # Main Camera thread
        self.eeg_thread = EEG.EEG()  # Main EEG client thread
        self.eeg_thread.sample_signal.connect(self.get_samples)
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.StreamPlayback)
        self.current_samples = []
        self.current_original_duration = 0
        self.current_eyes_open = True
        self.current_movie = ""
        self.used_challenges = []

        # EEG samples collection
        self.current_log_saving = False
        self.current_log_filename = ""

    def __del__(self):
        self.wait()

    # loads challenges to a dict of lists
    def load_data(self):
        with open("data.json") as f:
            data = json.load(f)
        for key, challenges in data.items():
            for challenge in challenges:
                cal = Challenge.Challenge(key, challenge["name"], challenge["stages_duration"],
                                          challenge["stages_data"],
                                          challenge["text"])
                self.challenges[key].append(cal)
        self.eeg_thread.start()

    def pick_challenge(self, level=1):  # TODO guard against same sequential picks
        no_repeat = 5
        rand_key = random.choice(list(self.challenges))
        lst = self.challenges[rand_key]
        self.current_challenge = lst[random.randint(0, len(self.challenges[rand_key]) - 1)]
        print(self.current_challenge)

        lower_limit = 0 if len(self.used_challenges) < no_repeat else len(self.used_challenges) - no_repeat

        if level > 15:  # in case of too many recursions, reset all
            level = 1
            self.used_challenges = []

        if self.current_challenge in self.used_challenges[lower_limit:]:
            print("Collision found: " + str(self.used_challenges) + " contains " + str(self.current_challenge))
            self.pick_challenge(level=level + 1)

        self.used_challenges.append(self.current_challenge)

    def get_samples(self, index, voltages, accelerometer):
        if self.current_log_saving:  # if the saving is turned on (we are in Stage 3)
            self.current_samples.append((index, voltages, accelerometer))

    def get_another_challenge(self, stage=1, anew=False):
        if anew:
            self.pick_challenge()

        eyes = "OPEN" if self.current_eyes_open else "CLOSED"
        if stage > 3:
            stage = 1
            if self.current_log_saving:  # if we are in a new cycle and the log was being recorded
                with open(
                        "out/" + self.current_challenge.get_key() + "_" + eyes + "_" + self.current_challenge.get_name() + "_" + str(
                            int(time.time())) + ".json", "w") as f:
                    data = {}
                    for sample in self.current_samples:  # we write the current stage samples
                        data[str(sample[0])] = {}
                        data[str(sample[0])]['e'] = str(sample[1])  # eeg
                        data[str(sample[0])]['a'] = str(sample[2])  # aux

                    json_data = json.dumps(data)
                    f.write(str(json_data))
                self.current_samples = []  # and initiate the list again

            local = QUrl.fromLocalFile('bell_tibet.mp3')
            media = QMediaContent(local)
            self.mediaPlayer.setMedia(media)
            self.mediaPlayer.play()

            self.get_another_challenge(stage, True)  # we begin anew, overflow

        self.current_log_saving = False  # turn off log saving, we later turn it on in the Stage 3

        self.prompter_toggle_lights(True)  # default: color red is on - do nothing, yet
        self.signal_start_duration.emit(self.current_challenge.get_duration(stage),
                                        self.current_challenge.get_original_duration(stage))
        self.current_original_duration = self.current_challenge.get_original_duration(stage)
        self.current_movie = self.current_challenge.get_data(stage)
        self.progress.set_stage(stage)
        self.progress.set_remaining(self.current_challenge.get_duration(stage))

        if stage == 1:
            self.signal_label_text.emit(self.current_challenge.text)
            self.camera_thread.quit()  # turn of the camera if it is not already
            self.current_eyes_open = random.choice([True, False])

            # eyes always closed in meditation
            if self.current_challenge.get_key() == "meditation":
                self.current_eyes_open = False

            self.signal_eyes_open.emit(self.current_eyes_open)
        elif stage == 2:
            self.signal_eyes_open.emit(self.current_eyes_open)
            self.initiate_movie()
        else:
            # Stage 3 - we begin collecting samples AND recording the web camera

            self.current_log_saving = True
            self.camera_thread.set_details(self.current_challenge.get_duration(stage),
                                           "out/" + self.current_challenge.get_key() + "_" + eyes + "_" + self.current_challenge.get_name()
                                           + "_" + str(int(time.time())))
            self.camera_thread.start()
            self.initiate_movie()
            self.prompter_toggle_lights(False)

        self.progress.start()

    def change_sounds(self, k):
        self.mediaPlayer.setVolume(k * 50)

    def initiate_movie(self):
        self.signal_initiate_movie.emit(self.current_movie)

    def prompter_toggle_lights(self, red_first):
        self.signal_toggle_lights.emit(red_first)

    def start_new_stage(self, stage):
        self.get_another_challenge(stage)

    def change_progress(self, duration):
        self.signal_start_duration.emit(duration, self.current_original_duration)

    def run(self):
        print("Starting Challenges")
        self.load_data()
        self.get_another_challenge(1, True)
