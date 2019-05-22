# author: Jan Jilecek
import json, csv
import random
import time
from collections import defaultdict

import sys
from PyQt5.QtCore import QThread, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

import Beeper
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
        #self.camera_thread = Camera.WebcamRecorder()  # Main Camera thread
        self.eeg_thread = EEG.EEG()  # Main EEG client thread
        self.eeg_thread.sample_signal.connect(self.get_samples)
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.StreamPlayback)
        self.beeper_thread = Beeper.Beeper()  # beeper thread
        self.current_samples = []
        self.current_original_duration = 0
        self.current_eyes_open = True
        self.current_movie = ""
        self.used_challenges = []
        self.current_state_filename = ""
        self.session_labels_dict = dict()
        self.symbols_count = 15  # number of repetitions for each class
        self.symbols_status = defaultdict(lambda: 0)  # keep track


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
                                          challenge["text"],
                                          challenge["eyes"],
                                          challenge["index"])
                self.challenges[key].append(cal)
        self.eeg_thread.start()


    def pick_challenge(self, level=1):  # pick a challenge and  guard against same sequential picks in a row
        tryhard = False
        if tryhard:
            no_repeat = 5
            rand_key = random.choice(list(self.challenges))
            lst = self.challenges[rand_key]
            try:
                self.current_challenge = lst[random.randint(0, len(self.challenges[rand_key]) - 1)]
            except Exception as e:
                print("QUITTING. ALL CHALLENGES DONE")
                self.quit()
            print(self.current_challenge)

            lower_limit = 0 if len(self.used_challenges) < no_repeat else len(self.used_challenges) - no_repeat

            if level > 25:  # in case of too many recursions, reset all
                level = 1
                self.used_challenges = []

            if self.current_challenge in self.used_challenges[lower_limit:] or \
                    self.check_symbol_status_done(self.current_challenge.get_index()):  # or the challenge is done
                print("Collision found: " + str(self.used_challenges) + " contains " + str(self.current_challenge))

                done = True
                for j in range(1, len(self.challenges) + 1):
                    if not self.check_symbol_status_done(j):
                        done = False
                        break
                if done:
                    print("ALL CHALLENGES DONE, QUITTING")
                    sys.exit(0)

                self.pick_challenge(level=level + 1)  # pick a new one


            self.used_challenges.append(self.current_challenge)
            self.symbols_status[self.current_challenge.get_index()] += 1  # increment status info
        else:
            rand_key = random.choice(["left_hand", "right_hand"])
            lst = self.challenges[rand_key]
            self.current_challenge = lst[0]


    # check if we've already done all of the tries for a symbol
    def check_symbol_status_done(self, index):
        return self.symbols_status[index] >= self.symbols_count

    def get_samples(self, index, voltages, accelerometer):
        if self.current_log_saving:  # if the saving is turned on (we are in Stage 3)
            self.current_samples.append((index, voltages, accelerometer))

    def record_camera(self):
        self.current_log_saving = True
        self.camera_thread.set_details(120,
                                       "out/" + self.current_log_filename + "_" + str(int(time.time())))
        self.camera_thread.start()

    def save_log(self):
        with open(self.current_log_filename + "_dataset.csv", "a", newline='') as f:
            dataset_writer = csv.writer(f, delimiter=",")

            for item in self.current_samples:
                if (len(item[1]) == 8):
                    dataset_writer.writerow([item[0], item[1][0], item[1][1], item[1][2], item[1][3], item[1][4], item[1][5], item[1][6], item[1][7],
                                             item[2][0], item[2][1], item[2][2], self.current_challenge.get_index()])
                else:
                    print("INCOMPLETE SAMPLE DETECTED.")



        self.current_samples = []  # and initiate the list again

    def get_another_challenge(self, stage=1, anew=False):

        print("CURRENT STAGE: " + str(stage))
        if anew:
            self.pick_challenge()

        if stage > 3:
            stage = 1
            self.save_log()  # save new samples after every level
            #self.camera_thread.end = True  # split the camera feed to a new file
            #self.record_camera()  # and begin recording to another file

            self.get_another_challenge(stage, True)  # we begin anew, overflow

        # self.current_log_saving = False  # turn off log saving, we later turn it on in the Stage 3

        # stage 2 is when the actual task is being carried out

        if stage == 1:  # text stage
            self.current_original_duration = self.current_challenge.get_duration(stage)
        elif stage == 2: # stage for focusing or doing
            self.current_original_duration = random.uniform(10, 15)
        else: # pause stage
            self.current_original_duration = random.uniform(5, 7)

        self.prompter_toggle_lights(True)  # default: color red is on - do nothing, yet
        self.signal_start_duration.emit(int(self.current_original_duration),
                                        int(self.current_original_duration))

        print("Current original duration set to " + str(self.current_original_duration))

        self.current_eyes_open = self.current_challenge.get_eyes()
        self.current_movie = self.current_challenge.get_data(stage)
        self.progress.set_stage(stage)
        self.progress.set_remaining(self.current_original_duration)

        if stage == 1:
            self.signal_label_text.emit(self.current_challenge.text)

            self.save_current_state_change(self.current_challenge.get_index(),
                                           stage,
                                           0,
                                           0,
                                           1,  # text changed
                                           0,
                                           0)
            # self.camera_thread.quit()  # turn of the camera if it is not already

            self.beeper_thread.set_stage_sound(self.current_challenge.get_data(2))
            self.beeper_thread.start()
            self.save_current_state_change(self.current_challenge.get_index(),
                                           stage,
                                           0,
                                           0,
                                           0,
                                           1,  # challenge sound emitted
                                           0)

            self.signal_eyes_open.emit(self.current_eyes_open)
            self.save_current_state_change(self.current_challenge.get_index(),
                                           stage,
                                           0,
                                           1,  # eyes changed
                                           0,
                                           0,
                                           0)
        elif stage == 2:
            # Stage 2
            self.current_log_saving = True # we begin recording

            if self.current_challenge.get_index() == 1:  # left hand
                self.current_movie = "challenges/left_arrow.png"
            elif self.current_challenge.get_index() == 2:  # right hand
                self.current_movie = "challenges/right_arrow.png"
            else:
                self.current_movie = "challenges/medi.png"

            self.initiate_movie()
            self.save_current_state_change(self.current_challenge.get_index(),
                                           stage,
                                           0,
                                           0,
                                           0,
                                           0,
                                           1)  # image changed

            self.beeper_thread.set_start_sound()
            self.beeper_thread.start()
            self.save_current_state_change(self.current_challenge.get_index(),
                                           stage,
                                           0,
                                           0,
                                           0,
                                           2,  # start sound emitted
                                           0)

            self.prompter_toggle_lights(False)
            self.save_current_state_change(self.current_challenge.get_index(),
                                           stage,
                                           1,  # colors changed
                                           0,
                                           0,
                                           0,
                                           0)
        else:
            # stage 3
            # pause between stages
            print(self.symbols_status)
            self.current_log_saving = False # turn off recording
            self.initiate_movie()
            self.save_current_state_change(self.current_challenge.get_index(),
                                           stage,
                                           0,
                                           0,
                                           0,
                                           0,
                                           1)  # image changed

            self.beeper_thread.set_end_sound()  # end of stage
            self.beeper_thread.start()
            self.save_current_state_change(self.current_challenge.get_index(),
                                           stage,
                                           0,
                                           0,
                                           0,
                                           3,  # end sound emitted
                                           0)

            self.prompter_toggle_lights(True)
            self.save_current_state_change(self.current_challenge.get_index(),
                                           stage,
                                           1,  # colors changed
                                           0,
                                           0,
                                           0,
                                           0)

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

    def save_current_state_change(self, index=0, stage=0, color_lights=0, eyes=0, text=0, sound=0, image=0):
        self.session_labels_dict[str(int(time.time()))] = [
            index, stage, color_lights, eyes, text, sound, image
        ]

        with open("out/" + self.current_log_filename + "_labels.json", "a") as f:
            json.dump(self.session_labels_dict, f)

        self.session_labels_dict

    def run(self):
        print("Starting Challenges")
        self.load_data()

        self.current_log_filename = str(int(time.time()))
        self.current_state_filename = self.current_log_filename + "_labels"
        fieldnames = ['timestamp', 'node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7', 'node8', 'accX', 'accY', 'accZ', 'class']
        with open(self.current_log_filename + "_dataset.csv", "w", newline='') as f:
            dataset_writer = csv.writer(f, delimiter=",")
            dataset_writer.writerow(fieldnames)

        self.get_another_challenge(1, True)
