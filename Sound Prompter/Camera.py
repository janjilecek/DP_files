# author: Jan Jilecek

import cv2
import time

from PyQt5.QtCore import QThread, pyqtSignal


class WebcamRecorder(QThread):
    end = False

    def __init__(self):
        super().__init__()

    def run(self):
        self.end = False
        self.record(self.duration, self.name)

    def set_details(self, _duration, _name):
        self.duration = _duration
        self.name = _name

    # burns/watermarks text to the frame
    def write_text(self, frame, text, height=480):
        cv2.putText(img=frame, text=text, org=(int(20), int(height - 30)),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2,
                    color=(0, 255, 0))

    # record video for the time period specified
    def record(self, seconds=10, name="default"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():  # Check if camera opened successfully
            print("Camera feed could not be opened.")

        frame_width = int(cap.get(3))  # default camera resolution
        frame_height = int(cap.get(4))
        start_time = time.time()
        print("Recording: " + str(start_time))
        # TODO use better compression algo
        out = cv2.VideoWriter(name + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                              (frame_width, frame_height))  # 30 FPS

        while int(time.time() - start_time) < seconds:
            ret, frame = cap.read()
            if ret:
                self.write_text(frame, str(int(round(time.time() * 1000))), frame_height)  # write time in milliseconds
                out.write(frame)
                if self.end:
                    print("Ending camera recording...")
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
