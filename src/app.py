from __future__ import annotations
import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2
import numpy as np
import keras
import time
import sys

from config import config_parser
from mp_util_legacy import create_holistic, mediapipe_extract_single
from llm_util import slt_sentence_predict

from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

def log_debug(message: str):
    print(f"[DEBUG] {message}")

class VideoThread(QThread):
    update_frame = Signal(QImage)
    update_pred_probability = Signal(dict)
    update_cur_pred_label = Signal(str)
    update_cur_pred_sentence = Signal(str)
    update_cur_pred_natural = Signal(str)

    def __init__(self, config_path: str, parent=None):
        log_debug("Constructing video thread")
        QThread.__init__(self, parent)
        self.config = config_parser(config_path)
        self.cap = cv2.VideoCapture(self.config.capture.source)
        self.width = int(self.config.capture.resolution_width)
        self.height = int(self.config.capture.resolution_height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.sequence = []
        self.sentence = []
        self.slr_model = keras.models.load_model(self.config.paths.model)
        self.holistic = create_holistic()
        self.frame_num = 0
        self.seq_max_len = self.config.sequence.frame
        self.all_labels = sorted(os.listdir(self.config.paths.keypoints))
        self.threshold = 0.9

        self.status = False
        self.killed = False
        log_debug("Video thread constructed")

    def clear(self):
        self.sequence = []
        self.sentence = []
        self.frame_num = 0

    def run(self):
        self.run_isolated()

    def run_isolated(self):
        # frames = []
        while True:
            if self.killed:
                log_debug("Kill signal received")
                break
            if not self.status:
                continue
            if not self.cap.isOpened():
                self.update_frame.emit(QImage())
                self.update_pred_probability.emit({})
                self.update_cur_pred_label.emit("")
                self.update_cur_pred_sentence.emit("")
                continue
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_num += 1
            # frames.append(frame)
            keypoint = mediapipe_extract_single(frame, self.holistic, visualize=True)
            keypoint = keypoint.flatten()
            self.sequence.append(keypoint)
            self.sequence = self.sequence[-self.seq_max_len:]
            if self.frame_num % 30 == 0:
                prediction = self.predict_gloss()
                if prediction is not None:
                    self.update_pred_probability.emit({label: probability for label, probability in zip(self.all_labels, prediction)})
                    cur_pred_label = self.all_labels[np.argmax(prediction)]
                    self.update_cur_pred_label.emit(cur_pred_label)
                    if len(self.sentence) == 0 or (len(self.sentence) > 0 and cur_pred_label != self.sentence[-1]):
                        self.sentence.append(cur_pred_label)
                    self.update_cur_pred_sentence.emit(" ".join(self.sentence))
                else:
                    self.update_pred_probability.emit({})
                    self.update_cur_pred_label.emit("")
                    self.update_cur_pred_sentence.emit("")
                # add timeout
                time.sleep(1)
            self.update_frame.emit(self.convert_frame(frame))
        log_debug("All frames processed, releasing camera")
        self.cap.release()
        self.status = False

    def predict_gloss(self):
        if len(self.sequence) == self.seq_max_len:
            sequence = np.array(self.sequence)
            sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
            prediction = self.slr_model.predict(sequence, verbose=0)[0]
            return prediction
        return None

    def predict_natural(self):
        result = slt_sentence_predict(" ".join(self.sentence), self.config)
        self.update_cur_pred_natural.emit(result)

    def convert_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    
    def __del__(self):
        self.cap.release()
        
class MainWindow(QMainWindow):
    def __init__(self, config_path: str):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition")
        self.setFixedSize(1920, 1080)

        self.video_thread = VideoThread(config_path, self)
        self.video_thread.update_frame.connect(self.update_frame)
        self.video_thread.update_pred_probability.connect(self.update_pred_probability)
        self.video_thread.update_cur_pred_label.connect(self.update_cur_pred_label)
        self.video_thread.update_cur_pred_sentence.connect(self.update_cur_pred_sentence)
        self.video_thread.update_cur_pred_natural.connect(self.update_cur_pred_natural)
        # self.video_thread.start()

        self.video_label = QLabel()
        self.pred_probability = QLabel()
        self.cur_pred_label = QLabel()
        self.cur_pred_sentence = QLabel()
        self.cur_pred_sentence.setWordWrap(True)
        self.cur_pred_natural = QLabel()
        self.cur_pred_natural.setWordWrap(True)

        self.button_start = QPushButton("Start")
        self.button_start.clicked.connect(self.start_video)
        self.button_stop = QPushButton("Stop")
        self.button_stop.clicked.connect(self.stop_video)
        self.button_pred_natural = QPushButton("Predict Natural Sentence")
        self.button_pred_natural.clicked.connect(self.predict_natural)
        self.button_clear = QPushButton("Clear")
        self.button_clear.clicked.connect(self.clear)
        self.button_kill = QPushButton("Kill Thread")
        self.button_kill.clicked.connect(self.kill_video)
        self.button_restart = QPushButton("Restart")
        self.button_restart.clicked.connect(self.restart_video)

        buttons_layout = QVBoxLayout()
        buttons_layout_row1 = QHBoxLayout()
        buttons_layout_row1.addWidget(self.button_start)
        buttons_layout_row1.addWidget(self.button_stop)
        buttons_layout_row1.addWidget(self.button_pred_natural)
        buttons_layout_row2 = QHBoxLayout()
        buttons_layout_row2.addWidget(self.button_clear)
        buttons_layout_row2.addWidget(self.button_kill)
        buttons_layout_row2.addWidget(self.button_restart)

        buttons_layout.addLayout(buttons_layout_row1)
        buttons_layout.addLayout(buttons_layout_row2)

        cam_pred_layout = QHBoxLayout()
        cam_pred_layout.addWidget(self.video_label)
        cam_pred_layout.addWidget(self.pred_probability)

        widget_layout = QVBoxLayout()
        widget_layout.addLayout(buttons_layout)
        widget_layout.addLayout(cam_pred_layout)
        widget_layout.addWidget(self.cur_pred_label)
        widget_layout.addWidget(self.cur_pred_sentence)
        widget_layout.addWidget(self.cur_pred_natural)

        widget = QWidget()
        widget.setLayout(widget_layout)
        self.setCentralWidget(widget)

        log_debug("Main window starting")
        self.update_frame(QImage())
        self.update_pred_probability({})
        self.update_cur_pred_label("")
        self.update_cur_pred_sentence("")
        self.update_cur_pred_natural("")
        log_debug("Initial labels defined. Starting video thread...")
        self.video_thread.start()
        self.video_thread.status = True
        log_debug("Main window started")

    @Slot()
    def start_video(self):
        log_debug("Start button clicked")
        self.video_thread.status = True

    @Slot()
    def stop_video(self):
        log_debug("Stop button clicked")
        self.video_thread.status = False

    @Slot()
    def predict_natural(self):
        log_debug("Predict Sentence button clicked")
        self.update_cur_pred_natural("Please wait...")
        self.video_thread.predict_natural()

    @Slot()
    def clear(self):
        log_debug("Clear button clicked")
        self.video_thread.clear()
        self.update_pred_probability({})
        self.update_cur_pred_label("")
        self.update_cur_pred_sentence("")
        self.update_cur_pred_natural("")
        
    @Slot()
    def kill_video(self):
        log_debug("Kill button clicked")
        if self.video_thread.killed: return
        self.video_thread.killed = True
        self.video_thread.exit()
        log_debug("Thread killed")

    @Slot()
    def restart_video(self):
        log_debug("Restart button clicked")
        if not self.video_thread.killed:
            self.video_thread.killed = True
            self.video_thread.exit()
        log_debug("Thread killed, restarting...")
        self.video_thread = VideoThread("config/config_clip.yaml" if os.name == "nt" else "config/config_clip_linux.yaml", self)
        self.video_thread.update_frame.connect(self.update_frame)
        self.video_thread.update_pred_probability.connect(self.update_pred_probability)
        self.video_thread.update_cur_pred_label.connect(self.update_cur_pred_label)
        self.video_thread.update_cur_pred_sentence.connect(self.update_cur_pred_sentence)
        self.video_thread.update_cur_pred_natural.connect(self.update_cur_pred_natural)
        self.video_thread.start()
        self.video_thread.status = True

    @Slot(QImage)
    def update_frame(self, frame):
        if frame is None:
            self.video_label.clear()
        else:
            self.video_label.setPixmap(QPixmap.fromImage(frame))

    @Slot(dict)
    def update_pred_probability(self, probabilities):
        output = "Prediction Probabilities:\n"
        for label, probability in probabilities.items():
            output += f"{label}: {probability}\n"
        self.pred_probability.setText(output)

    @Slot(str)
    def update_cur_pred_label(self, label):
        self.cur_pred_label.setText(f"Current Prediction: {label}")

    @Slot(str)
    def update_cur_pred_sentence(self, sentence):
        self.cur_pred_sentence.setText(f"Current Sentence: {sentence}")

    @Slot(str)
    def update_cur_pred_natural(self, sentence):
        self.cur_pred_natural.setText(f"Predicted Natural Sentence: {sentence}")

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    log_debug("Starting the application")
    app = QApplication([])
    window = MainWindow("config/config_clip.yaml" if os.name == "nt" else "config/config_clip_linux.yaml")
    window.show()
    sys.exit(app.exec())
