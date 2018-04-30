import numpy as np
from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyaudio
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtGui, QtCore
import  wave
import os
import sys
import trimmer as tr
import speech_recognition as s_r
import pickle
import re
import os
import shutil





# agar az ghabl vojood dasht pak kon dobare besaz

if os.path.exists(os.path.join(os.getcwd() , "Trimmed_voice")):

    shutil.rmtree(os.path.join(os.getcwd() , "Trimmed_voice"))
    os.mkdir(os.path.join(os.getcwd(), "Trimmed_voice"))
else:
    os.mkdir(os.path.join(os.getcwd(), "Trimmed_voice"))



# Load saved train vices

f = open('clf.pckl', 'rb')
g = open('train.pckl', 'rb')
train = pickle.load(g)
clf = pickle.load(f)
f.close()
g.close()


Form = uic.loadUiType(os.path.join(os.getcwd(), 'main.ui'))[0]

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
#RECORD_SECONDS = 0.5
WAVE_OUTPUT_FILENAME = "recorded.wav"

audio = pyaudio.PyAudio()

class plotW(QMainWindow, Form):

    def __init__(self):
        QMainWindow.__init__(self)
        Form.__init__(self)
        self.setupUi(self)

        self.Play.clicked.connect(self.start)
        self.Stop.clicked.connect(self.stop)
        self.fig = Figure(frameon=False)
        self.fig.patch.set_color('w')


        self.ax = self.fig.add_axes([0.1, 0., 0.8, 1.], frameon=False)
        self.canvas = FigureCanvas(self.fig)
        self.str = ""
    def start(self):
         self.state.setText("Recording...\nPress Stop button to stop recording ")
         self.Plotthread = Plotthread()

         self.Plotthread.frame_signal.connect(self.update_plot)

         self.Plotthread.start()

    def update_plot(self, a):

        self.str = self.str + " " + str(a)
        #print(self.str)
        self.number.setText(self.str)
    def stop(self):
        self.Plotthread.stop_flag = True
        self.state.setText(" Finish recording ")
        #print(self.str)



class Plotthread(QtCore.QThread):
    frame_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        QtCore.QThread.__init__(self)
        self.stop_flag = False

    def run(self):

         frames = []
         stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
         print("recording...")
         while self.stop_flag == False:
             print("recording...")
             # start Recording

             data = stream.read(CHUNK)
             frames.append(data)



         print( "---------------finisshhhhhhhhhhhhhhed--------------------------")
         #storing the voice in a file
         stream.stop_stream()
         stream.close()
         audio.terminate()

         waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
         waveFile.setnchannels(CHANNELS)
         waveFile.setsampwidth(audio.get_sample_size(FORMAT))
         waveFile.setframerate(RATE)
         waveFile.writeframes(b''.join(frames))
         waveFile.close()

         #s_r.feature_extraction(WAVE_OUTPUT_FILENAME)
         tr.trimmer('recorded.wav')
         #

         addres = os.path.join(os.getcwd(), 'Trimmed_voice')
         list = os.listdir(addres)
         file_count = len(list)


         for i in range(file_count):
             test_path = os.path.join(os.getcwd(), "Trimmed_voice")

             test_mfcc = s_r.feature_extraction(
             os.path.join(test_path, "saleh0" + str(i) + ".wav"))  # Detect each number by feature extracting

             test = np.reshape(test_mfcc, (1, test_mfcc.shape[0] * test_mfcc.shape[1]))
             #print(type(clf.predict(test)))
             self.frame_signal.emit(clf.predict(test))











app = QApplication(sys.argv)
w = plotW()
w.show()
sys.exit(app.exec_())