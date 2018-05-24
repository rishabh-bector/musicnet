from mido import MidiFile, MidiTrack, Message
import mido
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
import os
from tqdm import *



###   HYPERPARAMETERS   ###

LOOKBACK = 25

EPOCHS = 100

TRAIN = False
TEST = True

LOAD = True
FILEFOLDER = 'training'

GENERATION = 500



###   ENCODERS/DECODERS   ###



class Decoder:
    def __init__(self, filename):
        self.filename = filename
        self.mid = MidiFile()
        self.track = MidiTrack()
        self.mid.tracks.append(self.track)

    def addNote(self, msg):
        if msg[0] > msg[1]:
            dataType = 'note_on'
        else:
            dataType = 'note_off'

        self.track.append(Message(dataType, note=reverseOneHot(msg[2:102]), time=reverseOneHot(msg[102:])))
        self.mid.save(self.filename)

    def output(self):
        for m in self.track:
            print(m)

class Encoder:
    def __init__(self, filename):
        self.filename = filename
        self.mid = MidiFile()
        self.track = MidiTrack()
        self.mid.tracks.append(self.track)

    def encodeMidi(self, filename):
        mid = MidiFile(filename)
        song = []
        highestNote, highestTime = findHighestValues(mid)
        for track in mid.tracks:
            print(len(track))
            for msg in track:
                try:
                    if msg.type == 'note_on':
                        msgType = [1, 0]
                    else:
                        msgType = [0, 1]
                    song.append(flatten([msgType, oneHot(msg.note, highestNote), oneHot(msg.time, highestTime)]))
                except Exception:
                    a = 0
        return song

    def addMsg(self, msg):
        self.track.append(msg)
        self.mid.save(self.filename)

    def processFolder(self, folderName):
        files = os.listdir(folderName + '/')
        print('Encoding {} MIDI files'.format(len(files)))
        for f in tqdm(files):
            try:
                midFile = MidiFile(f)
            except Exception:
                try:
                    midFile = mido.read_syx_file(f)
                except Exception:
                    continue
            for track in midFile.tracks:
                for msg in track:
                    self.track.append(msg)
        self.mid.save(self.filename)
        return self.encodeMidi(self.filename)



###   HELPER FUNCTIONS   ###



def oneHot(dataPoint, highest):
    oneHotList = [0 for x in range(highest + 1)]
    oneHotList[dataPoint] = 1
    return oneHotList

def reverseOneHot(data):
    highest = -100
    for i in data:
        if i > highest:
            highest = i
    if highest == 0:
        print('Hmmm...')
        print(len(data))
    return data.index(highest)

def flatten(data):
    final = []
    for l1 in data:
        try:
            for item in l1:
                final.append(item)
        except Exception:
            final.append(l1)
    return final

def findHighestValues(mid):
    highestNote = 0
    highestTime = 0
    for track in mid.tracks:
        for msg in track:
            try:
                if msg.note > highestNote:
                    highestNote = msg.note
            except Exception:
                a = 0
            try:
                if msg.time > highestTime:
                    highestTime = msg.time
            except Exception:
                a = 0
    return 100, 500



###   MAIN   ###


encoder = Encoder('training.mid')
song = encoder.encodeMidi('training.mid')


print(len(song[0]))
dataX = []
dataY = []

place = 0
for msg in song:
    if place < (len(song) - LOOKBACK):
        dataX.append(flatten(song[place:place+LOOKBACK]))
        place += 1

for msg in song[LOOKBACK:]:
    dataY.append(flatten(msg))

dataY = np.array(dataY)

print(len(dataX[0]))
total = len(dataX[0])

dataX = np.array(dataX)
dataX = np.reshape(dataX, (len(dataX), LOOKBACK, int(total/LOOKBACK)))

print(type(dataX))

if TRAIN:
    print('building model...')
    model = Sequential()
    model.add(LSTM(256, input_shape=(dataX.shape[1], dataX.shape[2]), return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(256, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(dataY.shape[1], activation='tanh'))
    model.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop')
    if LOAD:
        model = load_model('model')
    print('beginning training...')
    model.fit(dataX, dataY, nb_epoch=EPOCHS)
    model.save('model')



###   TESTING   ###



def fixOneHot(output):
    try:
        output = output.tolist()
    except Exception:
        a = 0
    data = [0 for i in range(len(output))]
    highest = -100
    for i in output:
        if i > highest:
            highest = i
    print(highest)
    data[output.index(highest)] = 1
    return data

def fixCombinedOneHot(output):
    s0 = fixOneHot(output[:2])
    s1 = fixOneHot(output[2:102])
    s2 = fixOneHot(output[102:])
    return s0 + s1 + s2

def oneHotTest(fixed):
    count = 0
    for i in fixed:
        if i == 1:
            count += 1
    if count != 3:
        print('oh NO')


if TEST:
    print('loading model...')
    model = load_model('model')
    decoder = Decoder('output.mid')
    start = np.random.randint(0, len(dataX) - 1)
    seed = dataX[start]
    for j in seed:
        decoder.addNote(j.tolist())

    print('beginning generation...')
    for i in range(GENERATION):
        x = np.reshape(np.array([seed]), (1, LOOKBACK, int(total/LOOKBACK)))
        prediction = model.predict(x)[0]
        prediction = prediction.tolist()
        decoder.addNote(prediction)
        prediction = fixCombinedOneHot(prediction)
        oneHotTest(prediction)
        seed = seed.tolist()
        seed.append(prediction)
        seed = seed[1:]
        seed = np.array(seed)
        print('Note {} generated'.format(i))

    decoder.output()
