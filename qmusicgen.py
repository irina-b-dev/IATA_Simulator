from NQubitSystem import NQubitSystem
import numpy as np
from constants import gates_map
from Gate import Gate
import qrandomnrgen
import pyaudio     #sudo apt-get install python-pyaudio
from scipy.io import wavfile
from music21 import stream, note

notes = {'000': 'C', '001': 'D', '010': 'E', '011': 'F', '100': 'G', '101': 'A', '110': 'B', '111': 'C'}
quarterLength = {'000': 4.0 , '001': 2.0, '010': 1.0, '011': 0.5, '100': 0.25, '101': 3.0, '110': 2.5, '111': 0.75}
pitches = {'00': '', '01': '#', '10':'b', '11':''}
nrsounds = int(input("Enter number of sounds for composition:"))

# wavfile.write("file1.wav", sampling_rate, Y)
def create_note():
    pitch = qrandomnrgen.samplerandominrange(127)
    octave = int(pitch / 12)
    octave = str(octave)
    note1 = notes[qrandomnrgen.samplenrquantumgenerator(3)]
    pitch = pitches[qrandomnrgen.sample2quantumgenerator()]
    result = note1 + pitch + octave
    ql = quarterLength[qrandomnrgen.samplenrquantumgenerator(3)]
    return result, ql

s = stream.Stream()
for i in range(nrsounds):
    result, ql = create_note()
    n = note.Note(result, quarterLength=ql) 
    s.append(n)

s.show("midi")