from NQubitSystem import NQubitSystem
import numpy as np
from constants import gates_map
from Gate import Gate
import qrandomnrgen
import math        #import needed modules
import pyaudio     #sudo apt-get install python-pyaudio
from scipy.io import wavfile


PyAudio = pyaudio.PyAudio
sampling_rate = 44100     #number of frames per second/frameset.
freq = {'00': 500 , '01': 1000, '10': 2000, '11': 10000}   #Hz, waves per second, 261.63=C4-note.
length = {'00': 2 , '01': 3, '10': 4, '11': 5}    #seconds to play sound
overtones = 20

nrsounds = int(input("Enter number of sounds for composition:"))

# final_signal = np.ones(1, dtype=int)
# print(final_signal)
""" for i in range(nrsounds):
    amplitude = qrandomnrgen.samplerandominrange(5)
    f = freq[qrandomnrgen.sample2quantumgenerator()]
    l = length[qrandomnrgen.sample2quantumgenerator()]
    t = np.arange(0, l, 1/sampling_rate)
    signal = amplitude * np.sin(np.pi * 2 * f * t)
    signal *= 32767
    signal = np.int16(signal)
    
    wavfile.write("file.wav", sampling_rate, signal) """
#final_signal = np.append(final_signal, signal)
#print(type(signal))
frequency = 1000
time = np.arange(0, 1, 1/sampling_rate)
for i in range(nrsounds):
    Y = np.sin(2 * np.pi * frequency * time) * np.exp(-0.0004 * 2 * np.pi * frequency * time) 
    for j in range(overtones):
        Y += np.sin((j+1) * 2 * np.pi * frequency * time) * np.exp(-0.0004 * 2 * np.pi * frequency * time) / 2**(j+1)
    #Y += Y * Y * Y
    #Y *= 1 + 16 * time * np.exp(-6 * time)
    Y *= 32767
    Y = np.int16(Y)

wavfile.write("file1.wav", sampling_rate, Y)
    
#print(final_signal)
#wavfile.write("file.wav", sampling_rate, final_signal)

#final_signal = np.int16(np.array(final_signal))
#wavfile.write("file.wav", sampling_rate, final_signal)
