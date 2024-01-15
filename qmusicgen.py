from NQubitSystem import NQubitSystem
import numpy as np
from constants import gates_map
from Gate import Gate
import qrandomnrgen
from music21 import stream, note
from pydub import AudioSegment
import librosa
import librosa.display
import subprocess 
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

notes = {'000': 'C', '001': 'D', '010': 'E', '011': 'F', '100': 'G', '101': 'A', '110': 'B', '111': 'C'}
quarterLength = {'000': 4.0 , '001': 2.0, '010': 1.0, '011': 0.5, '100': 0.25, '101': 3.0, '110': 2.5, '111': 0.75}
pitches = {'00': '', '01': '#', '10':'b', '11':''}
nrsounds = int(input("Enter number of sounds for composition:"))

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

# Create wav file
midi_file = s.write('midi', fp='output.mid')
# audio = AudioSegment.from_file(midi_file, format="mid")
# audio.export("output.wav", format="wav")

# output_wav_file = 'output.wav'

#run(['fluidsynth', '-F', output_wav_file, '--audio-driver=dsound', '--gain=3.0', 'default.sf2', midi_file])
# subprocess.run(['timidity', midi_file, '-Ow', '-o', output_wav_file])
# Listen to the output
s.show("midi")

# Create spectogram 
#audio_file = "output.wav"
#y, sr = librosa.load(audio_file)
#D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
#librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
#plt.colorbar(format='%+2.0f dB')
#plt.title('Spectrogram')
#plt.show()