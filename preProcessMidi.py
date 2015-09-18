__author__ = 'shawn'
from mido import MidiFile
import time

file_name = 'test-sageev-bach-1.mid'
# Things to do
#   add time in which each note started
#   Lenght untill next note.
#   Check for drift
#   Tick Number (Assume 8 ticks per beat (0-31)) and bpm is 60
#   Generate the "inbetween" data with conntiuation flag
#   Creating the binary vectors

t_notes = []
b_notes = []
base_note_split = 53
print('Reading from ' + file_name + ' and generating data ... ')
t_lastNotePlayed = -1
b_lastNotePlayed = -1
for message in MidiFile(file_name).play():
    if hasattr(message, 'velocity') and message.type == 'note_on':
        note = message.note
        velocity = message.velocity / 127
        pitch_class = note % 12
        octave = (note // 12) -1
        mutation = 0
        if note <= base_note_split:
            if b_lastNotePlayed != -1:
                mutation = note - b_lastNotePlayed
            b_lastNotePlayed = note
            b_notes.append({'note': note, 'velocity': velocity,'pitch_class': pitch_class, 'octave': octave, 'mutation': mutation})
        else:
            if t_lastNotePlayed != -1:
                mutation = note - t_lastNotePlayed
            t_lastNotePlayed = note
            t_notes.append({'note': note, 'velocity': velocity, 'pitch_class': pitch_class, 'octave': octave, 'mutation': mutation})
print(t_notes)
print(b_notes)
