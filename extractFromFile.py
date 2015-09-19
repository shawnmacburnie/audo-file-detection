__author__ = 'shawn'
from mido import MidiFile
import time
def extract(file_name):
    bass_note_split = 53
    startTime = time.time()
    current_time_in_song = 0
    current_files = {}
    target = open(file_name.replace('.mid','') + '_extracted.txt','w')
    print('Reading from ' + file_name + ' and generating data ... ')
    for message in MidiFile(file_name).play():
        current_time_in_song += message.time
        if hasattr(message, 'velocity'):
            if message.velocity == 0 or message.type == 'note_off':
                note = message.note
                pitch_class = note % 12
                octave = (note // 12) -1
                bass_note = 0
                if note <= bass_note_split:
                    bass_note = 1
                target.write('note=' + str(note) + ' pitch_class=' + str(pitch_class) + ' octave=' + str(octave) +' start=' + str(current_files[message.note][0]) + ' duration=' + str(current_time_in_song - current_files[message.note][0])  + ' velocity=' + str(current_files[message.note][1].velocity) + ' base=' + str(bass_note) + '\n')
            else:
                current_files[message.note] = [current_time_in_song,message]
    target.close()
    print('Data generated and saved to ' + file_name.replace('.mid','') + '_extracted.txt')
    print("Process took: " + str(round(time.time() - startTime, 3)))
extract('test-sageev-bach-1.mid')





