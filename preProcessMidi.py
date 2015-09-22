__author__ = 'shawn'
from mido import MidiFile
import time
import math

file_name = 'test-sageev-bach-1_extracted.txt'
# Things to do
#   add time in which each note started
#   Lenght untill next note.
#   Check for drift
#   Tick Number (Assume 8 ticks per beat (0-31)) and bpm is 60
#   Generate the "inbetween" data with conntiuation flag
#   Creating the binary vectors
def parse_line(data):
    data = data.replace('\n','').split(' ')
    pitch_class = data[1].split('=')[1]
    octave = data[2].split('=')[1]
    start = data[3].split('=')[1]
    duration = data[4].split('=')[1]
    velocity = data[5].split('=')[1]
    bass = data[6].split('=')[1]
    return {'pitch_class': int(pitch_class), 'octave': int(octave), 'velocity': int(velocity), 'bass': int(bass),  'start': float(start), 'duration': float(duration)}

def makeVector(size,onIndex):
    x = [0] * size
    x[onIndex] = 1
    return x


bpm = 60
spb = bpm / 60
spba = spb * 4
spt = spb / 4
print(bpm)
# print(bps)
# print(tps)
with open(file_name) as f:
    lines = f.readlines()
    t_data = {}
    b_data = {}
    start_offset = -1
    largestKey = -1
    for line in lines:
        line = parse_line(line)

        if start_offset < 0:
            start_offset = line['start']

        current_time = line['start'] - start_offset
        pitch = makeVector(12, line['pitch_class'])
        octave = makeVector(11, line['octave'])
        velocity = line['velocity'] / 127
        firs_time_played = 1
        while current_time <= line['start'] + line["duration"] - start_offset:
            current_beat = (math.floor(current_time * 1000) % 4000 )// 1000
            current_tick = math.floor((float('0.' + str(current_time).split('.')[1]) * 1000 )/ 250)  %4

            pos_in_bar = ((current_beat * 4) + current_tick )/ 16
            note_length = (line['start'] + line["duration"] - start_offset) / spba
            if note_length > 1:
                note_length = 1
            if math.floor((current_time * 1000 )/ 250) > largestKey:
                largestKey = math.floor((current_time * 1000 )/ 250)
            if line['bass']:
                if not (math.floor((current_time * 1000 )/ 250) in b_data):
                    b_data.update({math.floor((current_time * 1000 )/ 250): []})
                b_data[math.floor((current_time * 1000 )/ 250)].append([pitch, octave, makeVector(4, current_beat), makeVector(4, current_tick), pos_in_bar, velocity, note_length, firs_time_played])
            else:
                if not (math.floor((current_time * 1000 )/ 250) in t_data):
                    t_data.update({math.floor((current_time * 1000 )/ 250): []})
                t_data[math.floor((current_time * 1000 )/ 250)].append([pitch, octave, makeVector(4, current_beat), makeVector(4, current_tick), pos_in_bar, velocity, note_length, firs_time_played])
            firs_time_played = 0
            current_time += spt
    target = open(file_name + '_net_data.txt','w')
    print(t_data)
    print(b_data)

    for i in range(0,largestKey + 1):
        t_current = []
        if i in t_data:
            t_current = t_data[i][0]
        else:
             t_current = [0] * 32
        first = True
        for y in t_current:
            if not first:
                target.write(' ')
            first = False
            if isinstance(y, list):
                fist_in_inside_loop = True;
                for a in y:
                    if not fist_in_inside_loop:
                        target.write(' ')
                    fist_in_inside_loop = False
                    target.write(str(a))
            else:
                target.write(str(y))
        target.write('\n')

        b_current = []
        print(i)

        if i in b_data:
            b_current = b_data[i][0]
        else:
            b_current = [0] * 35
        first = True
        for y in b_current:
            if not first:
                target.write(' ')
            first = False
            if isinstance(y, list):
                fist_in_inside_loop = True;
                for a in y:
                    if not fist_in_inside_loop:
                        target.write(' ')
                    fist_in_inside_loop = False
                    target.write(str(a))
            else:
                target.write(str(y))
        target.write('\n')
