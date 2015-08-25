from scipy.io.wavfile import read
from scipy.io.wavfile import write

sound1 = read('train1.wav')
sound2 = read('train2.wav')

audio3 = (sound1[1] + sound2[1])
print(audio3)
write('train1And2.wav', sound1[0],audio3)