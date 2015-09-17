from scipy.io.wavfile import read
from scipy.io.wavfile import write

sound1 = read('train1.wav')
print(len(sound1[1]))
# audio3 = (sound1[1] + sound2[1])
# print(audio3)
write('testTrain.wav', sound1[0],sound1[1][:10000])