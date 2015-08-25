__author__ = 'shawn'

from scipy.io.wavfile import read
import matplotlib.pyplot as plt
def show_amplitude(file_name):
    input_data = read(file_name)
    audio = input_data[1]
    # plot the first 1024 samples
    plt.plot(audio[0:1024])
    # label the axes
    plt.ylabel("Amplitude")
    plt.xlabel("Time (samples)")
    # set the title
    plt.title(file_name)
    # display the plot
    plt.show()

import scipy
from scipy.signal import hann
from scipy.fftpack import rfft
def show_magnitued(file_name):
    # read audio samples
    input_data = read(file_name)
    audio = input_data[1]
    print(audio)
    # apply a Hanning window
    window = hann(1024)
    audio = audio[0:1024] * window
    # fft
    mags = abs(rfft(audio))
    # convert to dB
    mags = 20 * scipy.log10(mags)
    # normalise to 0 dB max
    # mags -= max(mags)
    file = open('tmp.txt', 'w')
    for i in mags:
        file.write(str(i) + '\n')
    file.close()
    # plot
    plt.plot(mags)
    # label the axes
    plt.ylabel("Magnitude (dB)")
    plt.xlabel("Frequency Bin")
    # set the title
    plt.title(file_name + " Spectrum")
    plt.show()



show_amplitude("train1.wav")
# show_magnitued("train1.wav")