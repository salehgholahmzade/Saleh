import numpy as np
import matplotlib
from pylab import*
import scipy.io.wavfile as wavefile
from scipy.fftpack import fft
from scipy.fftpack import dct
from scipy.misc import imresize
from sklearn import mixture
import pickle

#########################################################################################

# Extract features of voice file


# Load saved train vices

# f = open('gmm.pckl', 'rb')
# g = open('train.pckl', 'rb')
# train = pickle.load(g)
# gmm = pickle.load(f)
# f.close()
# g.close()



def feature_extraction(address):

    sample_rate, signal = wavefile.read(address)  # File assumed to be in the same directory

    time_duration = signal.shape[0] / sample_rate

    signal = signal[0:int(2.5 * sample_rate)]  # Keep the first 2.5 or less

    signal = signal[(signal != 0)]
    signal = signal[abs(signal) > 20]


    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    ###################################  Framing  ################################################

    frame_size = 0.025
    frame_stride = 0.01

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length

    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)  # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    ##############################  Windowing  ###########################################

    frames *= np.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **

    ############################# Short Time Fourier Transform #############################

    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    # Make filter bank

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))

    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # Convert to dB

    ################################# SPECTROGRAM ######################################

    data = filter_banks[0:num_frames, :]
    #print('old_data_size', data.shape)
    new_x = 80
    data = set_size(data, new_x , 40)
    data = data.T
    #print('new_data_size', data.shape)
    #print('num_frames', num_frames)
    ############################### Plot spectogram ####################################

    # fig = plt.figure(figsize = (new_x, nfilt), dpi=50)
    #
    # plt.xlabel("time (ms)", fontsize=30)
    # plt.ylabel("Frequency  ", fontsize=30)
    #
    # ax = fig.add_subplot(111)
    #
    # plt.title('Spectrogram', fontsize=30)
    #
    # cax = ax.matshow(data, aspect="auto")
    # fig.colorbar(cax, aspect="auto")
    #
    # plt.show()

    ######################### Mel-Frequency Cepstral Coefficient (MFCC) #####################################

    num_ceps = 12
    mfcc = dct(data, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13

    (nframes, ncoeff) = mfcc.shape

    n = np.arange(ncoeff)
    cep_lifter = 22

    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift

    #mfcc = mfcc[0:num_frames, :]
    #mfcc = set_size(mfcc, 200, 12)
    mfcc = mfcc.T
    #print('mfcc_size', mfcc.shape)

    ################################ Plot MFCC ###################################3

    # fig = plt.figure(figsize=(new_x, num_ceps), dpi=50)
    #
    # #labels
    # plt.xlabel("time (ms)", fontsize=30)
    # plt.ylabel("MFCC_coefficient", fontsize=30)
    #
    # ax = fig.add_subplot(111)
    #
    # plt.axis([0, new_x - 1, 0, num_ceps - 1])
    #
    # plt.title('MFCCS', fontsize=30)
    #
    # cax = ax.matshow(data, aspect = "auto")
    # fig.colorbar(cax, aspect = "auto")
    #
    # plt.show()

    #return mfcc
    return data
##############################################################################################

########## Set frame size  ###############

def set_size(mat ,lx , ly):
    a = imresize(mat, [lx, ly], 'bilinear', mode='F')
    return a

# if __name__ == '__main__' :
#     a = feature_extraction('saleh00.wav')
#     test = np.reshape(a, (1, a.shape[0] * a.shape[1]))
#     print(gmm.predict(test))