import os

import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile


def read_audio_file(input_file):
    """
    This function returns a numpy array that stores the audio samples of a
    specified WAV of AIFF file
    """

    sampling_rate = 0
    signal = np.array([])
    if isinstance(input_file, str):
        extension = os.path.splitext(input_file)[1].lower()
        if extension in [".wav"]:
            sampling_rate, signal = wavfile.read(input_file)  # from scipy.io
        elif extension in [".mp3", ".au", ".ogg"]:
            sampling_rate, signal = read_audio_generic(input_file)
        else:
            print("Error: unknown file type {extension}")
    else:
        sampling_rate, signal = read_audio_generic(input_file)

    if signal.ndim == 2 and signal.shape[1] == 1:
        signal = signal.flatten()

    return sampling_rate, signal


def read_audio_generic(input_file):
    """
    Function to read audio files with the following extensions
    [".mp3", ".au", ".ogg"], containing PCM (int16 or int32) data.

    """
    sampling_rate = -1
    signal = np.array([])
    try:
        audiofile = AudioSegment.from_file(input_file, parameters=["-ac", "1"])
        data = np.array([])
        if audiofile.sample_width == 2:
            data = np.fromstring(audiofile._data, np.int16)
        elif audiofile.sample_width == 4:
            data = np.fromstring(audiofile._data, np.int32)

        if data.size > 0:
            sampling_rate = audiofile.frame_rate
            temp_signal = []
            for chn in list(range(audiofile.channels)):
                temp_signal.append(data[chn :: audiofile.channels])
            signal = np.array(temp_signal).T
    except:
        print("Error: file not found or other I/O error. (DECODING FAILED)")
    return sampling_rate, signal


def stereo_to_mono(signal):
    """
    This function converts the input signal
    (stored in a numpy array) to MONO (if it is STEREO)
    """

    if signal.ndim == 2:
        if signal.shape[1] == 1:
            signal = signal.flatten()
        else:
            if signal.shape[1] == 2:
                signal = (signal[:, 1] / 2) + (signal[:, 0] / 2)
    return signal
