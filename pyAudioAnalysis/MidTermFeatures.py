import glob
import os
import time

import numpy as np

from pyAudioAnalysis import ShortTermFeatures, audioBasicIO

STEP_SIZE = 5  # seconds
eps = 0.00000001

""" Time-domain audio features """


def mid_feature_extraction(signal, sampling_rate, mid_window, mid_step, short_window, short_step):
    """
    Mid-term feature extraction

    ['zcr_mean', 'energy_mean', 'energy_entropy_mean', 'spectral_centroid_mean',
     'spectral_spread_mean', 'spectral_entropy_mean', 'spectral_flux_mean',
     'spectral_rolloff_mean', 'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean',
     'mfcc_4_mean', 'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_7_mean', 'mfcc_8_mean',
     'mfcc_9_mean', 'mfcc_10_mean', 'mfcc_11_mean', 'mfcc_12_mean', 'mfcc_13_mean',
     'chroma_1_mean', 'chroma_2_mean', 'chroma_3_mean', 'chroma_4_mean', 'chroma_5_mean',
     'chroma_6_mean', 'chroma_7_mean', 'chroma_8_mean', 'chroma_9_mean', 'chroma_10_mean',
     'chroma_11_mean', 'chroma_12_mean', 'chroma_std_mean', 'zcr_std', 'energy_std',
     'energy_entropy_std', 'spectral_centroid_std', 'spectral_spread_std',
     'spectral_entropy_std', 'spectral_flux_std', 'spectral_rolloff_std', 'mfcc_1_std',
     'mfcc_2_std', 'mfcc_3_std', 'mfcc_4_std', 'mfcc_5_std', 'mfcc_6_std', 'mfcc_7_std',
     'mfcc_8_std', 'mfcc_9_std', 'mfcc_10_std', 'mfcc_11_std', 'mfcc_12_std', 'mfcc_13_std',
     'chroma_1_std', 'chroma_2_std', 'chroma_3_std', 'chroma_4_std', 'chroma_5_std',
     'chroma_6_std', 'chroma_7_std', 'chroma_8_std', 'chroma_9_std', 'chroma_10_std',
     'chroma_11_std', 'chroma_12_std', 'chroma_std_std']
    """

    short_features = ShortTermFeatures.feature_extraction(
        signal, sampling_rate, short_window, short_step
    )

    n_stats = 2
    n_feats = len(short_features)
    # mid_window_ratio = int(round(mid_window / short_step))
    mid_window_ratio = round((mid_window - (short_window - short_step)) / short_step)
    mt_step_ratio = int(round(mid_step / short_step))

    mid_features = []
    for i in range(n_stats * n_feats):
        mid_features.append([])

    # for each of the short-term features:
    for i in range(n_feats):
        cur_position = 0
        num_short_features = len(short_features[i])

        while cur_position < num_short_features:
            end = cur_position + mid_window_ratio
            if end > num_short_features:
                end = num_short_features
            cur_st_feats = short_features[i][cur_position:end]

            mid_features[i].append(np.mean(cur_st_feats))
            mid_features[i + n_feats].append(np.std(cur_st_feats))
            cur_position += mt_step_ratio
    mid_features = np.array(mid_features)
    mid_features = np.nan_to_num(mid_features)
    return mid_features, short_features


def directory_feature_extraction(folder_path, mid_window, mid_step, short_window, short_step):
    """
    This function extracts the mid-term features of the WAVE files of a
    particular folder.

    The resulting feature vector is extracted by long-term averaging the
    mid-term features.
    Therefore ONE FEATURE VECTOR is extracted for each WAV file.

    ARGUMENTS:
        - folder_path:        the path of the WAVE directory
        - mid_window, mid_step:    mid-term window and step (in seconds)
        - short_window, short_step:    short-term window and step (in seconds)
    """

    mid_term_features = np.array([])
    process_times = []

    types = ("*.wav", "*.aif", "*.aiff", "*.mp3", "*.au", "*.ogg")
    wav_file_list = []
    for files in types:
        wav_file_list.extend(glob.glob(os.path.join(folder_path, files)))

    wav_file_list = sorted(wav_file_list)
    wav_file_list2 = []
    for i, file_path in enumerate(wav_file_list):
        print("Analyzing file {0:d} of {1:d}: {2:s}".format(i + 1, len(wav_file_list), file_path))
        if os.stat(file_path).st_size == 0:
            print("   (EMPTY FILE -- SKIPPING)")
            continue
        sampling_rate, signal = audioBasicIO.read_audio_file(file_path)
        if sampling_rate == 0:
            continue

        t1 = time.time()
        signal = audioBasicIO.stereo_to_mono(signal)
        if signal.shape[0] < float(sampling_rate) / 5:
            print("  (AUDIO FILE TOO SMALL - SKIPPING)")
            continue
        for i in range(0, len(signal), STEP_SIZE * sampling_rate):
            subsignal = signal[i : i + (STEP_SIZE * sampling_rate)]
            if len(subsignal) < round(sampling_rate * short_step):
                # the remaining subsignal is smaller than the short_step, skip it
                continue
            wav_file_list2.append(file_path + "#" + str(i))
            mid_features, short_features = mid_feature_extraction(
                subsignal,
                sampling_rate,
                round(mid_window * sampling_rate),
                round(mid_step * sampling_rate),
                round(sampling_rate * short_window),
                round(sampling_rate * short_step),
            )

            mid_features = np.transpose(mid_features)
            mid_features = mid_features.mean(axis=0)
            # long term averaging of mid-term statistics
            if (not np.isnan(mid_features).any()) and (not np.isinf(mid_features).any()):
                if len(mid_term_features) == 0:
                    # append feature vector
                    mid_term_features = mid_features
                else:
                    mid_term_features = np.vstack((mid_term_features, mid_features))
        t2 = time.time()
        duration = float(len(signal)) / sampling_rate
        process_times.append((t2 - t1) / duration)
    if len(process_times) > 0:
        print(
            "Feature extraction complexity ratio: "
            "{0:.1f} x realtime".format((1.0 / np.mean(np.array(process_times))))
        )
    return mid_term_features, wav_file_list2


def multiple_directory_feature_extraction(
    path_list, mid_window, mid_step, short_window, short_step
):
    """
    Same as dirWavFeatureExtraction, but instead of a single dir it
    takes a list of paths as input and returns a list of feature matrices.
    EXAMPLE:
    [features, classNames] =
           a.dirsWavFeatureExtraction(['audioData/classSegmentsRec/noise',
                                       'audioData/classSegmentsRec/speech',
                                       'audioData/classSegmentsRec/brush-teeth',
                                       'audioData/classSegmentsRec/shower'], 1,
                                       1, 0.02, 0.02);

    It can be used during the training process of a classification model ,
    in order to get feature matrices from various audio classes (each stored in
    a separate path)
    """

    # feature extraction for each class:
    features = []
    class_names = []
    file_names = []
    for i, d in enumerate(path_list):
        f, fn = directory_feature_extraction(d, mid_window, mid_step, short_window, short_step)
        if f.shape[0] > 0:
            # if at least one audio file has been found in the provided folder:
            features.append(f)
            file_names.append(fn)
            if d[-1] == os.sep:
                class_names.append(d.split(os.sep)[-2])
            else:
                class_names.append(d.split(os.sep)[-1])
    return features, class_names, file_names
