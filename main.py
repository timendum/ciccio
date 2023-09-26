print("Starting...")
# flake8: noqa: E402
import argparse
import os
import subprocess

import requests

from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO

import podcast

MODEL_NAME = "data/svmSM"


def train(args):
    return aT.extract_features_and_train(
        ["data/ko", "data/ok"],
        1,
        1,
        aT.shortTermWindow,
        aT.shortTermStep,
        "svm",
        MODEL_NAME,
    )


def _analyze(signal, sampling_rate, model):
    (
        classifier,
        mean,
        std,
        _,
        mid_window,
        mid_step,
        short_window,
        short_step,
    ) = model
    classes = []
    probabilites = []
    for i in range(0, len(signal), aF.STEP_SIZE * sampling_rate):
        subsignal = signal[i : i + (aF.STEP_SIZE * sampling_rate)]
        # feature extraction:
        (
            mid_features,
            _,
        ) = aF.mid_feature_extraction(
            subsignal,
            sampling_rate,
            mid_window * sampling_rate,
            mid_step * sampling_rate,
            round(sampling_rate * short_window),
            round(sampling_rate * short_step),
        )
        # long term averaging of mid-term statistics
        mid_features = mid_features.mean(axis=1)
        feature_vector = (mid_features - mean) / std  # normalization

        # classification
        class_id = classifier.predict(feature_vector.reshape(1, -1))[0]
        classes.append(class_id)
        probability = classifier.predict_proba(feature_vector.reshape(1, -1))[0]
        probabilites.append(probability)
    return classes, probabilites


def _find_splits(classes, probabilites):
    current_class = 0  # 0 = ko, 1 = ok
    ongoning_change = 0  # number of steps with a different class since last split time
    split_at = []
    for n_step, class_id in enumerate(classes):
        if class_id != current_class:
            if ongoning_change >= 5:
                # switch! too many different steps
                current_class = class_id
                delta = 0
                # I'm going to look at the last ko (class = 0) frame
                # and set a look back variable (delta) based on how much that frame was a "ok",
                # looking at its probability
                if current_class == 1:
                    if n_step - ongoning_change > 0:
                        # look at previous "ko" frame
                        delta = -(probabilites[n_step - ongoning_change - 1][1] + 0.2)
                        # go back base on the probabilty of speech + 0.1
                else:
                    # look at last "ko" frame
                    delta = probabilites[n_step - ongoning_change][1] + 0.2
                    # include it based on the probabilty of speech + 0.1
                split_at.append((n_step - ongoning_change + delta) * aF.STEP_SIZE)
                ongoning_change = 0
            else:
                ongoning_change += 1
        else:
            ongoning_change = 0
    return split_at


def _split_file(source, split_at) -> list[str]:
    target = source.replace(".mp3", "")
    filenames = []
    for i in range(0, len(split_at), 2):
        split_command = ["ffmpeg", "-y", "-i", source, "-ss", str(split_at[i])]
        try:
            split_command.extend(["-to", str(split_at[i + 1])])
        except IndexError:
            # end of file
            pass
        filename = f"{target}_{int(i/2)+1:02}.mp3"
        split_command.extend(["-acodec", "copy", filename])
        subprocess.run(split_command, capture_output=True)
        filenames.append(filename)
    return filenames


def split(args) -> list[str]:
    print("Loading Model...")
    model = aT.load_model(MODEL_NAME)
    print("Parsing file...")
    sampling_rate, signal = audioBasicIO.read_audio_file(args.source)
    print("Converting file...")
    signal = audioBasicIO.stereo_to_mono(signal)
    print("Analyzing...")
    classes, probabilites = _analyze(signal, sampling_rate, model)
    print("Splitting...")
    split_at = _find_splits(classes, probabilites)
    return _split_file(args.source, split_at)

def download(args) -> None:
    puntata = podcast.find_mp3()
    if not puntata:
        raise ValueError("No puntata")
    if podcast.already_done(puntata, args.outdir):
        return
    r = requests.get(puntata.mp3, allow_redirects=True)
    open("puntata.mp3", "wb").write(r.content)
    namespace = argparse.Namespace()
    setattr(namespace, "source", "puntata.mp3")
    files = split(namespace)
    podcast.make_feed(puntata, files, args.outdir)
    os.unlink("puntata.mp3")


def main():
    parser = argparse.ArgumentParser("ciccio")
    subparsers = parser.add_subparsers(help="Modules:")
    parser_train = subparsers.add_parser("train", help="Train the module")
    parser_train.set_defaults(func=train)
    parser_split = subparsers.add_parser("split", help="Split an mp3")
    parser_split.set_defaults(func=split)
    parser_split.add_argument("source", type=str)
    parser_download = subparsers.add_parser("download", help="Download a new episode and split it")
    parser_download.add_argument("outdir", type=str)
    parser_download.set_defaults(func=download)
    args = parser.parse_args()
    if "func" in args:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
