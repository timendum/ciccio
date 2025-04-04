# flake8: noqa: E402
import argparse
import json
import logging
import logging.config
import os
import subprocess
import sys
from itertools import chain
from math import ceil

from pydub.utils import mediainfo_json

import podcast

MODEL_NAME = "data/svmSM"

# The input mp3 file for the `download` command will be splitted in smaller chunks,
# so the memory footprint is smaller.
# This will determine the duration (in seconds) of chunks.
TIME_SPLIT = 60 * 10


def _get_logger(logger_name=__name__) -> logging.Logger:
    try:
        with open("logging.json", encoding="utf-8") as logconfigf:
            logging.config.dictConfig(json.load(logconfigf))
        logger = logging.getLogger(logger_name)
    except OSError:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        consoleh = logging.StreamHandler(sys.stdout)
        consoleh.set_name("console")
        consoleh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        )
        logger.addHandler(consoleh)
    if "-q" in sys.argv:
        for handler in logging.getLogger(logger_name).handlers:
            if handler.get_name() == "console":
                handler.setLevel(logging.ERROR)
    elif "-v" in sys.argv:
        for handler in logging.getLogger(logger_name).handlers:
            if handler.get_name() == "console":
                handler.setLevel(logging.DEBUG)
    return logger


LOGGER = _get_logger()
LOGGER.debug("Starting...")

# Slow imports
import numpy as np
import pyfftw
import requests
import scipy

from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioTrainTest as aT


def train(_):
    return aT.extract_features_and_train(
        ["data/ko", "data/ok"],
        1,
        1,
        aT.shortTermWindow,
        aT.shortTermStep,
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
    input_name = classifier.get_inputs()[0].name
    label_name = classifier.get_outputs()[0].name
    prob_name = classifier.get_outputs()[1].name
    with scipy.fft.set_backend(pyfftw.interfaces.scipy_fft):
        # Turn on the cache for optimum performance
        pyfftw.interfaces.cache.enable()
        for i in range(0, len(signal), aF.STEP_SIZE * sampling_rate):
            subsignal = signal[i : i + (aF.STEP_SIZE * sampling_rate)]
            if len(subsignal) < round(sampling_rate * short_step):
                LOGGER.debug(
                    "Remaining signal is too short: from %d to %d",
                    i / sampling_rate,
                    i / sampling_rate + aF.STEP_SIZE,
                )
                # keep previous
                classes.append(classes[-1])
                probabilites.append(probabilites[-1])
                continue
            # feature extraction:
            mid_features, _ = aF.mid_feature_extraction(
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
            # class_id = classifier.predict(feature_vector.reshape(1, -1))[0]
            # probability = classifier.predict_proba(feature_vector.reshape(1, -1))[0]
            cout = classifier.run(
                [label_name, prob_name],
                {input_name: feature_vector.reshape(1, -1).astype(np.float32)},
            )
            classes.append(cout[0][0])
            probabilites.append(list(cout[1][0].values()))
            # LOGGER.debug("Step: %d/%d", i / sampling_rate, len(signal) / sampling_rate)
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


def _split_file(source, split_at, fade=False) -> list[str]:
    """ "
    Split add:
    - from split_at[0] to split_at[1]
    - from split_at[2] to split_at[3]
    Chunks in the middle (ie between 1 and 2) will be skipped.
    """
    LOGGER.debug("Splitting %s at %s", source, split_at)
    target = source.replace(".mp3", "")
    filenames = []
    delta = 0
    if fade:
        delta = 3
    for i in range(0, len(split_at), 2):
        split_command = ["ffmpeg", "-y", "-i", source, "-ss", str(split_at[i] - delta)]
        try:
            split_command.extend(["-to", str(split_at[i + 1] + delta)])
        except IndexError:
            # end of file
            pass
        filename = f"{target}_{int(i/2)+1:02}.mp3"
        if fade:
            try:
                split_command.extend(
                    [
                        "-af",
                        f"afade=t=in:st={split_at[i] - delta}:d={delta}:curve=qsin,"
                        f"afade=t=out:st={split_at[i + 1] + delta}:d={delta}:curve=qsin",
                    ]
                )
            except IndexError:
                # end of file
                split_command.extend(
                    ["-af", f"afade=t=in:st={split_at[i] - delta}:d={delta}:curve=qsin"]
                )
        else:
            split_command.extend(["-acodec", "copy"])
        split_command.append(filename)
        subprocess.run(split_command, capture_output=True)
        filenames.append(filename)
    return filenames


def split(args) -> list[str]:
    LOGGER.info("Loading Model...")
    model = aT.load_model(MODEL_NAME)
    LOGGER.info("Parsing file...")
    sampling_rate, signal = audioBasicIO.read_audio_file(args.source)
    LOGGER.info("Converting file...")
    signal = audioBasicIO.stereo_to_mono(signal)
    LOGGER.info("Analyzing...")
    classes, probabilites = _analyze(signal, sampling_rate, model)
    LOGGER.info("Splitting...")
    split_at = _find_splits(classes, probabilites)
    LOGGER.info("Generating podcast files")
    return _split_file(args.source, split_at, True)


def download(args) -> None:
    puntata = podcast.find_mp3(args.url)
    if not puntata:
        raise ValueError("No puntata")
    if podcast.already_done(puntata, args.outdir):
        return
    r = requests.get(puntata.mp3, allow_redirects=True)
    r.raise_for_status()
    namespace = argparse.Namespace()
    setattr(namespace, "source", "puntata.mp3")
    open(namespace.source, "wb").write(r.content)
    mediainfo = mediainfo_json(namespace.source)
    LOGGER.info("Loading Model...")
    model = aT.load_model(MODEL_NAME)
    duration = ceil(float(mediainfo["streams"][0]["duration"]))
    # Split at every TIME_SPLIT, until the end of file (the end here is bigger then duration)
    ranges = range(0, (ceil(duration / TIME_SPLIT) + 1) * TIME_SPLIT, TIME_SPLIT)
    # join to [range0, range1, range1, range2, range2, range3, ...]
    files = _split_file(
        namespace.source, list(chain.from_iterable(zip(ranges, ranges[1:])))
    )
    fclasses, fprobabilites = [], []
    for (
        i,
        file,
    ) in enumerate(files):
        LOGGER.info("Parsing file... %d/%d", i + 1, len(files))
        sampling_rate, signal = audioBasicIO.read_audio_file(file)
        LOGGER.info("Converting file... %d/%d", i + 1, len(files))
        signal = audioBasicIO.stereo_to_mono(signal)
        LOGGER.info("Analyzing... %d/%d", i + 1, len(files))
        classes, probabilites = _analyze(signal, sampling_rate, model)
        fclasses.extend(classes)
        fprobabilites.extend(probabilites)
        os.unlink(file)
    LOGGER.info("Splitting...")
    split_at = _find_splits(fclasses, fprobabilites)
    files = _split_file(namespace.source, split_at)
    podcast.make_feed(puntata, files, args.outdir)
    podcast.make_site(puntata, files, args.outdir)
    os.unlink(namespace.source)


def main():
    parser = argparse.ArgumentParser("ciccio")
    subparsers = parser.add_subparsers(help="Modules:")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-q", "--quiet", action="store_true")
    group.add_argument("-v", "--verbose", action="store_true")
    parser_train = subparsers.add_parser("train", help="Train the module")
    parser_train.set_defaults(func=train)
    parser_split = subparsers.add_parser("split", help="Split an mp3")
    parser_split.set_defaults(func=split)
    parser_split.add_argument("source", type=str)
    parser_download = subparsers.add_parser(
        "download", help="Download a new episode and split it"
    )
    parser_download.add_argument("outdir", type=str)
    parser_download.add_argument("-url", type=str)
    parser_download.set_defaults(func=download)
    args = parser.parse_args()
    if "func" in args:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
