import os
import pathlib
import random
import sys
import warnings

import librosa
import numpy as np
import skimage.io
from audioread import NoBackendError
from sklearn.preprocessing import minmax_scale

warnings.simplefilter("ignore")

IMAGE_X = 150
IMAGE_Y = 128
SAMPLE_LENGTH_S = 2.
SAMPLES_PER_TRACK = 10

OUT_PATH = './data/fma_small_images128'


def load_track(file_path):
    track, sr = librosa.load(file_path, mono=True, sr=None)
    sample_length_points = int(sr * SAMPLE_LENGTH_S)
    return track, sr, sample_length_points


def gen_samples_for_track(file_path, out_dir):
    source_dir, source_name = os.path.split(file_path)
    dest_dir = os.path.join(out_dir, os.path.basename(source_dir))
    os.makedirs(dest_dir, exist_ok=True)

    track, sr, sample_length_points = None, None, None

    for sample_idx in range(SAMPLES_PER_TRACK):
        dest_file = os.path.join(dest_dir, "{}.{}.png".format(source_name, sample_idx))
        if os.path.isfile(dest_file):
            continue

        if track is None:
            try:
                track, sr, sample_length_points = load_track(file_path)
            except NoBackendError:
                print("File {} invalid".format(file_path), file=sys.stderr)
                return
        if len(track) < sample_length_points:
            print("File {} too short".format(file_path), file=sys.stderr)
            return

        idx = random.randint(0, len(track) - sample_length_points - 1)
        f = track[idx:idx + sample_length_points + 1]
        f = librosa.util.normalize(f)
        hop = sample_length_points // IMAGE_X
        spec = librosa.power_to_db(
            librosa.feature.melspectrogram(y=f, sr=sr, hop_length=hop, n_mels=IMAGE_Y)[:, :IMAGE_X],
            ref=np.max)
        img = minmax_scale(spec, feature_range=(0, 255)).astype(np.uint8)
        img = np.flip(img, axis=0)
        try:
            skimage.io.imsave(fname=dest_file, arr=img)
        except KeyboardInterrupt as ex:
            os.remove(dest_file) if os.path.exists(dest_file) else None
            raise ex


mp3_files = list(str(p) for p in pathlib.Path('./data/fma_small').rglob('*.mp3'))
last_proc = 0
for idx, p in enumerate(mp3_files):
    proc = idx * 100 // len(mp3_files)
    if proc > last_proc:
        last_proc = proc
        print('{}%'.format(proc))
    gen_samples_for_track(p, OUT_PATH)
