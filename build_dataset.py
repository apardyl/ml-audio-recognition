import os
import pathlib
import random
import sys
import warnings

import torch
import torchaudio

from config import OUT_FREQ, SAMPLES_PER_TRACK, SAMPLE_LENGTH_S

warnings.simplefilter("ignore")

OUT_PATH = './data/fma_small_samples'

torchaudio.set_audio_backend('sox_io')


def load_track(file_path):
    track, sr = torchaudio.load(file_path, normalize=True, channels_first=True)
    track = track.mean(dim=0, keepdim=True)
    track = torchaudio.transforms.Resample(orig_freq=sr, new_freq=OUT_FREQ).forward(track)
    return track


def gen_samples_for_track(file_path, out_dir):
    source_dir, source_name = os.path.split(file_path)
    dest_dir = os.path.join(out_dir, os.path.basename(source_dir))
    os.makedirs(dest_dir, exist_ok=True)

    track = None

    for sample_idx in range(SAMPLES_PER_TRACK):
        dest_file = os.path.join(dest_dir, "{}.{}.pck".format(source_name, sample_idx))
        if os.path.isfile(dest_file):
            continue

        if track is None:
            try:
                track = load_track(file_path)
            except Exception as ex:
                print("File {} invalid - {}".format(file_path, ex), file=sys.stderr)
                return
        if track.shape[1] < OUT_FREQ * 10:
            print("File {} too short".format(file_path), file=sys.stderr)
            return

        sample_length_points = int(OUT_FREQ * SAMPLE_LENGTH_S)
        idx = random.randint(0, track.shape[1] - sample_length_points - 1)
        f = track[:, idx:idx + sample_length_points + 1].clone().data

        try:
            torch.save(f, dest_file)
        except KeyboardInterrupt as ex:
            os.remove(dest_file) if os.path.exists(dest_file) else None
            raise ex


with torch.no_grad():
    mp3_files = list(str(p) for p in pathlib.Path('./data/fma_small').rglob('*.mp3'))
    last_proc = 0
    for idx, p in enumerate(mp3_files):
        proc = idx * 100 // len(mp3_files)
        if proc > last_proc:
            last_proc = proc
            print('{}%'.format(proc), flush=True)

        gen_samples_for_track(p, OUT_PATH)
