import argparse
import pathlib
import random
import sqlite3
import sys
import warnings
from operator import itemgetter
from typing import List, Tuple, Dict

import torch
import numpy as np
import torchaudio

from config import OUT_FREQ
from models import SmallEncoder, LargeEncoder
from searcher import Searcher
from utils import transform_melspectrogram, load_model_state, transform_distort_audio

torchaudio.set_audio_backend('sox_io')


def get_candidates(samples: torch.Tensor, small_encoder: SmallEncoder, large_encoder: LargeEncoder, searcher: Searcher,
                   database: sqlite3.Connection) -> Tuple[
    List[List[Tuple[float, int]]], Dict[int, Tuple[str, int, np.ndarray, np.ndarray]]]:
    samples_s = transform_melspectrogram(samples, large=False).cuda()
    samples_l = transform_melspectrogram(samples, large=True).cuda()

    enc_s = small_encoder(samples_s).cpu()
    enc_l = large_encoder(samples_l).cpu()

    lookup = searcher.search(enc_s, k=100)[1]
    candidates_ids = np.unique(lookup)
    candidates = database.execute(
        'select id, name, offset, s_hash, l_hash from samples where id in ({});'.format(
            ','.join(str(x) for x in candidates_ids))).fetchall()
    candidates = {
        idx: (title, offset, np.frombuffer(hash_s, dtype='float32'), np.frombuffer(hash_l, dtype='float32'))
        for idx, title, offset, hash_s, hash_l in candidates
    }

    best_matches = []
    for knn, e_l in zip(lookup, enc_l.numpy()):
        dists_l = [(((candidates[v][3] - e_l) ** 2).mean(), v)
                   for v in knn]
        dists_l = sorted(dists_l, key=itemgetter(0))
        best_matches.append(dists_l)
    return best_matches, candidates


def process_offset(matches: List[List[Tuple[float, int]]],
                   candidates: Dict[int, Tuple[str, int, np.ndarray, np.ndarray]]):
    # TODO: implement multi-sample flow.
    best_match = matches[0][0]
    return candidates[best_match[1]], best_match[0]


def recognize(track: torch.Tensor, small_encoder: SmallEncoder, large_encoder: LargeEncoder, searcher: Searcher,
              database: sqlite3.Connection):
    sample_length_points = int(OUT_FREQ * 1)
    track_len = track.shape[1]
    offsets = [int(x * sample_length_points) for x in [0, 0.2, 0.4, 0.6, 0.8]]
    # num_samples = int((track_len - sample_length_points * 0.8) // sample_length_points)
    num_samples = 1  # TODO: implement multi-sample flow.
    samples = []
    for offset in offsets:
        row = []
        for idx in range(num_samples):
            start_idx = idx * sample_length_points + offset
            sample = track[:, start_idx:start_idx + sample_length_points]
            row.append(sample)
        row = torch.stack(row, dim=0)
        samples.append(row)
    samples = torch.cat(samples, dim=0)

    best_matches, candidates = get_candidates(samples, small_encoder, large_encoder, searcher, database)
    best_per_offset = []
    for off_idx in range(len(offsets)):
        val = process_offset(best_matches[off_idx * num_samples:off_idx * num_samples + num_samples], candidates)
        best_per_offset.append(val)

    best = min(best_per_offset, key=itemgetter(1))
    return best


def interactive(small_encoder: SmallEncoder, large_encoder: LargeEncoder, searcher: Searcher,
                database: sqlite3.Connection, augment: bool):
    print('Input format: [filename] [sample start offset in seconds (float)] [sample length in seconds (float)]')
    for line in sys.stdin:
        line = line.split()
        if len(line) != 3:
            print('Invalid input')
        try:
            file_path = line[0]
            offset = float(line[1])
            length = float(line[2])
            length = max(length, 2)
        except Exception as ex:
            print('Invalid input', ex)
            continue
        try:
            track, sr = torchaudio.load(file_path, normalize=True, channels_first=True)
            track = track.mean(dim=0, keepdim=True)
            track = torchaudio.transforms.Resample(orig_freq=sr, new_freq=OUT_FREQ).forward(track)
            track = track[:, int(offset * OUT_FREQ):int(offset * OUT_FREQ + length * OUT_FREQ)]
            if track.shape[1] < OUT_FREQ:
                print('Invalid sample length or offset')
                continue
        except Exception as ex:
            print('Unable to load sample', ex)
            continue
        with torch.no_grad():
            if augment:
                track = transform_distort_audio(track)
            y = recognize(track, small_encoder, large_encoder, searcher, database)
            print('Prediction:', y[0][0], 'distance: ', y[1])


def process_dir(small_encoder: SmallEncoder, large_encoder: LargeEncoder, searcher: Searcher,
                database: sqlite3.Connection, augment: bool, sample_dir: str):
    dataset = list(str(p) for p in pathlib.Path(sample_dir).rglob('*.mp3'))
    for file_path in dataset:
        try:
            track, sr = torchaudio.load(file_path, normalize=True, channels_first=True)
            track = track.mean(dim=0, keepdim=True)
            track = torchaudio.transforms.Resample(orig_freq=sr, new_freq=OUT_FREQ).forward(track)
        except Exception as ex:
            print("File {} invalid - {}".format(file_path, ex), file=sys.stderr)
            continue
        if track.shape[1] < OUT_FREQ * 10:
            print("File {} too short".format(file_path), file=sys.stderr)
            continue

        offset = random.randint(0, track.shape[1] - 2 * OUT_FREQ)
        track = track[:, offset:offset + 4 * OUT_FREQ]
        with torch.no_grad():
            if augment:
                track = transform_distort_audio(track)
            y = recognize(track, small_encoder, large_encoder, searcher, database)
            print(
                'Prediction for {} ({} - {}): {} distance: {}'.format(file_path, offset, offset + 4 * OUT_FREQ, y[0][0],
                                                                      y[1]))


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    help_text = '''This tool works either in an interactive mode (audio sample paths, offsets and lengths provided via\n
stdin) or can be pointed to a directory containing mp3 files (random samples will be taken). Output the name of the\n
closest matching track in the database (and similarity score - the lower the better).'''

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='Recognize audio files.\n' + help_text,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--small_encoder', type=str, help='Path to saved SmallEncoder model', required=True)
    parser.add_argument('--large_encoder', type=str, help='Path to saved LargeEncoder model', required=True)
    parser.add_argument('--database', type=str, help='Path to track database', required=True)
    parser.add_argument('--index', type=str, help='Path to track lookup index', required=True)
    parser.add_argument('--interactive', help='Enable interactive mode', action='store_true')
    parser.add_argument('--augment', help='Enable input augmentation', action='store_true')
    parser.add_argument('--sample_dir', type=str, help='Path to samples to recongize')
    args = parser.parse_args()

    if args.interactive and args.sample_dir:
        print('Incompatible options. Please specify either --interactive or --sample_dir, not both at the same time')
        exit(-1)

    if not args.interactive and not args.sample_dir:
        print('No action. Please specify either --interactive or --sample_dir')
        exit(-1)

    database = sqlite3.connect(args.database)
    searcher = Searcher.load(args.index)

    small_encoder = SmallEncoder().cuda()
    load_model_state(args.small_encoder, small_encoder)
    large_encoder = LargeEncoder().cuda()
    load_model_state(args.large_encoder, large_encoder)
    large_encoder.eval()
    small_encoder.eval()

    if args.interactive:
        interactive(small_encoder, large_encoder, searcher, database, args.augment)
        exit(0)

    if args.sample_dir:
        process_dir(small_encoder, large_encoder, searcher, database, args.augment, args.sample_dir)
        exit(0)
