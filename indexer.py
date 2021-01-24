import argparse
import contextlib
import os
import sqlite3
import warnings

import torch

from dataset import AudioIndexingDataset
from models import SmallEncoder, LargeEncoder
from searcher import Searcher
from utils import load_model_state

if __name__ == "__main__":
    warnings.simplefilter("ignore")
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='Index audio files.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, help='Path to audio files to index', required=True)
    parser.add_argument('--small_encoder', type=str, help='Path to saved SmallEncoder model', required=True)
    parser.add_argument('--large_encoder', type=str, help='Path to saved LargeEncoder model', required=True)
    parser.add_argument('--database', type=str, help='Track database save location', required=True)
    parser.add_argument('--index', type=str, help='Track lookup index save location', required=True)
    args = parser.parse_args()

    with contextlib.suppress(FileNotFoundError):
        os.remove(args.database)
    with contextlib.suppress(FileNotFoundError):
        os.remove(args.index)

    database = sqlite3.connect(args.database)
    database.execute(
        """create table samples (id integer primary key, name text, offset integer, s_hash blob, l_hash blob);""")
    searcher = Searcher.get_simple_index(SmallEncoder.embedding_dim)

    data = AudioIndexingDataset(args.data)
    data_loader = torch.utils.data.DataLoader(data, batch_size=None, num_workers=4, prefetch_factor=2)
    small_encoder = SmallEncoder().cuda()
    load_model_state(args.small_encoder, small_encoder)
    large_encoder = LargeEncoder().cuda()
    load_model_state(args.large_encoder, large_encoder)
    large_encoder.eval()
    small_encoder.eval()
    counter = 0

    embeddings = []

    with torch.no_grad():
        for step, val in enumerate(data_loader):
            if val is None:
                continue
            (file_name, samples_s, samples_l) = val
            print('Indexing {}, {} of {}'.format(file_name, step + 1, len(data_loader)))
            hashes_s = small_encoder(samples_s.cuda()).cpu()
            hashes_l = large_encoder(samples_l.cuda()).cpu()
            embeddings.append(hashes_s)
            rows = [(counter + idx, file_name, idx, h_s.tobytes(), h_l.tobytes()) for idx, (h_s, h_l) in
                    enumerate(zip(hashes_s.numpy(), hashes_l.numpy()))]
            counter += len(rows)
            database.executemany('insert into samples values (?,?,?,?,?);', rows)
    database.commit()
    database.close()
    embeddings = torch.cat(embeddings, dim=0)
    searcher.add(embeddings)
    searcher.save(args.index)
