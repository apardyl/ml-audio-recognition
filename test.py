import argparse
import sys
import warnings
from operator import itemgetter

import torch

from config import TEST_BATCH_SIZE
from dataset import AudioSamplePairDualDataset
from models import SmallEncoder, LargeEncoder
from searcher import Searcher
from utils import load_model_state


def evaluate_all(small_encoder: SmallEncoder, large_encoder: LargeEncoder, test_loader: torch.utils.data.DataLoader,
                 lookup_samples: int):
    with torch.no_grad():
        small_encoder.eval()
        large_encoder.eval()
        searcher = Searcher.get_simple_index(small_encoder.embedding_dim)
        s_embeddings_x = []
        s_embeddings_y = []
        l_embeddings_x = []
        l_embeddings_y = []
        print('Calculating embeddings')
        for step, (x_s, y_s, x_l, y_l) in enumerate(test_loader):
            s_embeddings_x.append(small_encoder(x_s.cuda()))
            s_embeddings_y.append(small_encoder(y_s.cuda()))
            l_embeddings_x.append(large_encoder(x_l.cuda()))
            l_embeddings_y.append(large_encoder(y_l.cuda()))
            print('    Test batch {} of {}'.format(step + 1, len(test_loader)), file=sys.stderr)
        print('Merging results')
        s_embeddings_x = torch.cat(s_embeddings_x, dim=0).cpu()
        s_embeddings_y = torch.cat(s_embeddings_y, dim=0).cpu()
        l_embeddings_x = torch.cat(l_embeddings_x, dim=0).cpu()
        l_embeddings_y = torch.cat(l_embeddings_y, dim=0).cpu()
        print('Running kNN')
        searcher.add(s_embeddings_x)
        lookup = searcher.search(s_embeddings_y, lookup_samples)
        correct_100 = sum(y in x for y, x in enumerate(lookup[1]))
        correct_1 = sum(y == x[0] for y, x in enumerate(lookup[1]))
        print('Running verification')
        verified_1 = 0
        s_embeddings_x = s_embeddings_x.numpy()
        s_embeddings_y = s_embeddings_y.numpy()
        l_embeddings_x = l_embeddings_x.numpy()
        l_embeddings_y = l_embeddings_y.numpy()
        for idx, (knn, y_s, y_l) in enumerate(zip(lookup[1], s_embeddings_y, l_embeddings_y)):
            dists = [((((s_embeddings_x[v] - y_s) ** 2).mean() + ((l_embeddings_x[v] - y_l) ** 2).mean()), v)
                     for v in knn]
            best = min(dists, key=itemgetter(0))[1]
            if best == idx:
                verified_1 += 1
        print('Lookup accuracy: {}, correct guess: {}'.format(correct_100 / len(lookup[1]), correct_1 / len(lookup[1])))
        print('Verification accuracy: {}'.format(verified_1 / correct_100))
        print('Final accuracy: {}'.format(verified_1 / len(lookup[1])))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='Test models.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--small_encoder', type=str, help='Path to saved SmallEncoder model', required=True)
    parser.add_argument('--large_encoder', type=str, help='Path to saved LargeEncoder model', required=True)
    parser.add_argument('--lookup_samples', type=int, help='Number of samples to lookup in kNN', default=100)
    args = parser.parse_args()

    small_encoder = SmallEncoder().cuda()
    load_model_state(args.small_encoder, small_encoder)
    large_encoder = LargeEncoder().cuda()
    load_model_state(args.large_encoder, large_encoder)

    test_data = AudioSamplePairDualDataset(root_path='data/fma_small_samples', test=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=TEST_BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=8,
                                              pin_memory=True,
                                              prefetch_factor=4)

    evaluate_all(small_encoder, large_encoder, test_loader, args.lookup_samples)
