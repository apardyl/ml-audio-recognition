import copy
import datetime
import os
import shutil
import sys
import warnings

import numpy as np
import torch.utils.data
from torch import nn
from torchsummary import summary

from config import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE
from dataset import AudioSamplePairDataset
from models import SmallEncoder, LargeEncoder
from searcher import Searcher

SAVED_MODELS_PATH = 'saved_models'
CHECKPOINT_FILE = 'checkpoint.pck'


def evaluate_encoder(encoder: nn.Module, test_loader: torch.utils.data.DataLoader):
    with torch.no_grad():
        encoder.eval()
        searcher = Searcher.get_simple_index(encoder.embedding_dim)
        embeddings_x = []
        embeddings_y = []
        for step, (x, y_pos, _) in enumerate(test_loader):
            x, y_pos = x.cuda(), y_pos.cuda()
            x_enc = encoder(x)
            y_pos_enc = encoder(y_pos)
            embeddings_x.append(x_enc)
            embeddings_y.append(y_pos_enc)
            print('    Test batch {} of {}'.format(step, len(test_loader.dataset) // TEST_BATCH_SIZE), file=sys.stderr)

        embeddings_x = torch.cat(embeddings_x, dim=0)
        embeddings_y = torch.cat(embeddings_y, dim=0)
        searcher.add(embeddings_x)
        lookup = searcher.search(embeddings_y, 100)
        correct_100 = sum(y in x for y, x in enumerate(lookup[1])) / len(lookup[1])
        correct_50 = sum(y in x[:50] for y, x in enumerate(lookup[1])) / len(lookup[1])
        correct_10 = sum(y in x[:10] for y, x in enumerate(lookup[1])) / len(lookup[1])
        correct_1 = sum(y == x[0] for y, x in enumerate(lookup[1])) / len(lookup[1])
        print(
            'Test accuracy:\n    top1 {}\n    top10 {}\n    top50 {}\n    top100 {}'.format(
                correct_1, correct_10, correct_50, correct_100))
        return correct_1 * 100, correct_100 * 100, lookup, embeddings_x, embeddings_y


def evaluate_all(small_encoder: SmallEncoder, large_encoder: LargeEncoder, test_loader: torch.utils.data.DataLoader):
    _, lookup_score, lookup, embeddings_x, embeddings_y = evaluate_encoder(small_encoder, test_loader)


def save_train_state(epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, file_path):
    torch.save({
        'epoch': epoch,
        'model_state': encoder.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, file_path)


def load_train_state(file_path, model: nn.Module, optimizer: torch.optim.Optimizer):
    data = torch.load(file_path)
    model.load_state_dict(data['model_state'])
    optimizer.load_state_dict(data['optimizer_state'])
    return data['epoch']


def train_encoder(train_loader, test_loader, encoder):
    epochs = 20
    LR = 2e-3  # learning rate

    summary(model=encoder, input_size=next(iter(train_loader))[0].shape[1:], device='cuda')

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)
    loss_fn = torch.nn.TripletMarginLoss()
    epoch = 0

    if os.path.exists(CHECKPOINT_FILE):
        print('Loading checkpoint')
        epoch = load_train_state(CHECKPOINT_FILE, encoder, optimizer)
    print("Learning started")
    best_model = None
    best_score = -1

    while epoch < epochs:
        epoch += 1
        print(f"Epoch: {epoch}")
        epoch_losses = []
        encoder.train()
        for step, (x, y_pos, y_neg) in enumerate(train_loader):
            x, y_pos, y_neg = x.cuda(), y_pos.cuda(), y_neg.cuda()
            x_enc = encoder(x)
            y_pos_enc = encoder(y_pos)
            y_neg_enc = encoder(y_neg)
            loss_val = loss_fn(x_enc, y_pos_enc, y_neg_enc)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            epoch_losses.append(loss_val.item())
            print('    Batch {} of {} loss: {}'.format(step, len(train_loader.dataset) // TRAIN_BATCH_SIZE,
                                                       loss_val.item()),
                  file=sys.stderr)
        print(f'Train loss: {np.mean(epoch_losses):.4f}')
        score = evaluate_encoder(encoder, test_loader)[0]
        if score > best_score:
            best_model = copy.deepcopy(encoder)
            best_score = score
            print('New best score')
            save_train_state(epoch, encoder, optimizer, CHECKPOINT_FILE)
    save_file_path = os.path.join(SAVED_MODELS_PATH, '{}.{}.{:.2f}.pck'.format(encoder.__class__.__name__,
                                                                               datetime.datetime.now().isoformat(),
                                                                               best_score))
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    shutil.move(CHECKPOINT_FILE, save_file_path)

    return best_model, best_score


warnings.simplefilter("ignore")

train_data = AudioSamplePairDataset(root_path='data/fma_small_samples', test=False, large=True)
test_data = AudioSamplePairDataset(root_path='data/fma_small_samples', test=True, large=True)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=TRAIN_BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=8,
                                           pin_memory=True,
                                           prefetch_factor=4)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=TEST_BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=8,
                                          pin_memory=True,
                                          prefetch_factor=4)

encoder = SmallEncoder()
encoder = encoder.cuda()
encoder, accu = train_encoder(train_loader, test_loader, encoder)

# encoder = SmallEncoder()
# encoder.load_state_dict(torch.load('encoder.large.2021-01-18T01:19:18.356097.pck'))
# encoder = encoder.cuda()

# evaluate_encoder(encoder, test_loader)
# verifier = train_verifier(train_loader, test_loader, encoder)
