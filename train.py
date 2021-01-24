import argparse
import copy
import datetime
import os
import shutil
import sys
import warnings

import numpy as np
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from config import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE
from dataset import AudioSamplePairDataset
from models import SmallEncoder, LargeEncoder
from searcher import Searcher
from utils import load_train_state, save_train_state

SAVED_MODELS_PATH = 'saved_models'
CHECKPOINT_FILE = 'checkpoint.pck'
LOGS_DIR = 'runs'


def evaluate_encoder(encoder: nn.Module, test_loader: torch.utils.data.DataLoader, loss_fn: nn.TripletMarginLoss = None,
                     writer: SummaryWriter = None,
                     epoch: int = -1):
    with torch.no_grad():
        encoder.eval()
        searcher = Searcher.get_simple_index(encoder.embedding_dim)
        embeddings_x = []
        embeddings_y = []
        epoch_losses = []
        for step, (x, y_pos, y_neg) in enumerate(test_loader):
            x, y_pos, y_neg = x.cuda(), y_pos.cuda(), y_neg.cuda()
            x_enc = encoder(x)
            y_pos_enc = encoder(y_pos)
            y_neg_enc = encoder(y_neg)
            if loss_fn:
                loss_val = loss_fn(x_enc, y_pos_enc, y_neg_enc)
                epoch_losses.append(loss_val.item())
            embeddings_x.append(x_enc)
            embeddings_y.append(y_pos_enc)
            print('    Test batch {} of {}'.format(step + 1, len(test_loader)), file=sys.stderr)

        embeddings_x = torch.cat(embeddings_x, dim=0)
        embeddings_y = torch.cat(embeddings_y, dim=0)
        searcher.add(embeddings_x)
        lookup = searcher.search(embeddings_y, 100)
        correct_100 = sum(y in x for y, x in enumerate(lookup[1])) / len(lookup[1])
        correct_50 = sum(y in x[:50] for y, x in enumerate(lookup[1])) / len(lookup[1])
        correct_10 = sum(y in x[:10] for y, x in enumerate(lookup[1])) / len(lookup[1])
        correct_1 = sum(y == x[0] for y, x in enumerate(lookup[1])) / len(lookup[1])
        print(f'Test loss: {np.mean(epoch_losses):.4f}')
        print(
            'Test accuracy:\n    top1 {}\n    top10 {}\n    top50 {}\n    top100 {}'.format(
                correct_1, correct_10, correct_50, correct_100))
        if writer:
            writer.add_scalars('Accuracy', {
                'top1': correct_1,
                'top10': correct_10,
                'top50': correct_50,
                'top100': correct_100,
            }, global_step=epoch)
            writer.add_scalar('Loss/test', np.mean(epoch_losses), global_step=epoch)
            if epoch == -1 or epoch % 5 == 1:
                mat = torch.cat([embeddings_x[:1000], embeddings_y[:1000]], dim=0)
                labels = list(range(1000)) + list(range(1000))
                writer.add_embedding(mat, labels, tag='Embeddings', global_step=epoch)
        return correct_1 * 100, correct_100 * 100, lookup, embeddings_x, embeddings_y


def train_encoder(train_loader, test_loader, encoder, epochs=30):
    LR = 0.01

    summary(model=encoder, input_size=next(iter(train_loader))[0].shape[1:], device='cuda')

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    epoch = 0
    best_model = copy.deepcopy(encoder)
    best_score = -1

    if os.path.exists(CHECKPOINT_FILE):
        print('Loading checkpoint')
        epoch, best_score = load_train_state(CHECKPOINT_FILE, encoder, optimizer, scheduler)

    log_file = os.path.join(LOGS_DIR, encoder.__class__.__name__)
    writer = SummaryWriter(log_file, purge_step=epoch, flush_secs=60)

    print("Learning started")

    while epoch < epochs:
        epoch += 1
        print(f"Epoch: {epoch}")
        epoch_losses = []
        encoder.train()
        margin = np.sqrt(encoder.embedding_dim) / 4
        loss_fn = torch.nn.TripletMarginLoss(margin=margin)
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
            print('    Batch {} of {} loss: {}, lr: {}'.format(step + 1, len(train_loader),
                                                               loss_val.item(), optimizer.param_groups[0]["lr"]),
                  file=sys.stderr)
        print(f'Train loss: {np.mean(epoch_losses):.4f}')
        writer.add_scalars('Loss', {
            'train': np.mean(epoch_losses),
            'lr': optimizer.param_groups[0]["lr"]
        }, global_step=epoch)
        score = evaluate_encoder(encoder, test_loader, loss_fn, writer=writer, epoch=epoch)[0]
        if score > best_score:
            best_model = copy.deepcopy(encoder)
            best_score = score
            print('New best score')
            save_train_state(epoch, encoder, optimizer, scheduler, best_score, CHECKPOINT_FILE)
        scheduler.step()
    if best_score < 0:
        best_score = evaluate_encoder(encoder, test_loader, writer=writer)[0]

    writer.close()
    save_file_path = os.path.join(SAVED_MODELS_PATH, '{}.{}.{:.2f}.pck'.format(encoder.__class__.__name__,
                                                                               datetime.datetime.now().isoformat(),
                                                                               best_score))
    log_file_path = os.path.join(LOGS_DIR, '{}.{}.{:.2f}.pck'.format(encoder.__class__.__name__,
                                                                     datetime.datetime.now().isoformat(),
                                                                     best_score))
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    shutil.move(CHECKPOINT_FILE, save_file_path)
    shutil.move(log_file, log_file_path)

    return best_model, best_score


def train(large_model=False, epochs=30):
    print('Training {} encoder for {} epochs'.format('large' if large_model else 'small', epochs))
    train_data = AudioSamplePairDataset(root_path='data/fma_small_samples', test=False, large=large_model)
    test_data = AudioSamplePairDataset(root_path='data/fma_small_samples', test=True, large=large_model)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=TRAIN_BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=12,
                                               pin_memory=True,
                                               prefetch_factor=4)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=TEST_BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=8,
                                              pin_memory=True,
                                              prefetch_factor=4)
    encoder = LargeEncoder() if large_model else SmallEncoder()
    encoder = encoder.cuda()
    train_encoder(train_loader, test_loader, encoder, epochs)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='Train models.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, choices=['small', 'large', 'both'], default='both',
                        help='Which model to train')
    parser.add_argument('--epochs', type=int, default=30, help='How many epochs to train for')
    args = parser.parse_args()

    if args.model in ['small', 'both']:
        train(False, args.epochs)
    if args.model in ['large', 'both']:
        train(True, args.epochs)
