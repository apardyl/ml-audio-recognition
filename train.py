import datetime
import sys
import warnings

import numpy as np
import torch.utils.data
from torchsummary import summary

from config import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE
from dataset import AudioSamplePairDataset
from models import Encoder
from searcher import Searcher


def evaluate_encoder(encoder: Encoder, test_loader: torch.utils.data.DataLoader):
    with torch.no_grad():
        encoder.eval()
        searcher = Searcher.get_simple_index(encoder.small_embedding_dim)
        embeddings_x = []
        embeddings_y = []
        embeddings_x_large = []
        embeddings_y_large = []
        for step, (x, y_pos, _) in enumerate(test_loader):
            x, y_pos = x.cuda(), y_pos.cuda()
            x_enc = encoder(x)
            y_pos_enc = encoder(y_pos)
            embeddings_x.append(x_enc)
            embeddings_y.append(y_pos_enc)
            # embeddings_x_large.append(x_enc_large)
            # embeddings_y_large.append(y_pos_enc_large)
            print('    Test batch {} of {}'.format(step, len(test_loader.dataset) // TEST_BATCH_SIZE), file=sys.stderr)

        embeddings_x = torch.cat(embeddings_x, dim=0)
        embeddings_y = torch.cat(embeddings_y, dim=0)
        # embeddings_x_large = torch.cat(embeddings_x_large, dim=0)
        # embeddings_y_large = torch.cat(embeddings_y_large, dim=0)
        searcher.add(embeddings_x)
        lookup = searcher.search(embeddings_y, 100)
        correct_100 = sum(y in x for y, x in enumerate(lookup[1])) / len(lookup[1])
        correct_50 = sum(y in x[:50] for y, x in enumerate(lookup[1])) / len(lookup[1])
        correct_10 = sum(y in x[:10] for y, x in enumerate(lookup[1])) / len(lookup[1])
        correct_1 = sum(y == x[0] for y, x in enumerate(lookup[1])) / len(lookup[1])

        for idx, res in enumerate(lookup[1]):
            pass

        print(
            'Test accuracy:\n    top1 {}\n    top10 {}\n    top50 {}\n    top100 {}'.format(
                correct_1, correct_10, correct_50, correct_100))


def train_encoder(train_loader, test_loader):
    epochs = 5
    LR = 5e-3  # learning rate

    encoder = Encoder()
    encoder = encoder.cuda()

    summary(model=encoder, input_size=(1, 64, 64), device='cuda')

    optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
    loss_fn = torch.nn.TripletMarginLoss()

    print("Learning started")

    for epoch in range(epochs):
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
        evaluate_encoder(encoder, test_loader)
    return encoder


warnings.simplefilter("ignore")

train_data = AudioSamplePairDataset(root_path='data/fma_small_samples', test=False)
test_data = AudioSamplePairDataset(root_path='data/fma_small_samples', test=True)

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

encoder = train_encoder(train_loader, test_loader)
save_file_path = 'encoder.{0}.pck'.format(datetime.datetime.now().isoformat())
torch.save(encoder.state_dict(), save_file_path)

# encoder = Encoder()
# encoder.load_state_dict(torch.load('encoder.2021-01-16T22:39:26.230503.pck'))
# encoder = encoder.cuda()

# evaluate_encoder(encoder, test_loader)
# verifier = train_verifier(train_loader, test_loader, encoder)
