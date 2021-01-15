import matplotlib.pyplot as plt
import torch.utils.data
import numpy as np
from PIL import Image

from sklearn.decomposition import PCA


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


def plot_dataset(train_data, model):
    view_data = torch.stack([train_data[i][0] for i in range(5)])
    _, decoded_data = model.forward(view_data.cuda())
    decoded_data = decoded_data.cpu().detach().numpy()

    n_rows = 2 if decoded_data is not None else 1
    n_cols = len(view_data)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    if decoded_data is not None:
        for i in range(n_cols):
            axes[0][i].imshow(view_data.squeeze(1).data.numpy()[i])
            axes[0][i].set_xticks(())
            axes[0][i].set_yticks(())

        for i in range(n_cols):
            axes[1][i].clear()
            axes[1][i].imshow(decoded_data.squeeze(1)[i])
            axes[1][i].set_xticks(())
            axes[1][i].set_yticks(())

    else:
        for i in range(n_cols):
            axes[i].imshow(view_data.squeeze(1).data.numpy()[i])
            axes[i].set_xticks(())
            axes[i].set_yticks(())

    plt.show()


#
#
# def plot_pca(data, model):
#     labels = data.classes
#     plt.suptitle("Reduction of latent space")
#     _ = plt.figure(figsize=(10, 6))
#     pca = PCA(2)
#
#     z = model.encode(train_data.data.view(-1, 784).float().cuda())
#     reduced_z = pca.fit_transform(z.detach().cpu().numpy())
#
#     for class_idx in range(10):
#         indices = (data.targets == class_idx)
#         plt.scatter(
#             reduced_z[indices, 0], reduced_z[indices, 1],
#             s=2., label=labels[class_idx])
#
#     plt.legend()
#     plt.show()
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, CenterCrop

from autoencoder import AutoEncoder

transforms = Compose([CenterCrop(size=224), ToTensor()])

train_data = ImageFolder('./data/fma_small_images', transform=transforms, loader=pil_loader)

train_data = torch.utils.data.Subset(train_data, np.random.choice(len(train_data), 1000))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)

epochs = 25
LR = 5e-3  # learning rate

auto_encoder = AutoEncoder().cuda()

optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=LR)
rec_loss_fn = torch.nn.MSELoss()

print("Learning started")

for epoch in range(epochs):
    epoch_losses = []  # For logging purposes
    for step, (x, y) in enumerate(train_loader):
        x = x.cuda()
        encoded, decoded = auto_encoder(x)
        loss_val = rec_loss_fn(decoded, x)  # calculate loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss_val.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        epoch_losses.append(loss_val.item())
        print('    Batch loss', loss_val.item())

    print(f'Epoch: {epoch}  |  train loss: {np.mean(epoch_losses):.4f}')
    plot_dataset(train_data, auto_encoder)
