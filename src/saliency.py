import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

import torch.nn.functional as F  # useful stateless functions

import pickle

import matplotlib.pyplot as plt



import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-m", "--mode", default="random", help="Mode (random/black)")
parser.add_argument("-l", "--loader", default="valset", help="Data loader type (valset/valfullset)")

args = parser.parse_args()


data_dir = 'C:/Users/raffc/Downloads/coco2017'

train_dir = 'train2017_'+args.mode
val_dir = 'val2017_'+args.mode

val_full_dir = 'val2017_processed_images'

class_names = ['broccoli horse'.split()]

model_fname = 'classifier_{}.nn'.format(args.mode)


# pytorch convenience stuff
def flatten(x):
    N = x.shape[0]  # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


# Wrap `flatten` function in a module in order to stack it in nn.Sequential
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensor require gradient
    X.requires_grad_()
    # X.to(device=device)

    scores = model(X)
    loss = torch.nn.functional.cross_entropy(scores, y)

    loss.backward()
    squared = np.abs(X.grad)
    saliency, indices = torch.max(squared, 1)  # 1 is the axis

    return saliency, scores


def check_accuracy(dataset, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in dataset:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


if __name__ == '__main__':
    tensify = T.Compose([
        T.ToTensor()
    ])

    trainset = dset.ImageFolder('{}/{}'.format(data_dir, train_dir), tensify)
    valset = dset.ImageFolder('{}/{}'.format(data_dir, val_dir), tensify)
    valfullset = dset.ImageFolder('{}/{}'.format(data_dir, val_full_dir), tensify)

    # trainset_loader = torch.utils.data.DataLoader(
    #     trainset,
    #     batch_size=64, shuffle=True,
    #     num_workers=4)
    valset_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=5, shuffle=True,
        num_workers=4)
    valfullset_loader = torch.utils.data.DataLoader(
        valfullset,
        batch_size=5, shuffle=True,
        num_workers=4)

    # use the loader that is specified in the initial arguments
    loader = valset_loader
    if args.loader_mode == "valfullset":
        loader = valfullset_loader


    USE_GPU = True

    dtype = torch.float32

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Constant to control how frequently we print train loss
    print_every = 100

    print('using device:', device)

    # model
    model = torch.load(model_fname, 'cuda')

    print(model)

    # limit the number of batches we view
    upto = 5
    curr = 0

    class_names = ['broccoli horse'.split()]

    for X, y in loader:
        if upto <= curr:
            break
        curr += 1

        # Convert X and y from numpy arrays to Torch Tensors
        X_tensor = X.to(device=device)
        y_tensor = torch.LongTensor(y).to(device=device)

        # X = x.view([1] + list(x.shape))
        print(X.shape)

        # Compute saliency maps for images in X
        saliency, scores = compute_saliency_maps(X_tensor, y_tensor, model)
        print(scores)

        # Convert the saliency map from Torch Tensor to numpy array and show images
        # and saliency maps together.
        saliency = saliency.numpy()
        N = X.shape[0]
        for i in range(N):
            plt.subplot(2, N, i + 1)
            plt.imshow(X[i].permute(1, 2, 0).numpy())
            plt.axis('off')
            # plt.title(class_names[y[i]])
            _, idx = scores[i].max(0)
            plt.title('was {}, predicted {}'.format(y[i], idx.data))
            plt.subplot(2, N, N + i + 1)
            plt.imshow(saliency[i], cmap=plt.cm.hot)
            plt.axis('off')
            plt.gcf().set_size_inches(12, 5)
        plt.show()
