from pytorch_helper_functions import *
from local_paths import *
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

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument("-m", "--mode", default="random", help="Mode (random/black/normal)")
parser.add_argument("-u", "--unusual", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Validate on unusual set only")

args = parser.parse_args()


val_dir = 'val2017_'+args.mode
if args.mode == 'normal':
    val_dir = 'val2017_processed_images'

val_full_dir = 'val2017_processed_images'
unusual_dir = 'val2017_unusual'


# model_fname = 'classifier_{}.nn'.format(args.mode)
# model_fname = '../models/model_random_2018-12-06--14-03-22.nn'.format(args.mode)
# model_fname = '../models/model_random_2018-12-06--16-33-55.nn'.format(args.mode)
model_fname = latest_model(args.mode)



def uniform_sampler(dataset):
    nclasses = len(dataset.imgs)

    imgs_per_class = [0] * len(dataset.classes)
    for _, cl in dataset.imgs:
        imgs_per_class[cl] += 1

    weight_per_img = [0] * nclasses
    for idx, (_, cl) in enumerate(dataset.imgs):
        weight_per_img[idx] = nclasses / imgs_per_class[cl]

    return sampler.WeightedRandomSampler(
        torch.DoubleTensor(weight_per_img),
        nclasses
    )


def check_accuracy(dataset, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y, paths in dataset:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

# def check_accuracy_sampled(dataset, loader):
#     num_correct = 0
#     num_samples = 0
#     model.eval()  # set model to evaluation mode
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
#             y = y.to(device=device, dtype=torch.long)
#             scores = model(x)
#             _, preds = scores.max(1)
#             num_correct += (preds == y).sum()
#             num_samples += preds.size(0)
#         acc = float(num_correct) / num_samples
#         print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


if __name__ == '__main__':

    valset = ImageFolderWithPaths('{}/{}'.format(data_dir, val_dir))
    valfullset = ImageFolderWithPaths('{}/{}'.format(data_dir, val_full_dir))
    valunusualset = ImageFolderWithPaths('{}/{}'.format(data_dir, unusual_dir))

    num_workers = 0 # hangs otherwise
    valset_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=5, shuffle=True,
        num_workers=num_workers)
    valfullset_loader = torch.utils.data.DataLoader(
        valfullset,
        batch_size=5, shuffle=True,
        num_workers=num_workers)
    valunusualset_loader = torch.utils.data.DataLoader(
        valunusualset,
        batch_size=5, shuffle=True,
        num_workers=num_workers)

    # use the loader that is specified in the initial arguments
    loaders = [valset_loader, valfullset_loader, valunusualset_loader]
    if args.mode == "normal":
        loaders = loaders[1:]
    if args.unusual:
        loaders = [valunusualset_loader]

    USE_GPU = True
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('using device:', device)

    # model
    model = torch.load('../models/'+model_fname, device)
    # print(model)
    print('loaded model ' + model_fname)

    for loader in loaders:
        check_accuracy(loader, model)
