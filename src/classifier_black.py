import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

import torch.nn.functional as F  # useful stateless functions


data_dir = 'C:/Users/raffc/Downloads/coco2017'

train_dir = 'train2017_black'
val_dir = 'val2017_black'

val_full_dir = 'val2017_processed_images'

# pytorch convenience stuff
def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

# Wrap `flatten` function in a module in order to stack it
# in nn.Sequential
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

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

    trainset_loader = torch.utils.data.DataLoader(trainset,
         batch_size=64, shuffle=True,
         num_workers=4)
    valset_loader = torch.utils.data.DataLoader(valset,
         batch_size=64, shuffle=True,
         num_workers=4)
    valfullset_loader = torch.utils.data.DataLoader(valfullset,
         batch_size=64, shuffle=True,
         num_workers=4)


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
    
    channel_1 = 64
    channel_2 = 32
    channel_3 = 32
    hidden_1 = 2000
    hidden_2 = 400


    model = nn.Sequential(
        nn.Conv2d(3, channel_1, kernel_size=7, stride=1, padding=6),
        nn.MaxPool2d(2),
        nn.Conv2d(channel_1, channel_2, kernel_size=7, stride=1, padding=6),
        nn.MaxPool2d(2),
        nn.Conv2d(channel_2, channel_3, kernel_size=5, stride=1, padding=4),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(4608, hidden_1),
        nn.BatchNorm1d(hidden_1, eps=1e-05, momentum=0.1),
        nn.ReLU(),
        nn.Linear(hidden_1, hidden_2),
        nn.BatchNorm1d(hidden_2, eps=1e-05, momentum=0.1),
        nn.ReLU(),
        nn.Linear(hidden_2, 2),
    )

    model = model.cuda()

    epochs = 2


    learning_rate = 5e-3

    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                     momentum=0.9, nesterov=True)

    # train
    # model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(trainset_loader):

            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(valset_loader, model)
                check_accuracy(valfullset_loader, model)
                print()


    # final val check
    print('done training, getting final val acc')
    check_accuracy(valset_loader, model)
    check_accuracy(valfullset_loader, model)


    torch.save(model, 'classifier_black.nn')
