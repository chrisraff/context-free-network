from pytorch_helper_functions import *
from local_paths import *
import torch
import numpy as np
import torch.nn.functional as F  # useful stateless functions
import argparse
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument("-l", "--loader", default="valset", help="Data loader type (valset/valfullset)")
parser.add_argument("-m", "--mode", default="only_background", help="Mode (random/black)")
parser.add_argument("-n", "--threshold", default=0.9, type=float, help="minimum percentage confidence on incorrect class to be considered \"novel\"")
args = parser.parse_args()


train_dir = 'train2017_'+args.mode
val_dir = 'val2017_'+args.mode
val_full_dir = 'val2017_processed_images'

model_fname = latest_model()



def check_accuracy(dataset, model):
    all_images = set()
    confidently_wrong = set()

    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for i, (x, y, paths) in tqdm(enumerate(dataset)):
            # if i > 4:
            #     continue

            all_images.update(paths)

            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)  # scores shape: [batchsize, num_classes]

            scores = F.softmax(scores, dim=1)


            values, preds = scores.max(1)

            incorrect_preds = (preds != y)

            confident_preds = (values > args.threshold)

            novel_mask = (incorrect_preds & confident_preds)
            novel_ids = np.where(novel_mask == 1)

            paths = np.array(paths)
            novel_paths = paths[novel_ids]

            confidently_wrong.update(novel_paths)


        print("writing novel image paths to novel_image_paths.txt")
        novel_images = sorted(list(confidently_wrong))
        with open('novel_image_paths.txt', 'w') as f:
            for novel_image_path in novel_images:
                f.write("{}\n".format(novel_image_path))

        print("found {} novels images out of {} total images".format(len(confidently_wrong), len(all_images)))


if __name__ == '__main__':
    trainset = ImageFolderWithPaths('{}/{}'.format(data_dir, train_dir))
    valset = ImageFolderWithPaths('{}/{}'.format(data_dir, val_dir))
    valfullset = ImageFolderWithPaths('{}/{}'.format(data_dir, val_full_dir))

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
    if args.loader == "valfullset":
        loader = valfullset_loader

    # use the gpu if there is one
    USE_GPU = True
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("loading model from file")
    model = torch.load(model_fname, device)

    print("finding novel images")
    # check_accuracy(valset_loader, model)
    check_accuracy(valfullset_loader, model)
