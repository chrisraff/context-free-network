import torch
import torch.nn as nn
from torchvision import datasets, transforms as T


# pytorch convenience stuff
def flatten(x):
    N = x.shape[0]  # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


# Wrap `flatten` function in a module in order to stack it in nn.Sequential
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)


tensify = T.Compose([
    T.ToTensor()
])



class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, path):
        super().__init__(path, tensify)
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

# # EXAMPLE USAGE:
# # instantiate the dataset and dataloader
# data_dir = "your/data_dir/here"
# dataset = ImageFolderWithPaths(data_dir) # our custom dataset
# dataloader = torch.utils.DataLoader(dataset)

# # iterate over data
# for inputs, labels, paths in dataloader:
#     # use the above variables freely
# print(inputs, labels, paths)
