import kornia
import torch
from torch.utils.data import ConcatDataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

class RGBToYCbCr:
    def __init__(self):
        pass
    
    def __call__(self, x):
        return kornia.color.rgb_to_ycbcr(x)
    
class YCbCrToRGB:
    def __init__(self):
        pass
    
    def __call__(self, x):
        return kornia.color.ycbcr_to_rgb(x)

class GrayscaleToRGB:
    def __init__(self):
        pass
    
    def __call__(self, x):
        return x.repeat(1, 3, 1, 1)
    

def get_cifar_train_val_test(root, split_ratio=(0.8, 0.1, 0.1)):
    """torchvision.dataset.CIFAR10 does not have a validation set, this function combines the
    train and test sets of the CIFAR10 and splits the dataset randomly to a new train, validation
    and test set.
    """
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # ToTensor() implicitly normalizes to [0,1]
        RGBToYCbCr()            # converts RGB with range [0,1] to YCbCr
    ])   

    dataset = ConcatDataset([
        CIFAR10(root, train=True, transform=transform, download=True),
        CIFAR10(root, train=False, transform=transform)
    ])
    
    split_lengths = [int(len(dataset) * ratio) for ratio in split_ratio]
    train_set, val_set, test_set = random_split(dataset, 
                                                split_lengths, 
                                                generator=torch.Generator().manual_seed(42))
    return train_set, val_set, test_set