import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass



def get_transform(opt):
    transform_list = []
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5,),
                                            (0.5,))]
    return transforms.Compose(transform_list)





def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
