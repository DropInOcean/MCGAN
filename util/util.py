from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import scipy.io
import io


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    
    return image_numpy.astype(imtype)

def tensor2im_2(image_tensor):
    image_numpy = image_tensor[0].cpu().float().numpy()
    # print(image_numpy.shape)
    
    image_numpy = np.squeeze(image_numpy)
    # print(image_numpy.shape)
    
    image_numpy = (image_numpy + 1) / 2.0
    
    return image_numpy


def tensor2im_3(image_tensor,imtype=np.uint8):
    num_weight = len(image_tensor)
    print(num_weight)
    
    # for i, weights in enumerate(image_tensor):
    #     print(weights.shape)
    #     heatmap = weights.cpu().detach().numpy()
    #     print(heatmap.shape)
        
    #     if weights.shape[0] == 1:
    #         image_numpy = np.tile(heatmap, (3, 1, 1))
    
    #     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        
    # print(image_numpy.shape)
    
    # return image_numpy.astype(imtype)

    image_numpy = image_tensor[0].cpu().detach().float().numpy()
    # print(image_numpy.shape)
    image_numpy = np.squeeze(image_numpy)
        
    # print(image_numpy.shape)
    image_numpy = (image_numpy + 1) / 2.0
    
    return image_numpy        
        
        

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    print(image_numpy.shape)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
    
def save_image_2(image_numpy, image_path):
    scipy.io.savemat(image_path, {'data': image_numpy})
    # num_py = len(image_numpy)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
