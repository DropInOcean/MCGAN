import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import scipy.io
import torch.utils.data
import random
    

class UnalignedDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'T1')
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'T2')
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        self.input_nc = self.opt.output_nc if self.opt.which_direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.which_direction == 'BtoA' else self.opt.output_nc

        
    def __getitem__(self, index):
        # read a image given a random integer index
        A_path = self.A_paths[index % self.A_size]
        
        B_path = self.B_paths[index % self.B_size]
        
        
        A=scipy.io.loadmat(A_path)
        A = A['data']
        B=scipy.io.loadmat(B_path)
        B = B['data']
        
        data_x = np.array(A)
        data_y=np.array(B)
        
        data_x = np.expand_dims(data_x, axis=0)
        data_y = np.expand_dims(data_y, axis=0)
        
        data_x[:,...]=(data_x[:,...]-0.5)/0.5
        data_y[:,...]=(data_y[:,...]-0.5)/0.5
        
        data_x = data_x.astype(np.float32)
        data_y = data_y.astype(np.float32)
        
        

        return {'A': torch.from_numpy(data_x[:,...]), 'B': torch.from_numpy(data_y[:,...]), 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)

    
