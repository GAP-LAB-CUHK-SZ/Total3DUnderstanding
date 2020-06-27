# Base data of networks
# author: ynie
# date: Feb, 2020
import os
from torch.utils.data import Dataset
import json

class SUNRGBD(Dataset):
    def __init__(self, config, mode):
        '''
        initiate SUNRGBD dataset for data loading
        :param config: config file
        :param mode: train/val/test mode
        '''
        self.config = config
        self.mode = mode
        split_file = os.path.join(config['data']['split'], mode + '.json')
        with open(split_file) as file:
            self.split = json.load(file)

    def __len__(self):
        return len(self.split)


class PIX3D(Dataset):
    def __init__(self, config, mode):
        '''
        initiate PIX3D dataset for data loading
        :param config: config file
        :param mode: train/val/test mode
        '''
        self.config = config
        self.mode = mode
        split_file = os.path.join(config['data']['split'], mode + '.json')
        with open(split_file) as file:
            self.split = json.load(file)

    def __len__(self):
        return len(self.split)