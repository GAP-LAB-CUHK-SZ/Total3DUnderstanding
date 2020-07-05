"""
Created on September, 2019

@author: Yinyu Nie

Basic configurations for furniture reconstruction from images.
"""

class Config(object):
    def __init__(self, dataset):
        """
        Configuration of data paths.
        """
        self.dataset = dataset
        self.root_path = './data/' + self.dataset
        self.train_split = self.root_path + '/splits/train.json'
        self.test_split = self.root_path + '/splits/test.json'
        self.metadata_path = self.root_path + '/metadata'
        self.train_test_data_path = self.root_path + '/train_test_data'
        if dataset == 'pix3d':
            self.metadata_file = self.metadata_path + '/pix3d.json'
            self.classnames = ['misc',
                               'bed', 'bookcase', 'chair', 'desk', 'sofa',
                               'table', 'tool', 'wardrobe']

number_pnts_on_template = 2562
neighbors = 30