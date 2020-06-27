"""
Created on May, 2019

@author: Yinyu Nie

Data configurations.

"""


class Relation_Config(object):
    def __init__(self):
        self.d_g = 64
        self.d_k = 64
        self.Nr = 16

num_samples_on_each_model = 5000
n_object_per_image_in_training = 8

import os
import numpy as np
import pickle

NYU40CLASSES = ['void',
                'wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'blinds', 'desk', 'shelves',
                'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
                'person', 'night_stand', 'toilet', 'sink', 'lamp',
                'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']

NYU37_TO_PIX3D_CLS_MAPPING = {0:0, 1:0, 2:0, 3:8, 4:1, 5:3, 6:5, 7:6, 8:8, 9:2, 10:2, 11:0, 12:0, 13:2, 14:4,
                              15:2, 16:2, 17:8, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0, 24:8, 25:8, 26:0, 27:0, 28:0,
                              29:8, 30:8, 31:0, 32:8, 33:0, 34:0, 35:0, 36:0, 37:8}

RECON_3D_CLS = [3,4,5,6,7,8,10,14,15,17,24,25,29,30,32]

number_pnts_on_template = 2562

pix3d_n_classes = 9

cls_reg_ratio = 10
obj_cam_ratio = 1

class Config(object):
    def __init__(self, dataset):
        """
        Configuration of data paths.
        """
        self.dataset = dataset

        if self.dataset == 'sunrgbd':
            self.metadata_path = './data/sunrgbd'
            self.train_test_data_path = os.path.join(self.metadata_path, 'sunrgbd_train_test_data')
            self.__size_avg_path = os.path.join(self.metadata_path, 'preprocessed/size_avg_category.pkl')
            self.__layout_avg_file = os.path.join(self.metadata_path, 'preprocessed/layout_avg_file.pkl')
            self.bins = self.__initiate_bins()
            self.evaluation_path = './evaluation/sunrgbd'
            if not os.path.exists(self.train_test_data_path):
                os.mkdir(self.train_test_data_path)

    def __initiate_bins(self):
        bin = {}

        if self.dataset == 'sunrgbd':
            # there are faithful priors for layout locations, we can use it for regression.
            if os.path.exists(self.__layout_avg_file):
                with open(self.__layout_avg_file, 'rb') as file:
                    layout_avg_dict = pickle.load(file)
            else:
                raise IOError('No layout average file in %s. Please check.' % (self.__layout_avg_file))

            bin['layout_centroid_avg'] = layout_avg_dict['layout_centroid_avg']
            bin['layout_coeffs_avg'] = layout_avg_dict['layout_coeffs_avg']

            '''layout orientation bin'''
            NUM_LAYOUT_ORI_BIN = 2
            ORI_LAYOUT_BIN_WIDTH = np.pi / 4
            bin['layout_ori_bin'] = [[np.pi / 4 + i * ORI_LAYOUT_BIN_WIDTH, np.pi / 4 + (i + 1) * ORI_LAYOUT_BIN_WIDTH] for i in range(NUM_LAYOUT_ORI_BIN)]

            '''camera bin'''
            PITCH_NUMBER_BINS = 2
            PITCH_WIDTH = 40 * np.pi / 180
            ROLL_NUMBER_BINS = 2
            ROLL_WIDTH = 20 * np.pi / 180

            # pitch_bin = [[-60 * np.pi/180, -20 * np.pi/180], [-20 * np.pi/180, 20 * np.pi/180]]
            bin['pitch_bin'] = [[-60.0 * np.pi / 180 + i * PITCH_WIDTH, -60.0 * np.pi / 180 + (i + 1) * PITCH_WIDTH] for
                                i in range(PITCH_NUMBER_BINS)]
            # roll_bin = [[-20 * np.pi/180, 0 * np.pi/180], [0 * np.pi/180, 20 * np.pi/180]]
            bin['roll_bin'] = [[-20.0 * np.pi / 180 + i * ROLL_WIDTH, -20.0 * np.pi / 180 + (i + 1) * ROLL_WIDTH] for i in
                               range(ROLL_NUMBER_BINS)]

            '''bbox orin, size and centroid bin'''
            # orientation bin
            NUM_ORI_BIN = 6
            ORI_BIN_WIDTH = float(2 * np.pi / NUM_ORI_BIN) # 60 degrees width for each bin.
            # orientation bin ranges from -np.pi to np.pi.
            bin['ori_bin'] = [[(i - NUM_ORI_BIN / 2) * ORI_BIN_WIDTH, (i - NUM_ORI_BIN / 2 + 1) * ORI_BIN_WIDTH] for i
                              in range(NUM_ORI_BIN)]

            if os.path.exists(self.__size_avg_path):
                with open(self.__size_avg_path, 'rb') as file:
                    avg_size = pickle.load(file)
            else:
                raise IOError('No object average size file in %s. Please check.' % (self.__size_avg_path))

            bin['avg_size'] = np.vstack([avg_size[key] for key in range(len(avg_size))])

            # for each object bbox, the distance between camera and object centroid will be estimated.

            NUM_DEPTH_BIN = 6
            DEPTH_WIDTH = 1.0
            # centroid_bin = [0, 6]
            bin['centroid_bin'] = [[i * DEPTH_WIDTH, (i + 1) * DEPTH_WIDTH] for i in
                                   range(NUM_DEPTH_BIN)]
        else:
            raise NameError('Please specify a correct dataset name.')

        return bin
