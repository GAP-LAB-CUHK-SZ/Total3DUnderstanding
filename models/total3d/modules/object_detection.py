# Definition of PoseNet
# author: ynie
# date: March, 2020

import torch
import torch.nn as nn
from models.registers import MODULES
from models.modules import resnet
from models.modules.resnet import model_urls
import torch.utils.model_zoo as model_zoo
from models.total3d.modules.relation_net import RelationNet
from configs.data_config import NYU40CLASSES


@MODULES.register_module
class Bdb3DNet(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(Bdb3DNet, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Module parameters'''
        bin = cfg.dataset_config.bins
        self.OBJ_ORI_BIN = len(bin['ori_bin'])
        self.OBJ_CENTER_BIN = len(bin['centroid_bin'])

        # set up neural network blocks
        self.resnet = nn.DataParallel(resnet.resnet34(pretrained=False))

        # set up relational network blocks
        self.relnet = RelationNet()

        # branch to predict the size
        self.fc1 = nn.Linear(2048 + len(NYU40CLASSES), 128)
        self.fc2 = nn.Linear(128, 3)

        # branch to predict the orientation
        self.fc3 = nn.Linear(2048 + len(NYU40CLASSES), 128)
        self.fc4 = nn.Linear(128, self.OBJ_ORI_BIN * 2)

        # branch to predict the centroid
        self.fc5 = nn.Linear(2048 + len(NYU40CLASSES), 128)
        self.fc_centroid = nn.Linear(128, self.OBJ_CENTER_BIN * 2)

        # branch to predict the 2D offset
        self.fc_off_1 = nn.Linear(2048 + len(NYU40CLASSES), 128)
        self.fc_off_2 = nn.Linear(128, 2)

        self.relu_1 = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(p=0.5)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

        # initialize resnet weights
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = self.resnet.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)

    def forward(self, x, size_cls, g_features, split, rel_pair_counts):
        '''
        Extract relational features for object bounding box estimation.

        The definition of 'batch' in train.py indicates the number of images we process in a single forward broadcasting.
        In this implementation, we speed-up the efficiency by processing all objects in a batch in parallel.

        As each image contains various number (N_i) of objects, it refers to an issue to assign which image an object belongs to.
        We address the problem by involving a look-up table in 'split'.

        Therefore, The meaning of 'batch' in this function actually refers to a 'patch' of objects.

        :param x: Patch_size x Channel_size x Height x Width
        :param size_cls: Patches x Number_of_classes.
        :param g_features: SUM(N_i^2) x 64
        g_features records the geometric features (64-D) between each pair of objects (see Hu et al. [2]). So the dimension
        is Number_of_pairs_in_images x 64 (or SUM(N_i^2) x 64). N_i is the number of objects in the i-th image.
        :param split: Batch_size x 2
        split records which batch a object belongs to.
        e.g. split = torch.tensor([[0, 5], [5, 8]]) when batch size is 2, and there are 5 objects in the first batch and
        3 objects in the second batch.
        Then the first 5 objects in the whole patch belongs to the first batch, and the rest belongs to the second batch.
        :param rel_pair_counts: (Batch_size + 1)
        rel_pair_counts records which batch a geometric feature belongs to, and gives the start and end index.
        e.g. rel_pair_counts = torch.tensor([0, 49, 113]).
        The batch size is two. The first 49 geometric features are from the first batch.
        The index begins from 0 and ends at 49. The second 64 geometric features are from the second batch.
        The index begins from 49 and ends at 113.
        :return: Object bounding box properties.
        '''

        # get appearance feature from resnet.
        a_features = self.resnet(x)
        a_features = a_features.view(a_features.size(0), -1)

        # extract relational features from other objects.
        r_features = self.relnet(a_features, g_features, split, rel_pair_counts)

        a_r_features = torch.add(a_features, r_features)

        # add object category information
        a_r_features = torch.cat([a_r_features, size_cls], 1)

        # branch to predict the size
        size = self.fc1(a_r_features)
        size = self.relu_1(size)
        size = self.dropout_1(size)
        size = self.fc2(size)

        # branch to predict the orientation
        ori = self.fc3(a_r_features)
        ori = self.relu_1(ori)
        ori = self.dropout_1(ori)
        ori = self.fc4(ori)
        ori = ori.view(-1, self.OBJ_ORI_BIN, 2)
        ori_reg = ori[:, :, 0]
        ori_cls = ori[:, :, 1]

        # branch to predict the centroid
        centroid = self.fc5(a_r_features)
        centroid = self.relu_1(centroid)
        centroid = self.dropout_1(centroid)
        centroid = self.fc_centroid(centroid)
        centroid = centroid.view(-1, self.OBJ_CENTER_BIN, 2)
        centroid_cls = centroid[:, :, 0]
        centroid_reg = centroid[:, :, 1]

        # branch to predict the 2D offset
        offset = self.fc_off_1(a_r_features)
        offset = self.relu_1(offset)
        offset = self.dropout_1(offset)
        offset = self.fc_off_2(offset)

        return size, ori_reg, ori_cls, centroid_reg, centroid_cls, offset