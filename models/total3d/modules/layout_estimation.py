# Definition of PoseNet
# author: ynie
# date: March, 2020

import torch
import torch.nn as nn
from models.registers import MODULES
from models.modules import resnet
from models.modules.resnet import model_urls
import torch.utils.model_zoo as model_zoo


@MODULES.register_module
class PoseNet(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(PoseNet, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Module parameters'''
        bin = cfg.dataset_config.bins
        self.PITCH_BIN = len(bin['pitch_bin'])
        self.ROLL_BIN = len(bin['roll_bin'])
        self.LO_ORI_BIN = len(bin['layout_ori_bin'])

        '''Modules'''
        self.resnet = resnet.resnet34(pretrained=False)
        self.fc_1 = nn.Linear(2048, 1024)
        self.fc_2 = nn.Linear(1024, (self.PITCH_BIN + self.ROLL_BIN) * 2)

        # fc for layout
        self.fc_layout = nn.Linear(2048, 2048)
        # for layout orientation
        self.fc_3 = nn.Linear(2048, 1024)
        self.fc_4 = nn.Linear(1024, self.LO_ORI_BIN * 2)
        # for layout centroid and coefficients
        self.fc_5 = nn.Linear(2048, 1024)
        self.fc_6 = nn.Linear(1024, 6)

        self.relu_1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout_1 = nn.Dropout(p=0.5)

        # initiate weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

        # load pretrained resnet
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = self.resnet.state_dict()
        pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)


    def forward(self, x):
        x = self.resnet(x)

        # branch for camera parameters
        cam = self.fc_1(x)
        cam = self.relu_1(cam)
        cam = self.dropout_1(cam)
        cam = self.fc_2(cam)
        pitch_reg = cam[:, 0: self.PITCH_BIN]
        pitch_cls = cam[:, self.PITCH_BIN: self.PITCH_BIN * 2]
        roll_reg = cam[:, self.PITCH_BIN * 2: self.PITCH_BIN * 2 + self.ROLL_BIN]
        roll_cls = cam[:, self.PITCH_BIN * 2 + self.ROLL_BIN: self.PITCH_BIN * 2 + self.ROLL_BIN * 2]

        # branch for layout orientation, centroid and coefficients
        lo = self.fc_layout(x)
        lo = self.relu_1(lo)
        lo = self.dropout_1(lo)
        # branch for layout orientation
        lo_ori = self.fc_3(lo)
        lo_ori = self.relu_1(lo_ori)
        lo_ori = self.dropout_1(lo_ori)
        lo_ori = self.fc_4(lo_ori)
        lo_ori_reg = lo_ori[:, :self.LO_ORI_BIN]
        lo_ori_cls = lo_ori[:, self.LO_ORI_BIN:]

        # branch for layout centroid and coefficients
        lo_ct = self.fc_5(lo)
        lo_ct = self.relu_1(lo_ct)
        lo_ct = self.dropout_1(lo_ct)
        lo_ct = self.fc_6(lo_ct)
        lo_centroid = lo_ct[:, :3]
        lo_coeffs = lo_ct[:, 3:]

        return pitch_reg, roll_reg, pitch_cls, roll_cls, lo_ori_reg, lo_ori_cls, lo_centroid, lo_coeffs