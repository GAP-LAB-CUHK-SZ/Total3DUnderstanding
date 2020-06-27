# Definition of PoseNet
# author: ynie
# date: March, 2020

import torch
import torch.nn as nn
from models.registers import MODULES
from configs.data_config import number_pnts_on_template, pix3d_n_classes
from models.modules import resnet
from models.modules.resnet import model_urls
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from net_utils.misc import weights_init, sphere_edges, sphere_faces, sphere_edge2face, sphere_adjacency, sphere_points_normals, sample_points_on_edges

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 2500, output_dim = 3):
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck_size, bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(bottleneck_size, bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(bottleneck_size//2, bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(bottleneck_size//4, output_dim, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size//4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

class EREstimate(nn.Module):
    def __init__(self, bottleneck_size=2500, output_dim = 3):
        super(EREstimate, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck_size, bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(bottleneck_size, bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(bottleneck_size//2, bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(bottleneck_size//4, output_dim, 1)

        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size//4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x


@MODULES.register_module
class DensTMNet(nn.Module):
    def __init__(self, cfg, optim_spec=None, bottleneck_size=1024, n_classes=pix3d_n_classes, pretrained_encoder=True):
        super(DensTMNet, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Module parameters'''
        self.num_points = number_pnts_on_template
        self.subnetworks = cfg.config['data']['tmn_subnetworks']
        self.train_e_e = cfg.config['data']['with_edge_classifier']

        '''Modules'''
        self.encoder = resnet.resnet18_full(pretrained=False, num_classes=1024)
        self.decoders = nn.ModuleList(
            [PointGenCon(bottleneck_size=3 + bottleneck_size + n_classes) for i in range(0, self.subnetworks)])

        if self.train_e_e:
            self.error_estimators = nn.ModuleList(
                [EREstimate(bottleneck_size=3 + bottleneck_size + n_classes, output_dim=1) for i in range(0, max(self.subnetworks-1, 1))])

            self.face_samples = cfg.config['data']['face_samples']

        # initialize weight
        self.apply(weights_init)

        # initialize resnet
        if pretrained_encoder:
            pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
            model_dict = self.encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('fc.')}
            model_dict.update(pretrained_dict)
            self.encoder.load_state_dict(model_dict)

    def unfreeze_parts(self, loose_parts):
        # freeze all
        for param in self.parameters():
            param.requires_grad = False
        print('All layers freezed.')
        # unfreeze parts
        if 'encoder' in loose_parts:
            for param in self.encoder.parameters():
                param.requires_grad = True
            print('Encoder unfrozen.')

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print('Encoder freezed.')

    def freeze_by_stage(self, stage, loose_parts):
        if stage >= 1:
            # freeze all
            for param in self.parameters():
                param.requires_grad = False
            print('All layers freezed.')

            if 'decoder' in loose_parts:
                # unfreeze the last sub-network of decoders.
                for param in self.decoders[-1].parameters():
                    param.requires_grad = True
                print('Decoder unfrozen.')

            if 'ee' in loose_parts and hasattr(self, 'error_estimators'):
                # unfreeze the last sub-network of error estimators.
                for param in self.error_estimators[-1].parameters():
                    param.requires_grad = True
                print('EE unfrozen.')

    def forward(self, image, size_cls, threshold = 0.1, factor = 1.):
        mode = 'train' if self.training else 'test'
        device = image.device

        n_batch = image.size(0)
        n_edges = sphere_edges.shape[0]

        if mode == 'test':
            current_faces = sphere_faces.clone().unsqueeze(0).to(device)
            current_faces = current_faces.repeat(n_batch, 1, 1)
        else:
            current_faces = None

        current_edges = sphere_edges.clone().unsqueeze(0).to(device)
        current_edges = current_edges.repeat(n_batch, 1, 1)

        # image encoding
        image = image[:,:3,:,:].contiguous()
        image = self.encoder(image)
        image = torch.cat([image, size_cls], 1)

        current_shape_grid = sphere_points_normals[:, :3].t().expand(n_batch, 3, self.num_points).to(device)

        # outputs for saving
        out_shape_points = []
        out_sampled_mesh_points = []
        out_indicators = []

        # boundary faces for boundary refinement
        boundary_point_ids = torch.zeros(size=(n_batch, self.num_points), dtype=torch.uint8).to(device)
        remove_edges_list = []

        # AtlasNet deformation + topoly modification
        for i in range(self.subnetworks):
            current_image_grid = image.unsqueeze(2).expand(image.size(0), image.size(1),
                                                           current_shape_grid.size(2)).contiguous()
            current_image_grid = torch.cat((current_shape_grid, current_image_grid), 1).contiguous()
            current_shape_grid = current_shape_grid + self.decoders[i](current_image_grid)

            # save deformed point cloud
            out_shape_points.append(current_shape_grid)

            if i == self.subnetworks - 1 and self.subnetworks > 1:
                remove_edges_list = [item for item in remove_edges_list if len(item)]
                if remove_edges_list:
                    remove_edges_list = torch.unique(torch.cat(remove_edges_list), dim=0)
                    for batch_id in range(n_batch):
                        rm_edges = remove_edges_list[remove_edges_list[:, 0] == batch_id, 1]
                        if len(rm_edges) > 0:
                            rm_candidates, counts = torch.unique(sphere_edges[rm_edges], return_counts=True)
                            boundary_ids = counts < sphere_adjacency[rm_candidates - 1].sum(1)
                            boundary_point_ids[batch_id][rm_candidates[boundary_ids] - 1] = 1

                return out_shape_points, out_sampled_mesh_points, out_indicators, current_edges, boundary_point_ids, current_faces

            if self.train_e_e:
                # sampling from deformed mesh
                sampled_points = sample_points_on_edges(current_shape_grid, current_edges, quantity=self.face_samples, mode=mode)

                # save sampled points from deformed mesh
                out_sampled_mesh_points.append(sampled_points)

                # preprare for face error estimation
                current_image_grid = image.unsqueeze(2).expand(image.size(0), image.size(1), sampled_points.size(2)).contiguous()
                current_image_grid = torch.cat((sampled_points, current_image_grid), 1).contiguous()

                # estimate the distance from deformed points to gt mesh.
                indicators = self.error_estimators[i](current_image_grid)
                indicators = indicators.view(n_batch, 1, n_edges, self.face_samples)
                indicators = indicators.squeeze(1)
                indicators = torch.mean(indicators, dim=2)

                # save estimated distance values from deformed points to gt mesh.
                out_indicators.append(indicators)
                # remove faces and modify the topology
                remove_edges = torch.nonzero(torch.sigmoid(indicators) < threshold)
                remove_edges_list.append(remove_edges)

                for batch_id in range(n_batch):
                    rm_edges = remove_edges[remove_edges[:, 0] == batch_id, 1]
                    if len(rm_edges)>0:
                        # cutting edges in training
                        current_edges[batch_id][rm_edges, :] = 1
                        if mode == 'test':
                            current_faces[batch_id][sphere_edge2face[rm_edges].sum(0).type(torch.bool), :] = 1

                threshold *= factor

        return out_shape_points, out_sampled_mesh_points, out_indicators, current_edges, boundary_point_ids, current_faces