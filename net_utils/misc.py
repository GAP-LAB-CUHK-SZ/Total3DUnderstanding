import torch
import numpy as np
import pickle
from configs.data_config import number_pnts_on_template


#initialize the weighs of the network for Convolutional layers and batchnorm layers
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# load sphere faces and points
def load_template(number):
    file_name = './data/sphere%d.pkl' % (number)

    with open(file_name, 'rb') as file:
        sphere_obj = pickle.load(file)
        sphere_points_normals = torch.from_numpy(sphere_obj['v']).float()
        sphere_faces = torch.from_numpy(sphere_obj['f']).long()
        sphere_adjacency = torch.from_numpy(sphere_obj['adjacency'].todense()).long()
        sphere_edges = torch.from_numpy(sphere_obj['edges']).long()
        sphere_edge2face = torch.from_numpy(sphere_obj['edge2face'].todense()).type(torch.uint8)
    return sphere_points_normals, sphere_faces, sphere_adjacency, sphere_edges, sphere_edge2face

sphere_points_normals, sphere_faces, sphere_adjacency, sphere_edges, sphere_edge2face = load_template(number_pnts_on_template)


def sample_points_on_edges(points, edges, quantity = 1, mode = 'train'):
    n_batch = edges.shape[0]
    n_edges = edges.shape[1]

    if mode == 'train':
        # if the sampling rate is larger than 1, we randomly pick points on faces.
        weights = np.diff(np.sort(np.vstack(
            [np.zeros((1, n_edges * quantity)), np.random.uniform(0, 1, size=(1, n_edges * quantity)),
             np.ones((1, n_edges * quantity))]), axis=0), axis=0)
    else:
        # if in test mode, we pick the central point on faces.
        weights = 0.5 * np.ones((2, n_edges * quantity))

    weights = weights.reshape([2, quantity, n_edges])
    weights = torch.from_numpy(weights).float().to(points.device)
    weights = weights.transpose(1, 2)
    weights = weights.transpose(0, 1).contiguous()
    weights = weights.expand(n_batch, n_edges, 2, quantity).contiguous()
    weights = weights.view(n_batch * n_edges, 2, quantity)

    left_nodes = torch.gather(points.transpose(1, 2), 1,
                              (edges[:, :, 0] - 1).unsqueeze(-1).expand(edges.size(0), edges.size(1), 3))
    right_nodes = torch.gather(points.transpose(1, 2), 1,
                              (edges[:, :, 1] - 1).unsqueeze(-1).expand(edges.size(0), edges.size(1), 3))

    edge_points = torch.cat([left_nodes.unsqueeze(-1), right_nodes.unsqueeze(-1)], -1).view(n_batch*n_edges, 3, 2)

    new_point_set = torch.bmm(edge_points, weights).contiguous()
    new_point_set = new_point_set.view(n_batch, n_edges, 3, quantity)
    new_point_set = new_point_set.transpose(2, 3).contiguous()
    new_point_set = new_point_set.view(n_batch, n_edges * quantity, 3)
    new_point_set = new_point_set.transpose(1, 2).contiguous()
    return new_point_set
