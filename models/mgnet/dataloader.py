# Dataloader of Mesh Generation Net.
# author: ynie
# date: Feb, 2020
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.utils.data
from models.datasets import PIX3D
import collections
import pickle
from PIL import Image
from configs.data_config import pix3d_n_classes
from scipy.spatial import cKDTree
import numpy as np

default_collate = torch.utils.data.dataloader.default_collate


num_samples_on_each_model = 5000
neighbors = 30

HEIGHT_PATCH = 256
WIDTH_PATCH = 256

data_transforms_nocrop = transforms.Compose([
    transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms_crop = transforms.Compose([
    transforms.Resize((280, 280)),
    transforms.RandomCrop((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class MGNet_Dataset(PIX3D):
    def __init__(self, config, mode):
        super(MGNet_Dataset, self).__init__(config, mode)

    def __getitem__(self, index):
        file_path = self.split[index]
        with open(file_path, 'rb') as file:
            sequence = pickle.load(file)

        image = Image.fromarray(sequence['img'])
        class_id = sequence['class_id']
        gt_points = sequence['gt_3dpoints']

        data_transforms = data_transforms_crop if self.mode=='train' else data_transforms_nocrop

        cls_codes = torch.zeros(pix3d_n_classes)
        cls_codes[class_id] = 1

        tree = cKDTree(gt_points)
        dists, indices = tree.query(gt_points, k=neighbors)
        densities = np.array([max(dists[point_set, 1]) ** 2 for point_set in indices])

        if self.mode == 'train':
            p_ids = np.random.choice(gt_points.shape[0], num_samples_on_each_model, replace=False)
            gt_points = gt_points[p_ids, :]
            densities = densities[p_ids]

        sample = {'sequence_id':sequence['sample_id'],
                  'img':data_transforms(image),
                  'cls':cls_codes,
                  'mesh_points':gt_points,
                  'densities': densities}

        return sample

def recursive_convert_to_torch(elem):
    if torch.is_tensor(elem):
        return elem
    elif type(elem).__module__ == 'numpy':
        if elem.size == 0:
            return torch.zeros(elem.shape).type(torch.DoubleTensor)
        else:
            return torch.from_numpy(elem)
    elif isinstance(elem, int):
        return torch.LongTensor([elem])
    elif isinstance(elem, float):
        return torch.DoubleTensor([elem])
    elif isinstance(elem, collections.Mapping):
        return {key: recursive_convert_to_torch(elem[key]) for key in elem}
    elif isinstance(elem, collections.Sequence):
        return [recursive_convert_to_torch(samples) for samples in elem]
    else:
        return elem

def collate_fn(batch):
    """
    Data collater.

    Assumes each instance is a dict.
    Applies different collation rules for each field.
    Args:
        batches: List of loaded elements via Dataset.__getitem__
    """
    collated_batch = {}
    # iterate over keys
    for key in batch[0]:
        collated_batch[key] = default_collate([elem[key] for elem in batch])

    return collated_batch

def MGNet_dataloader(config, mode='train'):
    dataloader = DataLoader(dataset=MGNet_Dataset(config, mode),
                            num_workers=config['device']['num_workers'],
                            batch_size=config[mode]['batch_size'],
                            shuffle=(mode == 'train'),
                            collate_fn=collate_fn)
    return dataloader
