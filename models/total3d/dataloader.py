# Dataloader of Total3D.
# author: ynie
# date: Feb, 2020
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data
from torchvision import transforms
from models.datasets import SUNRGBD
import pickle
from PIL import Image
import numpy as np
from configs.data_config import Relation_Config, NYU40CLASSES
import math
import collections

default_collate = torch.utils.data.dataloader.default_collate

HEIGHT_PATCH = 256
WIDTH_PATCH = 256
rel_cfg = Relation_Config()
d_model = int(rel_cfg.d_g/4)

data_transforms_crop = transforms.Compose([
    transforms.Resize((280, 280)),
    transforms.RandomCrop((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms_nocrop = transforms.Compose([
    transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

pil2tensor = transforms.ToTensor()

class Total3D_Dataset(SUNRGBD):
    def __init__(self, config, mode):
        super(Total3D_Dataset, self).__init__(config, mode)

    def __getitem__(self, index):

        file_path = self.split[index]
        with open(file_path, 'rb') as f:
            sequence = pickle.load(f)
        image = Image.fromarray(sequence['rgb_img'])
        depth = Image.fromarray(sequence['depth_map'])
        camera = sequence['camera']
        boxes = sequence['boxes']

        # build relational geometric features for each object
        n_objects = boxes['bdb2D_pos'].shape[0]
        # g_feature: n_objects x n_objects x 4
        # Note that g_feature is not symmetric,
        # g_feature[m, n] is the feature of object m contributes to object n.
        # TODO: think about it, do we need to involve the geometric feature from each object itself?
        g_feature = [[((loc2[0] + loc2[2]) / 2. - (loc1[0] + loc1[2]) / 2.) / (loc1[2] - loc1[0]),
                      ((loc2[1] + loc2[3]) / 2. - (loc1[1] + loc1[3]) / 2.) / (loc1[3] - loc1[1]),
                      math.log((loc2[2] - loc2[0]) / (loc1[2] - loc1[0])),
                      math.log((loc2[3] - loc2[1]) / (loc1[3] - loc1[1]))] \
                     for id1, loc1 in enumerate(boxes['bdb2D_pos'])
                     for id2, loc2 in enumerate(boxes['bdb2D_pos'])]

        locs = [num for loc in g_feature for num in loc]

        pe = torch.zeros(len(locs), d_model)
        position = torch.from_numpy(np.array(locs)).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        boxes['g_feature'] = pe.view(n_objects * n_objects, rel_cfg.d_g)

        # encode class
        cls_codes = torch.zeros([len(boxes['size_cls']), len(NYU40CLASSES)])
        cls_codes[range(len(boxes['size_cls'])), boxes['size_cls']] = 1
        boxes['size_cls'] = cls_codes

        layout = sequence['layout']

        # TODO: If the training error is consistently larger than the test error. We remove the crop and add more intermediate FC layers with no dropout.
        # TODO: Or FC layers with more hidden neurons, which ensures more neurons pass through the dropout layer, or with larger learning rate, longer
        # TODO: decay rate.
        data_transforms = data_transforms_crop if self.mode=='train' else data_transforms_nocrop

        patch = []
        for bdb in boxes['bdb2D_pos']:
            img = image.crop((bdb[0], bdb[1], bdb[2], bdb[3]))
            img = data_transforms(img)
            patch.append(img)
        boxes['patch'] = torch.stack(patch)
        image = data_transforms_nocrop(image)
        return {'image':image, 'depth': pil2tensor(depth).squeeze(), 'boxes_batch':boxes, 'camera':camera, 'layout':layout, 'sequence_id': sequence['sequence_id']}

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
        if key == 'boxes_batch':
            collated_batch[key] = dict()
            for subkey in batch[0][key]:
                if subkey == 'mask':
                    tensor_batch = [elem[key][subkey] for elem in batch]
                else:
                    list_of_tensor = [recursive_convert_to_torch(elem[key][subkey]) for elem in batch]
                    tensor_batch = torch.cat(list_of_tensor)
                collated_batch[key][subkey] = tensor_batch
        elif key == 'depth':
            collated_batch[key] = [elem[key] for elem in batch]
        else:
            collated_batch[key] = default_collate([elem[key] for elem in batch])

    interval_list = [elem['boxes_batch']['patch'].shape[0] for elem in batch]
    collated_batch['obj_split'] = torch.tensor([[sum(interval_list[:i]), sum(interval_list[:i+1])] for i in range(len(interval_list))])

    return collated_batch

def Total3D_dataloader(config, mode='train'):
    dataloader = DataLoader(dataset=Total3D_Dataset(config, mode),
                            num_workers=config['device']['num_workers'],
                            batch_size=config[mode]['batch_size'],
                            shuffle=(mode == 'train'),
                            collate_fn=collate_fn)
    return dataloader
