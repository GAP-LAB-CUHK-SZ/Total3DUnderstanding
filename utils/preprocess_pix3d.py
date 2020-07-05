'''
Preprocess pix3d data.
author: ynie
date: Sep, 2019
'''

import sys
sys.path.append('.')
from configs.pix3d_config import Config
import json
from multiprocessing import Pool
import pickle
from libs.tools import read_obj, sample_pnts_from_obj, normalize_to_unit_square
import os
from PIL import Image
import numpy as np

def parse_sample(sample_info):
    sample_id = sample_info[0]
    print('Processing sample: %d.' % (sample_id))
    sample = sample_info[1]
    is_flip = sample_info[2]

    class_id = config.classnames.index(sample['category'])
    save_path = os.path.join(config.train_test_data_path, str(sample_id) + '.pkl')

    if os.path.exists(save_path):
        return None

    data = {}
    data['sample_id'] = sample_id

    '''get single object image'''
    img = np.array(Image.open(os.path.join(config.metadata_path, sample['img'])).convert('RGB'))
    img = img[sample['bbox'][1]:sample['bbox'][3], sample['bbox'][0]:sample['bbox'][2]]
    data['img'] = img

    '''get object model'''
    model_path = os.path.join(config.metadata_path, sample['model'])
    obj_data = read_obj(model_path, ['v', 'f'])
    sampled_points = sample_pnts_from_obj(obj_data, 10000, mode='random')
    sampled_points = normalize_to_unit_square(sampled_points)[0]

    data['gt_3dpoints'] = sampled_points

    '''get object category'''
    data['class_id'] = class_id

    pickle.dump(data, save_path)

    if is_flip:
        save_path = os.path.join(config.train_test_data_path, str(sample_id) + '_flipped.pkl')
        if os.path.exists(save_path):
            return None

        data_flip = {}
        data_flip['sample_id'] = sample_id
        img_flip = np.array(Image.open(os.path.join(config.metadata_path, sample['img'])).convert('RGB').transpose(Image.FLIP_LEFT_RIGHT))
        x1 = sample['img_size'][0] - 1 - sample['bbox'][2]
        x2 = sample['img_size'][0] - 1 - sample['bbox'][0]
        y1 = sample['bbox'][1]
        y2 = sample['bbox'][3]

        img_flip = img_flip[y1:y2, x1:x2]
        data_flip['img'] = img_flip
        data_flip['gt_3dpoints'] = sampled_points
        data_flip['class_id'] = class_id
        pickle.dump(data_flip, save_path)

if __name__ == '__main__':
    config = Config('pix3d')
    with open(config.metadata_file, 'r') as file:
        metadata = json.load(file)

    with open(config.train_split, 'r') as file:
        train_split = json.load(file)

    with open(config.test_split, 'r') as file:
        test_split = json.load(file)

    train_ids = [int(os.path.basename(file).split('.')[0]) for file in train_split if 'flipped' not in file]
    test_ids = [int(os.path.basename(file).split('.')[0]) for file in test_split if 'flipped' not in file]

    train_sample_info = [(sample_id, metadata[sample_id], True) for sample_id in train_ids]
    test_sample_info = [(sample_id, metadata[sample_id], False) for sample_id in test_ids]

    p = Pool(processes=1)
    p.map(parse_sample, train_sample_info)
    p.close()
    p.join()

    p = Pool(processes=1)
    p.map(parse_sample, test_sample_info)
    p.close()
    p.join()
