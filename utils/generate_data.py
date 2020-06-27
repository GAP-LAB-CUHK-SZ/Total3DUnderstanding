"""
Created on July, 2019

@author: Yinyu Nie

Preprocess the sunrgbd data

Cite:
Huang, Siyuan, et al. "Cooperative Holistic Scene Understanding: Unifying 3D Object, Layout,
and Camera Pose Estimation." Advances in Neural Information Processing Systems. 2018.
"""
from utils.sunrgbd_config import SUNRGBD_CONFIG
from configs.data_config import NYU40CLASSES
from utils.sunrgbd_utils import readsunrgbdframe, process_sunrgbd_frame, check_bdb
from utils.vis_tools_sunrgbd import Scene3D_SUNRGBD
import os
import numpy as np
import pickle
from configs.data_config import Config
from utils.sunrgbd_utils import proj_from_point_to_2d, get_corners_of_bb3d_no_index
from net_utils.libs import get_iou
from libs.tools import camera_cls_reg_sunrgbd, layout_size_avg_residual, ori_cls_reg, obj_size_avg_residual, bin_cls_reg, list_of_dict_to_dict_of_list
import json
from multiprocessing import Pool

def get_avg_data_by_sample(sample_input):

    i, sunrgbd_config = sample_input
    print('Processing SUNRGBD sample ID: %d.' % (i + 1))

    # initiate sizes of objects for each category
    sizes_category = {}
    for class_id in range(len(NYU40CLASSES)):
        sizes_category[class_id] = []

    # get parse sunrgbd data
    sequence = readsunrgbdframe(sunrgbd_config, image_id=i + 1)
    frame = process_sunrgbd_frame(sequence, flip=False)

    # get layout information
    layout_centroid = frame.layout_3D['centroid']
    layout_coeffs = frame.layout_3D['coeffs']

    # get object size information
    for bdb3d in frame.instance_data['bdb3d']:
        sizes_category[bdb3d['class_id']].append(bdb3d['coeffs'])

    sample_output = [sizes_category, layout_centroid, layout_coeffs]

    return sample_output

def write_avg_data(sunrgbd_config):
    '''
    get the average object sizes (by category), average layout centroid depth and average layout coefficients.
    '''

    # prepare samples for multi_processing
    sample_inputs = [(i, sunrgbd_config) for i in range(10335) if (i+1) not in sunrgbd_config.error_samples]

    p = Pool()
    avg_data = p.map(get_avg_data_by_sample, sample_inputs)
    p.close()
    p.join()

    # initiate sizes of objects for each category
    sizes_category = {}
    for class_id in range(len(NYU40CLASSES)):
        sizes_category[class_id] = []

    # the two lists below is just to mark down the layout avg centroid and avg coeffs
    layout_centroid = []
    layout_coeffs = []

    # combine all samples
    for avg_sample in avg_data:
        sizes_category_sample = avg_sample[0]
        layout_centroid_sample = avg_sample[1]
        layout_coeffs_sample = avg_sample[2]

        layout_centroid.append(layout_centroid_sample)
        layout_coeffs.append(layout_coeffs_sample)

        for key, value in sizes_category_sample.items():
            sizes_category[key] += value


    print('Average layout centroid:')
    print(np.array(layout_centroid).mean(axis=0))
    print('Average layout coefficients:')
    print(np.array(layout_coeffs).mean(axis=0))

    mean_size = np.vstack([np.vstack(sizes_category[key]) for key in sizes_category if sizes_category[key]]).mean(0)

    size_avg_category = {}
    for class_id, sizes in sizes_category.items():
        if not sizes_category[class_id]:
            # if a category does not exist, use the average size for initialization
            size_avg_category[class_id] = mean_size
            print('Class: %s does not exists in database.' % (NYU40CLASSES[class_id]))
        else:
            size_avg_category[class_id] = np.average(sizes_category[class_id], 0)

    # write object average size for each category
    with open(sunrgbd_config.obj_avg_size_file, 'wb') as file:
        pickle.dump(size_avg_category, file, protocol=pickle.HIGHEST_PROTOCOL)

    # use the new avg information to generate ground truth.
    dict = {}
    dict['layout_centroid_avg'] = np.average(layout_centroid, axis=0).tolist()
    dict['layout_coeffs_avg'] = np.average(layout_coeffs, axis=0).tolist()

    # write layout average data for each category
    with open(sunrgbd_config.layout_avg_file, 'wb') as file:
        pickle.dump(dict, file, protocol=pickle.HIGHEST_PROTOCOL)

def save_gt_result(sample_input):
    i = sample_input[0]
    gt_config = sample_input[1]
    sunrgbd_config = sample_input[2]
    metadata = sample_input[3]
    iou_threshold = sample_input[4]
    flip = sample_input[5]

    bin = gt_config.bins
    print('Processing sample ID: %d.' % (i + 1))
    sequence = readsunrgbdframe(sunrgbd_config, image_id=i + 1)
    frame = process_sunrgbd_frame(sequence, flip)

    # get 2D 3D bboxes of each object.
    instances = []

    for bdb3d_idx, bdb3d in enumerate(frame.instance_data['bdb3d']):

        center_from_3D, invalid_ids = proj_from_point_to_2d(bdb3d['centroid'], frame.cam_K, frame.cam_R)

        # fiter out objects whose 3D center is behind the camera center as we did in SUNCG dataset,
        # even though its not possible in SUNRGBD dataset.
        if invalid_ids.size:
            continue

        bdb3d_corners = get_corners_of_bb3d_no_index(bdb3d['basis'], bdb3d['coeffs'], bdb3d['centroid'])

        bdb2D_from_3D = proj_from_point_to_2d(bdb3d_corners, frame.cam_K, frame.cam_R)[0]

        bdb2D_from_3D = {'x1': max(bdb2D_from_3D[:, 0].min(), 0),
                         'y1': max(bdb2D_from_3D[:, 1].min(), 0),
                         'x2': min(bdb2D_from_3D[:, 0].max(), frame.rgb_img.shape[1] - 1),
                         'y2': min(bdb2D_from_3D[:, 1].max(), frame.rgb_img.shape[0] - 1)}

        if not check_bdb(bdb2D_from_3D, frame.rgb_img.shape[1] - 1, frame.rgb_img.shape[0] - 1):
            print('bdb2d from 3d gt: NYU class: %s is not valid' % (bdb3d['class_id']))
            continue

        # find the corresponding 2D bbox.
        max_iou = 0
        iou_idx = -1

        for bdb2d_idx, bdb2d in enumerate(frame.instance_data['bdb2d']):
            if bdb2d['class_id'] == bdb3d['class_id']:
                iou = get_iou(bdb2D_from_3D, bdb2d)
                if iou > iou_threshold and iou > max_iou:
                    iou_idx = bdb2d_idx
                    max_iou = iou

        if iou_idx >= 0:
            instance = {'id': bdb3d_idx,
                        'class_id': bdb3d['class_id'],
                        'bdb2D': frame.instance_data['bdb2d'][iou_idx],
                        'bdb3D': bdb3d,
                        'bdb3D_corners': bdb3d_corners,
                        'bdb2D_from_3D': bdb2D_from_3D,
                        'projected_2D_center': center_from_3D,
                        'mask': frame.instance_data['inst_masks'][iou_idx]}

            instances.append(instance)

    # filter out a sample without any objects
    if not instances:
        return None

    '''get camera information'''
    camera = {}
    camera['pitch_cls'], camera['pitch_reg'], \
    camera['roll_cls'], camera['roll_reg'] = camera_cls_reg_sunrgbd(frame.cam_R, bin, i + 1)
    camera['K'] = frame.cam_K

    '''get layout information'''
    layout = {}
    # layout basis is not needed, we regard it as world system, hence its an identity matrix.
    l_centroid = frame.layout_3D['centroid']
    l_coeffs = frame.layout_3D['coeffs']
    l_basis = frame.layout_3D['basis']

    layout['ori_cls'], layout['ori_reg'] = ori_cls_reg(l_basis[0, :], bin['layout_ori_bin'])

    # use the way for object bbox estimation to predict layout bbox.
    layout['centroid_reg'] = l_centroid - bin['layout_centroid_avg']
    layout['coeffs_reg'] = layout_size_avg_residual(l_coeffs, bin['layout_coeffs_avg'])

    layout['bdb3D'] = get_corners_of_bb3d_no_index(frame.layout_3D['basis'], frame.layout_3D['coeffs'],
                                                   frame.layout_3D['centroid'])

    '''get object instance information'''
    boxes_out = []

    for instance in instances:
        box_set = {}
        # Note that z-axis (3rd dimension) points toward the frontal direction
        box_set['ori_cls'], box_set['ori_reg'] = ori_cls_reg(instance['bdb3D']['basis'][2, :], bin['ori_bin'])

        box_set['size_reg'] = obj_size_avg_residual(instance['bdb3D']['coeffs'], metadata['size_avg_category'],
                                                    instance['class_id'])

        box_set['bdb3D'] = instance['bdb3D_corners']

        bdb2D_from_3D = instance['bdb2D_from_3D']

        box_set['bdb2D_from_3D'] = [bdb2D_from_3D['x1'] / float(frame.cam_K[0][2]),
                                    bdb2D_from_3D['y1'] / float(frame.cam_K[1][2]),
                                    bdb2D_from_3D['x2'] / float(frame.cam_K[0][2]),
                                    bdb2D_from_3D['y2'] / float(frame.cam_K[1][2])]

        box_set['bdb2D_pos'] = [instance['bdb2D']['x1'],
                                instance['bdb2D']['y1'],
                                instance['bdb2D']['x2'],
                                instance['bdb2D']['y2']]

        box_set['centroid_cls'], box_set['centroid_reg'] = bin_cls_reg(bin['centroid_bin'],
                                                                       np.linalg.norm(instance['bdb3D']['centroid']))

        # The shift between projected 2D center (from 3D centroid ) and 2D bounding box center
        delta_2D = []
        delta_2D.append(
            ((box_set['bdb2D_pos'][0] + box_set['bdb2D_pos'][2]) / 2. - instance['projected_2D_center'][0]) / (
                        box_set['bdb2D_pos'][2] - box_set['bdb2D_pos'][0]))
        delta_2D.append(
            ((box_set['bdb2D_pos'][1] + box_set['bdb2D_pos'][3]) / 2. - instance['projected_2D_center'][1]) / (
                        box_set['bdb2D_pos'][3] - box_set['bdb2D_pos'][1]))
        box_set['delta_2D'] = delta_2D
        box_set['size_cls'] = instance['class_id']
        box_set['mask'] = instance['mask']
        boxes_out.append(box_set)

    if not boxes_out:
        return None

    data = {}
    data['rgb_img'] = frame.rgb_img
    data['depth_map'] = frame.depth_map
    data['boxes'] = list_of_dict_to_dict_of_list(boxes_out)
    data['camera'] = camera
    data['layout'] = layout
    data['sequence_id'] = frame.sample_id

    if not flip:
        save_path = os.path.join(gt_config.train_test_data_path, str(data['sequence_id']) + '.pkl')
    else:
        save_path = os.path.join(gt_config.train_test_data_path, str(data['sequence_id']) + '_flip.pkl')

    with open(save_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return (frame.sample_id, save_path)

def preprocess(sunrgbd_config, gt_config, metadata, iou_threshold=0.1, flip=False, data_range=range(10335)):
    '''
    generate groundtruth data for training.
    :param samples: all samples from each viewpoint.
    :param metadata: required prior data for ground-truth generation.
    :param flip: flip objects to augment data.
    '''
    # save result by multi-processing
    sample_inputs = [(i, gt_config, sunrgbd_config, metadata, iou_threshold, flip) for i in data_range if (i+1) not in sunrgbd_config.error_samples]

    p = Pool(processes=4)
    save_outputs = p.map(save_gt_result, sample_inputs)
    p.close()
    p.join()

    # remove empty output
    save_outputs = [item for item in save_outputs if item]

    train_path_list = []
    test_path_list = []

    for sample_id, save_path in save_outputs:
        # get the path for data loading for training and testing
        if sample_id <= 5050:
            test_path_list.append(save_path)
        else:
            train_path_list.append(save_path)

    return train_path_list, test_path_list

if __name__ == '__main__':

    sunrgbd_config = SUNRGBD_CONFIG()

    if not os.path.exists(sunrgbd_config.obj_avg_size_file) or not os.path.exists(sunrgbd_config.layout_avg_file):
        write_avg_data(sunrgbd_config)

    gt_config = Config('sunrgbd')
    # load average size priors
    metadata = dict()
    with open(sunrgbd_config.obj_avg_size_file, 'rb') as file:
        size_avg_category = pickle.load(file)
    metadata['size_avg_category'] = size_avg_category

    train_path_list_no_flip, test_path_list_no_flip = preprocess(sunrgbd_config, gt_config, metadata, iou_threshold=0.1, flip=False)
    # flip training set
    train_path_list_flip, test_path_list_flip = preprocess(sunrgbd_config, gt_config, metadata, iou_threshold=0.1, flip=True, data_range=range(5050, 10335))

    # save training path and testing path
    train_path_list = train_path_list_no_flip + train_path_list_flip
    with open(os.path.join(gt_config.metadata_path, 'preprocessed', 'train.json'), 'w') as f:
        json.dump(train_path_list, f)

    test_path_list = test_path_list_no_flip + test_path_list_flip
    with open(os.path.join(gt_config.metadata_path, 'preprocessed', 'test.json'), 'w') as f:
        json.dump(test_path_list, f)

    # visualize SUNRGBD samples
    sequence = readsunrgbdframe(sunrgbd_config, image_id=274)
    scene = Scene3D_SUNRGBD(sequence)
    scene.draw_image()
    scene.draw_2dboxes()
    scene.draw_cls()
    scene.draw_inst()
    scene.draw_projected_bdb3d()
    scene.draw3D()