"""
Created on July, 2019

@author: Yinyu Nie

configuration of the SUNRGBD data

Cite:
Huang, Siyuan, et al. "Cooperative Holistic Scene Understanding: Unifying 3D Object, Layout,
and Camera Pose Estimation." Advances in Neural Information Processing Systems. 2018.

"""
import os

class SUNRGBD_CONFIG(object):
    def __init__(self):
        """
        Configuration of SUNRGBD data paths.
        """
        self.metadata_path = './data/sunrgbd'
        self.data_root = os.path.join(self.metadata_path, 'Dataset')
        self.clean_data_root = os.path.join(self.data_root, 'data_clean')
        self.datasource_path = os.path.join(self.data_root, 'SUNRGBD')
        self.class_mapping_file = os.path.join(self.metadata_path, 'class_mapping_from_toolbox.csv')
        self.obj_avg_size_file = os.path.join(self.metadata_path, 'preprocessed/size_avg_category.pkl')
        self.layout_avg_file = os.path.join(self.metadata_path, 'preprocessed/layout_avg_file.pkl')
        # denote those samples that are wrongly labeled.
        self.error_samples = [936, 1044, 1711, 2644, 8619,
                              8713, 8929, 8947, 9166, 9215,
                              9249, 9257, 9298, 9325, 9560, 9578]
        # self.class_mapping_894_to_40 = os.path.join(self.metadata_path, 'classMapping40.mat')

        if not os.path.exists(self.class_mapping_file):
            self.__save_nyuclass_mapping()

    def __save_nyuclass_mapping(self):

        import pandas as pd
        import scipy.io as sio
        from configs.data_config import NYU40CLASSES

        # read data
        class_mapping_sunrgbd_to_37 = os.path.join(self.metadata_path, 'SUNRGBD_37_mapping.mat')
        mapping_file = sio.loadmat(class_mapping_sunrgbd_to_37)

        mapping_list = mapping_file['SUNRGBD_37_mapping_unique']
        names_6585 = ['void'] + [mapping_file['seglistall'][0][i][0] for i in
                                 range(len(mapping_file['seglistall'][0]))]
        # construct data frame
        labels_37 = pd.DataFrame({
            'Label_37': list(range(0, 38)),
            'Name_37': NYU40CLASSES[:38]
        })

        parsing = mapping_list[:, 0]
        unique = mapping_list[:, 1]

        Name_6585 = [names_6585[index].lower() for index in parsing]

        labels_full = pd.DataFrame({
            'Label_6585': parsing,
            'Label_37': unique,
            'Name_6585': Name_6585})

        final_dataset = pd.merge(labels_full, labels_37)
        final_dataset.to_csv(self.class_mapping_file, sep=',')


class SUNRGBD_DATA(object):

    def __init__(self, K, R, scene_types, rgb_img, depth_map, layout_3D, sample_id, room_name, instance_data):
        self.__cam_K = K
        self.__cam_R = R
        self.__scene_types = scene_types
        self.__rgb_img = rgb_img
        self.__depth_map = depth_map
        self.__layout_3D = layout_3D
        self.__sample_id = sample_id
        self.__room_name = room_name
        self.__instance_data = instance_data

    def __str__(self):
        return 'room name: {}, sample id: {}'.format(self.__room_name, self.__sample_id)

    def __repr__(self):
        return self.__str__()

    @property
    def cam_K(self):
        return self.__cam_K

    @property
    def cam_R(self):
        return self.__cam_R

    @property
    def scene_types(self):
        return self.__scene_types

    @property
    def rgb_img(self):
        return self.__rgb_img

    @property
    def depth_map(self):
        return self.__depth_map

    @property
    def layout_3D(self):
        return self.__layout_3D

    @property
    def sample_id(self):
        return self.__sample_id

    @property
    def room_name(self):
        return self.__room_name

    @property
    def instance_data(self):
        return self.__instance_data



