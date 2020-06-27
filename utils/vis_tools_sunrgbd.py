"""
Created on June, 2019

@author: Yinyu Nie

Toolkit functions used for visualizing rooms.
"""

from utils.vis_tools import Scene3D
import numpy as np
from utils.sunrgbd_utils import get_NYU37_class_id, cvt2nyu37class_map, process_layout, get_world_R, transform_to_world,\
    cvt_R_ex_to_cam_R, check_bdb2d, process_bdb3d, process_bdb2d, get_inst_map, normalize_point, get_campact_layout
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from configs.data_config import NYU40CLASSES
import vtk
from utils.sunrgbd_utils import get_layout_info, proj_from_point_to_2d, get_corners_of_bb3d_no_index
from PIL import Image

nyu_colorbox = np.array(sns.color_palette("hls", n_colors=len(NYU40CLASSES)))

nyu_color_palette = dict()
for class_id, color in enumerate(nyu_colorbox):
    nyu_color_palette[class_id] = color.tolist()

class Scene3D_SUNRGBD(Scene3D):
    '''
    A class used to visualize SUNRGBD scene contents.
    '''
    def __init__(self, sequence):
        self.__cam_K = sequence.K
        self.__img_map = sequence.imgrgb
        self.__cls_map = np.array(Image.open(sequence.semantic_seg2d))
        self.__inst_map, self.__inst_classes = get_inst_map(sequence.seg2d, self.cls_map)
        # self.__inst_classes = get_NYU37_class_id(inst_names)
        # self.__cls_map = cvt2nyu37class_map(self.__inst_map, self.__inst_classes)
        self.__depth_map = sequence.imgdepth

        cam_R = cvt_R_ex_to_cam_R(sequence.R_ex)

        # define a world system
        world_R = get_world_R(cam_R)

        layout = process_layout(sequence.manhattan_layout)
        centroid = layout['centroid']
        vectors = np.diag(layout['coeffs']).dot(layout['basis'])
        # Set all points relative to layout orientation. (i.e. let layout orientation to be the world system.)
        # The forward direction (x-axis) of layout orientation should point toward camera forward direction.
        layout_3D = get_layout_info({'centroid': centroid, 'vectors': vectors}, cam_R[:, 0])

        self.__bdb2d = process_bdb2d(check_bdb2d(sequence.bdb2d, sequence.imgrgb.shape), sequence.imgrgb.shape)

        bdb3ds_ws = process_bdb3d(sequence.bdb3d)

        # transform everything to world system
        self.__layout, self.__bdb3d, self.__cam_R = transform_to_world(layout_3D, bdb3ds_ws, cam_R, world_R)

        # self.__layout = get_campact_layout(self.layout, self.depth_map, self.cam_K, self.cam_R, self.bdb3d)


    @property
    def inst_map(self):
        return self.__inst_map

    @property
    def cls_map(self):
        return self.__cls_map

    @property
    def img_map(self):
        return self.__img_map

    @property
    def depth_map(self):
        return self.__depth_map

    @property
    def cam_K(self):
        return self.__cam_K

    @property
    def inst_classes(self):
        return self.__inst_classes

    @property
    def layout(self):
        return self.__layout

    @property
    def cam_R(self):
        return self.__cam_R

    @property
    def bdb2d(self):
        return self.__bdb2d

    @property
    def bdb3d(self):
        return self.__bdb3d

    def draw_projected_bdb3d(self):
        from PIL import Image, ImageDraw, ImageFont

        img_map = Image.fromarray(self.img_map[:])

        draw = ImageDraw.Draw(img_map)

        width = 5

        for bdb3d in self.bdb3d:
            center_from_3D, invalid_ids = proj_from_point_to_2d(bdb3d['centroid'], self.cam_K, self.cam_R)
            bdb3d_corners = get_corners_of_bb3d_no_index(bdb3d['basis'], bdb3d['coeffs'], bdb3d['centroid'])
            bdb2D_from_3D = proj_from_point_to_2d(bdb3d_corners, self.cam_K, self.cam_R)[0]

            # bdb2D_from_3D = np.round(bdb2D_from_3D).astype('int32')
            bdb2D_from_3D = [tuple(item) for item in bdb2D_from_3D]

            color = nyu_color_palette[bdb3d['class_id']]

            draw.line([bdb2D_from_3D[0], bdb2D_from_3D[1], bdb2D_from_3D[2], bdb2D_from_3D[3], bdb2D_from_3D[0]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[4], bdb2D_from_3D[5], bdb2D_from_3D[6], bdb2D_from_3D[7], bdb2D_from_3D[4]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[0], bdb2D_from_3D[4]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[1], bdb2D_from_3D[5]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[2], bdb2D_from_3D[6]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)
            draw.line([bdb2D_from_3D[3], bdb2D_from_3D[7]],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=width)

            draw.text(tuple(center_from_3D), NYU40CLASSES[bdb3d['class_id']],
                      fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 20))

        img_map.show()

    def draw_image(self):
        plt.imshow(self.img_map)
        plt.axis('off')
        plt.show()

    def draw_cls(self):
        class_ids = np.unique(self.cls_map)
        color_box = []

        color_map = np.zeros_like(self.cls_map)
        for color_id, class_id in enumerate(class_ids):
            color_box.append(nyu_color_palette[class_id])
            color_map[self.cls_map == class_id] = color_id

        plt.figure()
        ax = plt.gca()
        im = ax.imshow(color_map, cmap=ListedColormap(color_box))
        colorbar = plt.colorbar(im)
        colorbar.set_ticks(np.arange((color_map.max() - color_map.min()) / (2 * len(class_ids)), color_map.max(),
                                     (color_map.max() - color_map.min()) / len(class_ids)))
        colorbar.set_ticklabels([NYU40CLASSES[id] for id in np.unique(self.cls_map)])
        plt.axis('equal')
        plt.axis('off')
        plt.show()

    def draw_inst(self):
        image = np.copy(self.img_map)

        plt.cla()

        for inst_id, class_id in self.inst_classes.items():
            mask = self.inst_map==inst_id

            if True not in mask:
                continue

            color = (255*np.array(nyu_color_palette[class_id])).astype(np.uint8)
            image[mask] = 0.6 * color + 0.4 * image[mask]

            centre = (np.min(np.argwhere(mask), axis=0) + np.max(np.argwhere(mask), axis=0))/2.
            plt.gca().text(centre[1], centre[0],
                           '{0:d}:{1:s}'.format(inst_id, NYU40CLASSES[class_id]),
                           bbox=dict(facecolor=color/255., alpha=0.9), fontsize=12, color='white')

        plt.axis("off")
        plt.imshow(image)
        plt.show()

    def draw_2dboxes(self, scale=1.0):
        plt.cla()
        plt.axis('off')
        plt.imshow(self.img_map)
        for bdb in self.bdb2d:
            bbox = np.array([bdb['x1'], bdb['y1'], bdb['x2'], bdb['y2']]) * scale
            color = nyu_color_palette[bdb['class_id']]
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, edgecolor=color,
                                 linewidth=2.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1], '{:s}'.format(NYU40CLASSES[bdb['class_id']]), bbox=dict(facecolor=color, alpha=0.5),
                           fontsize=9, color='white')
        plt.show()

    def set_render(self):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        '''draw layout system'''
        renderer.AddActor(self.set_axes_actor())

        '''draw layout bounding box'''
        centroid = self.layout['centroid']
        vectors = np.diag(self.layout['coeffs']).dot(self.layout['basis'])

        corners, faces = self.get_box_corners(centroid, vectors)
        color = (200, 200, 200)
        layout_actor = self.set_actor(self.set_mapper(self.set_cube_prop(corners, faces, color), 'box'))
        layout_actor.GetProperty().SetOpacity(0.1)
        renderer.AddActor(layout_actor)

        layout_line_actor = self.set_actor(self.set_mapper(self.set_bbox_line_actor(corners, faces, (0,0,0)), 'box'))
        layout_line_actor.GetProperty().SetOpacity(0.5)
        layout_line_actor.GetProperty().SetLineWidth(3)
        renderer.AddActor(layout_line_actor)
        # draw layout orientation
        color = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        for index in range(vectors.shape[0]):
            arrow_actor = self.set_arrow_actor(centroid, vectors[index])
            arrow_actor.GetProperty().SetColor(color[index])
            renderer.AddActor(arrow_actor)

        '''draw camera coordinate system'''
        color = [[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]
        centroid = [0, 0, 0]
        vectors = self.cam_R.T
        for index in range(vectors.shape[0]):
            arrow_actor = self.set_arrow_actor(centroid, vectors[index])
            arrow_actor.GetProperty().SetColor(color[index])
            renderer.AddActor(arrow_actor)

        # '''set camera property'''
        # camera = self.set_camera(centroid, vectors, self.cam_K)
        # renderer.SetActiveCamera(camera)

        '''draw point cloud from depth image'''
        point_actor = self.set_actor(self.set_mapper(self.set_points_property(np.eye(3)), 'box'))
        point_actor.GetProperty().SetPointSize(1)
        renderer.AddActor(point_actor)

        '''draw 3D object bboxes'''
        print()
        for bdb3d in self.bdb3d:
            # draw 3D bounding boxes
            color = nyu_color_palette[bdb3d['class_id']]
            # transform from world system to layout system
            centroid = bdb3d['centroid']
            vectors = np.diag(bdb3d['coeffs']).dot(bdb3d['basis'])
            corners, faces = self.get_box_corners(centroid, vectors)
            bbox_actor = self.set_actor(self.set_mapper(self.set_cube_prop(corners, faces, color), 'box'))
            bbox_actor.GetProperty().SetOpacity(0.3)
            renderer.AddActor(bbox_actor)

            # draw orientations
            color = [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [1., 0., 0.]]

            for index in range(vectors.shape[0]):
                arrow_actor = self.set_arrow_actor(centroid, vectors[index])
                arrow_actor.GetProperty().SetColor(color[index])
                renderer.AddActor(arrow_actor)

        renderer.SetBackground(1., 1., 1.)

        return renderer, None


