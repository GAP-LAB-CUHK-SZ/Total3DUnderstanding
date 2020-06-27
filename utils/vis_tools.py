"""
Created on April, 2019

@author: Yinyu Nie

Toolkit functions used for visualizing rooms.
"""

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from configs.data_config import NYU40CLASSES
from utils.sunrgbd_utils import get_cam_KRT, get_layout_info, correct_flipped_objects
from libs.tools import cvt2nyuclass_map, get_inst_classes, get_world_R
import numpy as np
import random
import seaborn as sns
import cv2
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

nyuclass_mapping = np.load('data/nyu40class_mapping.npy')
nyu_colorbox = np.array(sns.color_palette("hls", n_colors=len(NYU40CLASSES)))

nyu_color_palette = dict()
for class_id, color in enumerate(nyu_colorbox):
    nyu_color_palette[class_id] = color.tolist()

class Scene3D(object):
    '''
    A class used to visualize 3D scene contents.
    '''
    def __init__(self, sample):
        # self.cam_paras = sample['cam_paras']
        self._inst_map = cv2.imread(sample['instance_map_path'], -1)
        self._cls_map = cvt2nyuclass_map(cv2.imread(sample['category_map_path'], -1), nyuclass_mapping)
        self._img_map = cv2.cvtColor(cv2.imread(sample['image_path'], -1), cv2.COLOR_BGR2RGB)
        self._depth_map = cv2.imread(sample['depth_map_path'], -1) / 1000.
        self._inst_classes = get_inst_classes(self.inst_map, self.cls_map)
        self._layout = sample['room_bbox']
        self._compact_layout = sample['compact_bbox']
        self._instance_info = sample['instance_info']
        self._cam_K, self._cam_R, self._cam_T = get_cam_KRT(sample['cam_paras'],
                                                               [self.img_map.shape[1], self.img_map.shape[0]])
    @property
    def inst_map(self):
        return self._inst_map

    @property
    def cls_map(self):
        return self._cls_map

    @property
    def img_map(self):
        return self._img_map

    @property
    def depth_map(self):
        return self._depth_map

    @property
    def cam_K(self):
        return self._cam_K

    @property
    def inst_classes(self):
        return self._inst_classes

    @property
    def layout(self):
        return self._layout

    @property
    def compact_layout(self):
        return self._compact_layout

    @property
    def instance_info(self):
        return self._instance_info

    @property
    def cam_R(self):
        return self._cam_R

    @property
    def cam_T(self):
        return self._cam_T

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
            color_map[self.cls_map==class_id] = color_id

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

    def draw_depth(self):
        plt.imshow(self.depth_map, cmap='Greys')
        plt.axis('off')
        plt.show()

    def set_axes_actor(self):
        '''
        Set camera coordinate system
        '''
        transform = vtk.vtkTransform()
        transform.Translate(0., 0., 0.)
        # self defined
        axes = vtk.vtkAxesActor()
        axes.SetUserTransform(transform)
        axes.SetTotalLength(0.8, 0.8, 0.8)

        axes.SetTipTypeToCone()
        axes.SetConeRadius(30e-2)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(40e-3)

        vtk_textproperty = vtk.vtkTextProperty()
        vtk_textproperty.SetFontSize(1)
        vtk_textproperty.SetBold(True)
        vtk_textproperty.SetItalic(False)
        vtk_textproperty.SetShadow(True)

        for label in [axes.GetXAxisCaptionActor2D(), axes.GetYAxisCaptionActor2D(), axes.GetZAxisCaptionActor2D()]:
            label.SetCaptionTextProperty(vtk_textproperty)

        return axes

    def set_actor(self, mapper):
        '''
        vtk general actor
        :param mapper: vtk shape mapper
        :return: vtk actor
        '''
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

    def set_mapper(self, prop, mode):

        mapper = vtk.vtkPolyDataMapper()

        if mode == 'model':
            mapper.SetInputConnection(prop.GetOutputPort())

        elif mode == 'box':
            if vtk.VTK_MAJOR_VERSION <= 5:
                mapper.SetInput(prop)
            else:
                mapper.SetInputData(prop)

            # mapper.SetScalarRange(0, 7)

        else:
            raise IOError('No Mapper mode found.')

        return mapper

    def mkVtkIdList(self, it):
        vil = vtk.vtkIdList()
        for i in it:
            vil.InsertNextId(int(i))
        return vil

    def set_cube_prop(self, corners, faces, color):

        cube = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName('Color')

        color = np.uint8(np.array(color)*255)

        for i in range(8):
            points.InsertPoint(i, corners[i])

        for i in range(6):
            polys.InsertNextCell(self.mkVtkIdList(faces[i]))

        for i in range(8):
            colors.InsertNextTuple3(*color)

        # Assign the pieces to the vtkPolyData
        cube.SetPoints(points)
        del points
        cube.SetPolys(polys)
        del polys
        cube.GetPointData().SetScalars(colors)
        cube.GetPointData().SetActiveScalars('Color')
        del colors

        return cube

    def set_bbox_line_actor(self, corners, faces, color):
        edge_set1 = np.vstack([np.array(faces)[:, 0], np.array(faces)[:, 1]]).T
        edge_set2 = np.vstack([np.array(faces)[:, 1], np.array(faces)[:, 2]]).T
        edge_set3 = np.vstack([np.array(faces)[:, 2], np.array(faces)[:, 3]]).T
        edge_set4 = np.vstack([np.array(faces)[:, 3], np.array(faces)[:, 0]]).T
        edges = np.vstack([edge_set1, edge_set2, edge_set3, edge_set4])
        edges = np.unique(np.sort(edges, axis=1), axis=0)

        pts = vtk.vtkPoints()
        for corner in corners:
            pts.InsertNextPoint(corner)

        lines = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        for edge in edges:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, edge[0])
            line.GetPointIds().SetId(1, edge[1])
            lines.InsertNextCell(line)
            colors.InsertNextTuple3(*color)

        linesPolyData = vtk.vtkPolyData()
        linesPolyData.SetPoints(pts)
        linesPolyData.SetLines(lines)
        linesPolyData.GetCellData().SetScalars(colors)

        return linesPolyData

    def get_box_corners(self, center, vectors):
        '''
        Convert box center and vectors to the corner-form
        :param center:
        :param vectors:
        :return: corner points and faces related to the box
        '''
        corner_pnts = [None] * 8
        corner_pnts[0] = tuple(center - vectors[0] + vectors[1] - vectors[2])
        corner_pnts[1] = tuple(center - vectors[0] + vectors[1] + vectors[2])
        corner_pnts[2] = tuple(center + vectors[0] + vectors[1] + vectors[2])
        corner_pnts[3] = tuple(center + vectors[0] + vectors[1] - vectors[2])
        corner_pnts[4] = tuple(center - vectors[0] - vectors[1] - vectors[2])
        corner_pnts[5] = tuple(center - vectors[0] - vectors[1] + vectors[2])
        corner_pnts[6] = tuple(center + vectors[0] - vectors[1] + vectors[2])
        corner_pnts[7] = tuple(center + vectors[0] - vectors[1] - vectors[2])

        faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (0, 3, 7, 4)]

        return corner_pnts, faces

    def read_transform_model(self, instance, layout_R):
        '''
        read object model to vtk object type, and transform it from object system to camera coordinate system.
        (with its bounding box)
        :param instance: model path of instance.
        :return: vtk object (with bounding box) in transformed coordinate system.
        '''
        object = vtk.vtkOBJReader()
        object.SetFileName(instance['model_path'][0])
        object.Update()

        # transform point values
        # get points from object
        polydata = object.GetOutput()
        # read points using vtk_to_numpy
        points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float)

        # get transformation matrix (to tranform from object system to world system)
        transform_mat = np.array(instance['obj_property']['transform']).reshape(4, -1).T

        # correct those wrongly labeled objects to correct frontal direction.
        points, transform_mat, _, _ = correct_flipped_objects(points, transform_mat, instance['model_path'][0])

        points = np.hstack([points, np.ones([points.shape[0], 1])])

        # for bounding box (get center-vector form)
        max_point = points.max(0)
        min_point = points.min(0)
        center = (max_point + min_point) / 2.
        sizes = (max_point - min_point) / 2.
        vectors = np.array([[sizes[0], 0., 0.], [0., sizes[1], 0.], [0., 0., sizes[2]]])

        points = transform_mat.dot(points.T).T

        # for bounding box (transform to world system)
        center = transform_mat.dot(center)
        vectors = transform_mat[:3,:3].dot(vectors.T).T

        # make sure vectors are in upright and right-hand system
        vectors[1, :] = vectors[1, :] if vectors[1, 1] >= 0 else -vectors[1, :]
        vectors[0, :] = vectors[0, :] if np.linalg.det(vectors)>=0 else -vectors[0, :]

        # transform from world system to layout system
        points = (points[:,:3]-self.cam_T).dot(layout_R)

        # for the bounding box (transform from world system to layout system)
        center_layout = (center[:3] - self.cam_T).dot(layout_R)
        vectors_layout = vectors.dot(layout_R)

        points_array = numpy_to_vtk(points, deep=True)
        polydata.GetPoints().SetData(points_array)
        object.Update()

        return object, (center_layout, vectors_layout)


    def set_arrow_actor(self, startpoint, vector):
        '''
        Design an actor to draw an arrow from startpoint to startpoint + vector.
        :param startpoint: 3D point
        :param vector: 3D vector
        :return: an vtk arrow actor
        '''
        arrow_source = vtk.vtkArrowSource()
        arrow_source.SetTipLength(0.2)
        arrow_source.SetTipRadius(0.08)
        arrow_source.SetShaftRadius(0.02)

        vector = vector/np.linalg.norm(vector)*0.5

        endpoint = startpoint + vector

        # compute a basis
        normalisedX = [0 for i in range(3)]
        normalisedY = [0 for i in range(3)]
        normalisedZ = [0 for i in range(3)]

        # the X axis is a vector from start to end
        math = vtk.vtkMath()
        math.Subtract(endpoint, startpoint, normalisedX)
        length = math.Norm(normalisedX)
        math.Normalize(normalisedX)

        # the Z axis is an arbitrary vector cross X
        arbitrary = [0 for i in range(3)]
        arbitrary[0] = random.uniform(-10, 10)
        arbitrary[1] = random.uniform(-10, 10)
        arbitrary[2] = random.uniform(-10, 10)
        math.Cross(normalisedX, arbitrary, normalisedZ)
        math.Normalize(normalisedZ)


        # the Y axis is Z cross X
        math.Cross(normalisedZ, normalisedX, normalisedY)

        # create the direction cosine matrix
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        for i in range(3):
            matrix.SetElement(i, 0, normalisedX[i])
            matrix.SetElement(i, 1, normalisedY[i])
            matrix.SetElement(i, 2, normalisedZ[i])

        # apply the transform
        transform = vtk.vtkTransform()
        transform.Translate(startpoint)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)

        # create a mapper and an actor for the arrow
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()

        mapper.SetInputConnection(arrow_source.GetOutputPort())
        actor.SetUserMatrix(transform.GetMatrix())
        actor.SetMapper(mapper)

        return actor

    def set_points_property(self, layout_R):

        u, v = np.meshgrid(range(self.depth_map.shape[1]), range(self.depth_map.shape[0]))
        u = u.reshape([1, -1])[0]
        v = v.reshape([1, -1])[0]
        color_indices = self.img_map[v, u]

        z_cam = self.depth_map[v, u]

        # remove zeros
        non_zero_indices = np.argwhere(z_cam).T[0]
        z_cam = z_cam[non_zero_indices]
        u = u[non_zero_indices]
        v = v[non_zero_indices]
        color_indices = color_indices[non_zero_indices]

        # calculate coordinates
        x_cam = (u - self.cam_K[0][2])*z_cam/self.cam_K[0][0]
        y_cam = (v - self.cam_K[1][2])*z_cam/self.cam_K[1][1]

        # transform to toward-up-right coordinate system
        x3 = z_cam
        y3 = -y_cam
        z3 = x_cam

        # transform from camera system to layout system
        points_cam = np.vstack([x3, y3, z3]).T
        points_layout = points_cam.dot(self.cam_R.T).dot(layout_R)
        x3 = points_layout[:, 0]
        y3 = points_layout[:, 1]
        z3 = points_layout[:, 2]

        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName('Color')

        for x, y, z, c in zip(x3, y3, z3, color_indices):
            id = points.InsertNextPoint([x, y, z])
            colors.InsertNextTuple3(*c)
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(id)

        # Create a polydata object
        point = vtk.vtkPolyData()
        # Set the points and vertices we created as the geometry and topology of the polydata
        point.SetPoints(points)
        point.SetVerts(vertices)
        point.GetPointData().SetScalars(colors)
        point.GetPointData().SetActiveScalars('Color')

        return point

    def proj_layout_to_img(self, voxel_points, layout_R):

        points_cam = voxel_points.dot(layout_R.T).dot(self.cam_R)

        # convert to traditional image coordinate system
        T_cam = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
        points_cam = points_cam.dot(T_cam.T)
        # delete those points whose depth value is non-positive.
        invalid_ids = np.where(points_cam[:, 2] <= 0)[0]
        points_cam[invalid_ids, 2] = 0.0001

        points_cam_h = points_cam / points_cam[:, 2][:, None]
        pixels = self.cam_K.dot(points_cam_h.T)

        pixels = pixels[:2, :].T.astype(np.int)

        idx = (0 <= pixels[:, 0]) & \
              (pixels[:, 0] < self.img_map.shape[1]) & \
              (0 <= pixels[:, 1]) & \
              (pixels[:, 1] < self.img_map.shape[0])

        pixels = pixels[idx]

        return pixels

    def set_camera(self, position, focal_point, cam_K):
        camera = vtk.vtkCamera()
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal_point[0])
        camera.SetViewUp(*focal_point[1])
        camera.SetViewAngle((2*np.arctan(cam_K[1][2]/cam_K[0][0]))/np.pi*180)
        return camera

    def set_render(self):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        '''set world system'''
        world_R = get_world_R(self.cam_R)

        '''draw world system'''
        renderer.AddActor(self.set_axes_actor())

        '''draw compact layout (bounding box)'''
        center, vectors = self.compact_layout
        # move layout center to camera center
        center -= self.cam_T
        # transform layout and camera to world system
        center = center.dot(world_R)
        vectors = vectors.dot(world_R)
        cam_R = (world_R.T).dot(self.cam_R)

        # Set layout points forward the toward direction of camera
        layout_3D = get_layout_info({'centroid': center, 'vectors': vectors}, cam_R[:,0])
        center = layout_3D['centroid']
        vectors = np.diag(layout_3D['coeffs']).dot(layout_3D['basis'])

        corners, faces = self.get_box_corners(center, vectors)
        color = (200, 200, 200)
        layout_actor = self.set_actor(self.set_mapper(self.set_cube_prop(corners, faces, color), 'box'))
        layout_actor.GetProperty().SetOpacity(0.1)
        renderer.AddActor(layout_actor)

        # draw layout orientation
        color = [[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]
        for index in range(vectors.shape[0]):
            arrow_actor = self.set_arrow_actor(center, vectors[index])
            arrow_actor.GetProperty().SetColor(color[index])
            renderer.AddActor(arrow_actor)

        # '''draw full room layout (bounding box)'''
        # sizes = (np.array(self.layout['max']) - np.array(self.layout['min']))/2.
        # center = (np.array(self.layout['min']) + np.array(self.layout['max'])) / 2.
        # vectors = np.array([[sizes[0], 0., 0.], [0., sizes[1], 0.], [0., 0., sizes[2]]])
        #
        # # transform box to world system
        # center = (center - self.cam_T).dot(world_R)
        # vectors = vectors.dot(world_R)
        #
        # corners, faces = self.get_box_corners(center, vectors)
        # color = (200, 200, 200)
        # layout_actor = self.set_actor(self.set_mapper(self.set_cube_prop(corners, faces, color), 'box'))
        # layout_actor.GetProperty().SetOpacity(0.3)
        # renderer.AddActor(layout_actor)

        '''draw camera coordinate system'''
        color = [[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]
        center = [0,0,0]
        vectors = cam_R.T
        for index in range(vectors.shape[0]):
            arrow_actor = self.set_arrow_actor(center, vectors[index])
            arrow_actor.GetProperty().SetColor(color[index])
            renderer.AddActor(arrow_actor)

        # '''set camera property'''
        # camera = self.set_camera(center, vectors, self.cam_K)
        # renderer.SetActiveCamera(camera)

        '''draw point cloud from depth image'''
        point_actor = self.set_actor(self.set_mapper(self.set_points_property(world_R), 'box'))
        point_actor.GetProperty().SetPointSize(2)
        renderer.AddActor(point_actor)

        '''draw 3D objects (models with bounding boxes and orientations)'''
        voxel_image = np.zeros_like(self.img_map)
        for instance in self.instance_info:
            # only present in-room object-type stuff
            if not instance['inroom'] or instance['type']!='Object':
                continue

            object, bbox = self.read_transform_model(instance, world_R)
            color = nyu_color_palette[self.inst_classes[instance['inst_id']]]

            # draw 3D object models
            object_actor = self.set_actor(self.set_mapper(object, 'model'))
            object_actor.GetProperty().SetColor(color)
            renderer.AddActor(object_actor)

            # draw 3D object bounding boxes
            center, vectors = bbox
            corners, faces = self.get_box_corners(center, vectors)
            bbox_actor = self.set_actor(self.set_mapper(self.set_cube_prop(corners, faces, color), 'box'))
            bbox_actor.GetProperty().SetOpacity(0.3)
            renderer.AddActor(bbox_actor)

            # draw orientations
            color = [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [1., 0., 0.]]

            for index in range(vectors.shape[0]):
                arrow_actor = self.set_arrow_actor(center, vectors[index])
                arrow_actor.GetProperty().SetColor(color[index])
                renderer.AddActor(arrow_actor)

        renderer.SetBackground(1., 1., 1.)

        return renderer, voxel_image

    def set_render_window(self):

        render_window = vtk.vtkRenderWindow()
        renderer, voxel_proj = self.set_render()
        render_window.AddRenderer(renderer)
        render_window.SetSize(self.img_map.shape[1], self.img_map.shape[0])

        if isinstance(voxel_proj, np.ndarray):
            plt.imshow(voxel_proj); plt.show()

        return render_window

    def draw3D(self):
        '''
        Visualize 3D models with their bounding boxes.
        '''
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window = self.set_render_window()
        render_window_interactor.SetRenderWindow(render_window)
        render_window.Render()
        render_window_interactor.Start()