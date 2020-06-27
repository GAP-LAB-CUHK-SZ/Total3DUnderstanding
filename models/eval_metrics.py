# Evaluation metrics
# author: ynie
# date: May, 2020

from shapely.geometry.polygon import Polygon

def get_iou_cuboid(cu1, cu2):
    """
        Calculate the Intersection over Union (IoU) of two 3D cuboid.

        Parameters
        ----------
        cu1 : numpy array, 8x3
        cu2 : numpy array, 8x3

        Returns
        -------
        float
            in [0, 1]
    """

    # 2D projection on the horizontal plane (z-x plane)
    polygon2D_1 = Polygon(
        [(cu1[0][2], cu1[0][0]), (cu1[1][2], cu1[1][0]), (cu1[2][2], cu1[2][0]), (cu1[3][2], cu1[3][0])])

    polygon2D_2 = Polygon(
        [(cu2[0][2], cu2[0][0]), (cu2[1][2], cu2[1][0]), (cu2[2][2], cu2[2][0]), (cu2[3][2], cu2[3][0])])

    # 2D intersection area of the two projections.
    intersect_2D = polygon2D_1.intersection(polygon2D_2).area

    # the volume of the intersection part of cu1 and cu2
    inter_vol = intersect_2D * max(0.0, min(cu1[0][1], cu2[0][1]) - max(cu1[4][1], cu2[4][1]))

    # the volume of cu1 and cu2
    vol1 = polygon2D_1.area * (cu1[0][1]-cu1[4][1])
    vol2 = polygon2D_2.area * (cu2[0][1]-cu2[4][1])

    # return 3D IoU
    return inter_vol / (vol1 + vol2 - inter_vol)