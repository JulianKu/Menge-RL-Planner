import numpy as np
from typing import Union, Tuple
from torch import Tensor
from rospy import Time, Duration
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker

EPS = np.finfo(float).eps


def point_to_segment_dist(a: np.ndarray, b: np.ndarray, x: np.ndarray):
    """
    Calculate the shortest distance between point x and all line segments with two endpoints a[i], b[i]

    @:param: a: start point(s) of the line segment(s) (shape ... x d)
    @:param: b: end point(s) of the line segment(s)   (shape ... x d)
    @:param: x: point(s) to compute distance to       (shape ... x d)

    :return: d_min: shortest distance(s) for all lines/points
    """
    if len(a.shape) == 1:
        a = a.reshape(1, -1)
    if len(b.shape) == 1:
        b = b.reshape(1, -1)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)


    assert a.shape == b.shape, "Number of start and end points must be equal"
    assert a.shape[-1] == x.shape[-1], "Line and Point need to have same number of coordinates " \
                                       "(line: {}, point: {}".format(a.shape[1], x.shape[1])
    num_segments = len(a)
    num_points = len(x)

    if num_points != 1 and num_segments != 1 and num_points != num_segments:
        # No 1-to-1, 1-to-many, n-to-n but n-to-m comparison --> requires dim expansion
        a = np.repeat(a[np.newaxis, ...], num_points, axis=0)
        b = np.repeat(b[np.newaxis, ...], num_points, axis=0)
        x = np.repeat(x[:, np.newaxis, :], num_segments, axis=1)

    ab = b - a
    ax = x - a
    bx = x - b
    # compute points on line through a and b that are closest to x
    # equivalent formula for single point and line: a + dot(ax, ab)/dot(ab, ab) * ab
    # stabilized by adding EPS in denominator
    closest_points = a + ((ax * ab).sum(-1) / (ab * ab + EPS).sum(-1))[..., np.newaxis] * ab

    # determine whether closest_points lie on line segment
    ac = closest_points - a
    dot_product = (ab * ac).sum(-1)
    squared_norm_ab = (ab * ab).sum(-1)
    mask = np.bitwise_and(dot_product >= 0, dot_product <= squared_norm_ab)

    # distance a - x
    dist_a = np.linalg.norm(ax, axis=1)
    # distance b - x
    dist_b = np.linalg.norm(bx, axis=1)
    # distance closest point on line - x
    dist_c = np.linalg.norm(closest_points - x, axis=1)

    # if closest point on line outside line segment -> shortest distance to one of the segments bounds
    d_min = np.minimum(dist_a, dist_b)
    # if closest point on line inside line segment -> shortest distance to closest point
    d_min[mask] = dist_c[mask]

    return d_min


def mahalanobis_dist_nd(x: np.ndarray, y: np.ndarray):
    """
    compute the mahalanobis distance between two nd arrays where the last dimension gives the data dimensionality
    :param x: first array, shape (..., d)
    :param y: second array, shape (..., d)
    :return: mahalanobis distance between x and y
    """

    assert x.shape[-1] == y.shape[-1], "Last dimension has to be the same for both arrays"

    xx = x.reshape(-1, x.shape[-1])
    yy = y.reshape(-1, y.shape[-1])
    delta = xx - yy
    inv = np.linalg.inv(np.cov(np.vstack([xx, yy]).T)).T
    return_shape = x.shape[:-1] if len(x.shape) >= len(y.shape) else y.shape[:-1]

    return np.sqrt(np.einsum('nj,jk,nk->n', delta, inv, delta)).reshape(return_shape)


def human_poses_from_traj(traj_entry: Tuple[Tuple[Tensor,
                                                  Union[Tensor,
                                                       Tuple[Tensor, Tensor],
                                                       Tuple[Tensor, Tuple[Tensor, Tensor]]],
                                                  Tensor],
                                            np.ndarray,
                                            int]) -> np.ndarray:
    human_state = traj_entry[0][1]
    if isinstance(human_state, Tensor):
        human_tensor = human_state
    elif isinstance(human_state, tuple):
        if isinstance(human_state[1], Tensor):
            human_tensor = human_state[0]
        elif isinstance(human_state[1], tuple):
            human_tensor = human_state[1][0]
        else:
            raise NotImplementedError("Wrong trajectory format")
    else:
        raise NotImplementedError("Wrong trajectory format")
    human_pose = human_tensor.squeeze(0)[:, :2].cpu().data.numpy()

    return human_pose


def human_traj_ros_msg(traj_pose_list) -> MarkerArray:
    traj_marker_array = MarkerArray()

    for step, traj_pose in enumerate(traj_pose_list):
        if len(traj_pose) == 0:
            continue
        human_poses_marker = Marker()
        human_poses_marker.header.frame_id = "map"
        human_poses_marker.id = step + 1
        human_poses_marker.ns = "human_pred"
        human_poses_marker.type = human_poses_marker.SPHERE_LIST
        human_poses_marker.action = human_poses_marker.ADD
        human_poses_marker.lifetime = Duration(0.8)

        scale = 0.5 / (0.2 * step + 1.2)
        human_poses_marker.scale.x = scale
        human_poses_marker.scale.y = scale
        human_poses_marker.scale.z = 0.1
        human_poses_marker.pose.orientation.w = 1

        human_poses_marker.color.a = 0.6
        human_poses_marker.color.r = 1.0
        human_poses_marker.color.g = 0.1
        human_poses_marker.color.b = 0.0

        for human_pose in traj_pose:
            human_pnt = Point()
            human_pnt.x = human_pose[0]
            human_pnt.y = human_pose[1]
            human_poses_marker.points.append(human_pnt)

        traj_marker_array.markers.append(human_poses_marker)

    return traj_marker_array
