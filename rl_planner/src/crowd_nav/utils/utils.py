import numpy as np

eps = np.finfo(float).eps

def point_to_segment_dist(a: np.ndarray, b: np.ndarray, x: np.ndarray):
    """
    Calculate the shortest distance between point x and all line segments with two endpoints a[i], b[i]

    @:param: a: start point(s) of the line segment(s) (shape n x d)
    @:param: b: end point(s) of the line segment(s)   (shape n x d)
    @:param: x: point(s) to compute distance to       (shape m x d)

    :return: d_min: shortest distance(s) for all lines/points
    """
    if len(a.shape) == 1:
        a = a.reshape(1, len(a))
    if len(b.shape) == 1:
        b = b.reshape(1, len(b))
    if len(x.shape) == 1:
        x = x.reshape(1, len(x))

    assert a.shape == b.shape, "Number of start and end points must be equal"
    assert a.shape[1] == x.shape[1], "Line and Point need to have same number of coordinates " \
                                     "(line: {}, point: {}".format(a.shape[1], x.shape[1])
    assert x.shape[0] == 1 or a.shape[0] == 1 or x.shape[0] == a.shape[0], "You can only compare:\n" \
                                                                           "- a single point to multiple lines\n" \
                                                                           "- a single line to multiple points\n" \
                                                                           "- multiple points to multiple lines " \
                                                                           "on a 1-to-1 basis"

    ab = b - a
    ax = x - a
    bx = x - b
    # compute points on line through a and b that are closest to x
    # equivalent formula for single point and line: a + dot(ax, ab)/dot(ab, ab) * ab
    # stabilized by adding eps in denominator
    closest_points = a + ((ax * ab).sum(-1) / (ab * ab + eps).sum(-1)).reshape(-1, 1).repeat(2, axis=1) * ab

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

