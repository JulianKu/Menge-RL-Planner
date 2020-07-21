from copy import deepcopy
import numpy as np

EPS = np.finfo(float).eps


class DeviationWindow(object):

    def __init__(self, size):
        self.size = size
        self.index = 0
        self.content = size * [0]
        self.deviations = size * [0]

    def __str__(self):
        return str(self.content)

    def __call__(self, item):
        idx = self.index
        self.content[idx] = item
        self.deviations[idx] = item - self.content[(idx - 1) % self.size]
        self.index = (idx + 1) % self.size

    def __setitem__(self, key, value):
        self.content[key] = value

    def __getitem__(self, item):
        return self.content[item]

    def __abs__(self):
        return list(map(abs, self.deviations))

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.content)

    def get_last_item(self):
        return self.content[(self.index - 1) % self.size]

    def copy(self):
        return deepcopy(self)

    def sum(self, absolute=True):
        if absolute:
            return sum((map(abs, self.deviations)))
        else:
            return sum(self.deviations)

    def sum_except_one(self, absolute=True):
        if absolute:
            except_one = abs(self.deviations[self.index])
        else:
            except_one = self.deviations[self.index]
        return self.sum(absolute) - except_one

    def mean(self, absolute=True):
        return 1 / self.size * self.sum(absolute)

    def reset(self):
        self.index = 0
        self.content = self.size * [0]
        self.deviations = self.size * [0]


def point_to_segment_dist(a: np.ndarray, b: np.ndarray, x: np.ndarray):
    """
    Calculate the shortest distance between point x and all line segments with two endpoints a[i], b[i]

    @:param: a: start point(s) of the line segment(s) (shape ... x d)
    @:param: b: end point(s) of the line segment(s)   (shape ... x d)
    @:param: x: point(s) to compute distance to       (shape ... x d)

    :return: d_min: shortest distance(s) for all lines/points
    """

    a = a.reshape((1, -1)) if len(a.shape) <= 1 else a
    b = b.reshape((1, -1)) if len(b.shape) <= 1 else b
    x = x.reshape((1, -1)) if len(x.shape) <= 1 else x

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
    dist_a = np.linalg.norm(ax, axis=-1)
    # distance b - x
    dist_b = np.linalg.norm(bx, axis=-1)
    # distance closest point on line - x
    dist_c = np.linalg.norm(closest_points - x, axis=-1)

    # if closest point on line outside line segment -> shortest distance to one of the segments bounds
    d_min = np.minimum(dist_a, dist_b)
    # if closest point on line inside line segment -> shortest distance to closest point
    d_min[mask] = dist_c[mask]

    return d_min