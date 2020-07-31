from copy import deepcopy
import numpy as np
from os import path
from .params import parseXML
import xml.etree.ElementTree as ElT

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


def randomize_scenario(scenario_root, scenario_dir, human_num=5, discomfort_dist=0.) -> np.ndarray:

    RADIUS = 4.
    CENTER = (15., 15.)

    # parse scene file
    scene_xml = scenario_root.get('scene')
    if not path.isabs(scene_xml):
        scene_xml = path.join(scenario_dir, scene_xml)
    scene_root = parseXML(scene_xml)

    # parse behavior file
    behavior_xml = scenario_root.get('behavior')
    if not path.isabs(behavior_xml):
        behavior_xml = path.join(scenario_dir, behavior_xml)
    behavior_root = parseXML(behavior_xml)

    # get robot radius
    r_radius = scene_root.find("AgentProfile/Common[@external='1']").get('r')
    if r_radius is not None:
        r_radius = float(r_radius)
    else:
        inherited_agt_profile = scene_root.find("AgentProfile/Common[@external='1']/..").get('inherits')
        r_radius = float(scene_root.find("AgentProfile[@name='{}']/Common".format(
            inherited_agt_profile)).get('r'))

    # get human radius and v_pref
    human_profile = scene_root.find("AgentProfile[@name='group1']/Common")
    h_radius = float(human_profile.get('r'))
    h_v_pref = float(human_profile.get('pref_speed'))


    # determine range for angle samples to distribute humans on circle with sufficient distance
    min_dist = h_radius + r_radius + discomfort_dist
    min_angle = min_angle_diff = np.arccos(1 - .5 * (min_dist**2 / RADIUS**2))
    max_angle = np.pi - min_angle_diff
    angle_range = max_angle - min_angle

    assert angle_range > (human_num - 1) * min_angle_diff, \
        "Not enough space to sample {} humans with specified radii and discomfort dist"

    # draw angle samples making sure minimum distance is kept (also to robot)
    p_x = np.array([0])
    p_y = np.array([-RADIUS])
    while len(p_x) < human_num + 1:
        # draw random and scale to range
        sample = np.random.random() * angle_range - np.pi + min_angle
        # sample noise
        x_noise = (np.random.random() - 0.5) * h_v_pref
        y_noise = (np.random.random() - 0.5) * h_v_pref
        # scale to circle
        x = RADIUS * np.cos(sample) + x_noise
        y = RADIUS * np.sin(sample) + y_noise
        # check distances
        distances = np.linalg.norm((p_x - x, p_y - y), axis=0)
        if np.all(distances > min_dist):
            p_x = np.append(p_x, x)
            p_y = np.append(p_y, y)
    # not change robot position
    p_x = p_x[1:]
    p_y = p_y[1:]

    # start and goal at opposite sides of the center
    g_x = CENTER[0] - p_x
    g_y = CENTER[1] - p_y
    p_x += CENTER[0]
    p_y += CENTER[1]

    # randomly flip start and goal
    random_switch_sides = np.random.choice([True, False], human_num)
    p_x, g_x = np.where(random_switch_sides, p_x, g_x), np.where(random_switch_sides, g_x, p_x)
    p_y, g_y = np.where(random_switch_sides, p_y, g_y), np.where(random_switch_sides, g_y, p_y)

    # write start and goal positions to xml tree
    for i, x in enumerate(p_x):
        agt = scene_root.find("AgentGroup/StateSelector[@name='Walk{}']/../Generator/Agent".format(i+1))
        agt.set('p_x', str(x))
        agt.set('p_y', str(p_y[i]))
        gol = behavior_root.find("GoalSet/Goal[@id='{}']".format(i+1))
        gol.set('x', str(g_x[i]))
        gol.set('y', str(g_y[i]))

    # extract robot goal from behavior file
    robot_goal = behavior_root.find("GoalSet[@id='0']/Goal[@id='0']")
    goal = np.array((float(robot_goal.get('x')), float(robot_goal.get('y')), 0.))

    # write modified scene and behavior files
    ElT.ElementTree(scene_root).write(scene_xml, xml_declaration=True, encoding='utf-8', method="xml")
    ElT.ElementTree(behavior_root).write(behavior_xml, xml_declaration=True, encoding='utf-8', method="xml")

    return goal
