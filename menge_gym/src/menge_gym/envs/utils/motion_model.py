import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import copy
from typing import Tuple, List, Union


def plot_rect_around_center(ax, center, angle, width, height):
    ts = ax.transData
    # perform rotation around center
    lower_left_corner = (center[0] - width / 2, center[1] - height / 2)
    tr = patches.transforms.Affine2D().rotate_around(center[0], center[1], angle)
    rect = patches.Rectangle(lower_left_corner, width, height, fill=False, color='black', transform=tr + ts)
    # add to plot
    ax.add_patch(rect)


def rotate_around(p: Union[np.ndarray, torch.Tensor], c: Union[np.ndarray, torch.Tensor], angle: float):
    """
    rotate a point around another point by a given angle

    :param p: array/tensor, point to rotate
    :param c: array/tensor, center to rotate around
    :param angle: float, angle by which to rotate

    :return: p_new: array/tensor, rotated point (same type as p and c)
    """
    assert type(p) == type(c), "p and c both either need to be numpy arrays or torch tensors"

    p1 = p - c
    ca, sa = np.cos(angle), np.sin(angle)

    if isinstance(p, torch.Tensor):
        rotation = torch.Tensor(((ca, -sa), (sa, ca)))
        p_new = torch.mm(rotation, p1.view(2, -1)).view(-1) + c
    else:
        rotation = np.array(((ca, -sa), (sa, ca)))
        p_new = np.dot(rotation, p1) + c

    return p_new


def map_angle(angle: float):
    """
    maps angle to interval [-pi, pi)
    :param angle: float
    :return: mapped_angle: float
    """
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def perpendicular2d(a: Union[np.ndarray, torch.Tensor]):
    """
    Returns perpendicular vector(s) for 2D input vector(s)
    :param a: input array, either shape (2,) with (x,y) or shape (2,n) with (x,y) for each n
    :return: array with vectors perpendicular to a (same shape)
    """
    assert len(a) % 2 == 0, "Array needs to be 2D"

    if isinstance(a, torch.Tensor):
        perp_a = torch.empty_like(a)
    else:
        perp_a = np.empty_like(a)
    perp_a[0] = - a[1]
    perp_a[1] = a[0]

    return perp_a


def angleBetween(v1: Union[np.ndarray, torch.Tensor, Tuple[float, float], List[float]],
                 v2: Union[np.ndarray, torch.Tensor, Tuple[float, float], List[float]]):
    """
    computes the angle between two vectors v1 and v2
    """
    assert len(v1) == len(v2), "vectors need to be of the same length"

    if isinstance(v1, torch.Tensor) or isinstance(v2, torch.Tensor):
        return np.arctan2(*v2.flip(0)) - np.arctan2(*v1.flip(0))
    else:
        return np.arctan2(*v2[::-1]) - np.arctan2(*v1[::-1])


class ModifiedAckermannModel(object):
    """
    A modified version of the Ackermann single track vehicle model.
    Model can deal with arbitrary steering angles (not limited to small angles) but does not include slip angle
    """

    EPS = np.finfo(float).eps

    def __init__(self, full_length: float, lf_ratio: float, use_tensor: bool = False,
                 center: Union[np.ndarray, torch.Tensor, Tuple[float, float], List[float]] = None,
                 angle: float = None, timestep: float = None):
        """
        :param full_length: float, length of the vehicle
        :param lf_ratio: float, ratio specifying position of the center of mass in relation to the rest of the vehicle
                (length_front_to_center / total_length)
        :param use_tensor: bool, if true uses torch.Tensor for attributes and arguments
        :param center: (optional) tuple or array of x,y coordinates of the vehicle's center
                in relation to world coordinate system
        :param angle: (optional) float, angle of the vehicle base in relation to world coordinate system
        :param timestep: (optional) float, simulation time step
        """
        assert lf_ratio <= 1, 'ratio gives part of length that is in front of center of mass'

        self.use_tensor = use_tensor  # type: bool

        # vehicle geometrics
        self.length = full_length  # type: float
        self.length_front = full_length * lf_ratio  # type: float
        self.length_rear = full_length * (1 - lf_ratio)  # type: float
        self.lf_ratio = lf_ratio  # type: float

        # pose
        self.orientation = None  # type: Union[float, None]
        self.pos_center = None  # type: Union[np.ndarray, torch.Tensor, None]
        self.pos_front_wheel = None  # type: Union[np.ndarray, torch.Tensor, None]
        self.pos_rear_wheel = None  # type: Union[np.ndarray, torch.Tensor, None]

        if center is not None and angle is not None:
            self.setPose(center, angle)

        self.steering_angle = 0  # type: float
        self.velocity = 0  # type: float
        self.center_velocity_components = None  # type: Union[np.ndarray, torch.Tensor, None]
        self.steering_angle_global = self.orientation  # type: float
        self.dir_steering = None  # type: Union[int, None]
        self.turn_radius_center = None  # type: Union[float, None]
        self.turn_radius_front = None  # type: Union[float, None]
        self.turn_radius_rear = None  # type: Union[float, None]
        self.center_of_rotation = None  # type: Union[np.ndarray, torch.Tensor, None]

        self.timestep = None  # type: Union[float, None]
        if timestep:
            self.setTimeStep(timestep)

    def __copy__(self):
        model_class = self.__class__
        new_model = model_class.__new__(model_class)
        new_model.__dict__.update(self.__dict__)
        return new_model

    def copy(self):
        return copy(self)

    def setTimeStep(self, timestep: float):
        """
        set model timestep required for computeNextPosition
        :param timestep: float
        """

        assert timestep > 0, "only positive time steps are permitted"

        self.timestep = timestep

    def setPose(self, center: Union[np.ndarray, torch.Tensor, Tuple[float, float], List[float]], orientation: float):
        """
        set vehicle pose (center and orientation)

        :param center: tuple or array of x,y coordinates
        :param orientation: float, angle of the vehicle base in relation to world coordinate system
        """
        if isinstance(center, torch.Tensor) and self.use_tensor:
            center = center.view(-1).cpu()
        elif not isinstance(center, torch.Tensor) and self.use_tensor:
            center = torch.Tensor(center).view(-1)
        elif isinstance(center, np.ndarray) and not self.use_tensor:
            center = center.reshape(-1)
        elif isinstance(center, torch.Tensor) and not self.use_tensor:
            center = center.cpu().data.numpy().reshape(-1)
        else:
            center = np.array(center).reshape(-1)

        if isinstance(orientation, torch.Tensor):
            orientation = orientation.cpu()

        self.pos_center = center
        self.orientation = map_angle(orientation)
        if self.use_tensor:
            if isinstance(self.orientation, torch.Tensor):
                angle_legs = torch.Tensor((torch.sin(self.orientation), torch.cos(self.orientation))).view(-1).cpu()
            else:
                angle_legs = torch.Tensor((np.cos(self.orientation), np.sin(self.orientation))).view(-1)
        else:
            if isinstance(self.orientation, torch.Tensor):
                self.orientation = self.orientation.cpu().data.numpy()
            angle_legs = np.array((np.cos(self.orientation), np.sin(self.orientation))).reshape(-1)
        self.pos_front_wheel = center + angle_legs * self.length_front
        self.pos_rear_wheel = center - angle_legs * self.length_rear

    def computeCenterRotation(self, steering_angle: float):
        """
        compute the center of rotation and radii for vehicle wheels and center for a singe track model

        :param steering_angle: float, front wheel angle in relation to vehicle base
        """

        assert not np.isclose(steering_angle, 0), "center of rotation can only be computed for actual steering angles"

        abs_steering = abs(steering_angle)
        R = np.linalg.norm((self.length / np.tan(abs_steering), self.length_rear))

        dir_steering = np.sign(steering_angle)
        tan_steering = np.tan(abs_steering)
        alpha = np.arctan2(self.length, tan_steering * self.length_rear)
        theta = -np.pi + self.orientation - dir_steering * alpha

        angle_legs = (np.cos(theta), np.sin(theta))
        if self.use_tensor:
            angle_legs = torch.Tensor(angle_legs).view(-1)
        else:
            angle_legs = np.array(angle_legs).reshape(-1)
        self.center_of_rotation = self.pos_center + angle_legs * R

        self.dir_steering = dir_steering
        self.steering_angle_global = steering_angle + self.orientation
        self.turn_radius_center = R
        self.turn_radius_front = self.length / np.sin(abs_steering)
        self.turn_radius_rear = self.length / tan_steering

    def computeNextPosition(self, action: Union[np.ndarray, Tuple[float, float], List[float]] = None) -> np.ndarray:
        """
        compute the next position of the vehicle and update all attributes

        :param action: tuple or array of velocity and steering angle
        :return: new vehicle center position
        """

        assert self.timestep is not None, "to update the model's position setting a timestep by which to advance " \
                                          "is required first (use class method 'setTimeStep')"

        if isinstance(action, torch.Tensor):
            action = action.cpu()

        velocity, steering_angle = action if action is not None else (self.velocity, self.steering_angle)

        steering_angle = map_angle(steering_angle)
        self.steering_angle = steering_angle
        self.velocity = velocity
        distance = velocity * self.timestep
        orientation = self.orientation
        if np.isclose(steering_angle, 0):
            self.turn_radius_center = None
            self.turn_radius_front = None
            self.turn_radius_rear = None
            self.center_of_rotation = None

            self.dir_steering = 0
            self.steering_angle_global = orientation

            angle_legs = (np.cos(orientation), np.sin(orientation))
            if self.use_tensor:
                angle_legs = torch.Tensor(angle_legs).view(-1)
            else:
                angle_legs = np.array(angle_legs).reshape(-1)
            self.pos_front_wheel += angle_legs * distance
        else:
            self.computeCenterRotation(steering_angle)
            angle_driven_front_wheel = self.dir_steering * distance / self.turn_radius_front

            # determine new position
            self.pos_front_wheel = rotate_around(self.pos_front_wheel, self.center_of_rotation,
                                                 angle_driven_front_wheel)
            self.steering_angle_global += angle_driven_front_wheel
            orientation += angle_driven_front_wheel
            orientation = map_angle(orientation)

            angle_legs = (np.cos(orientation), np.sin(orientation))
            if self.use_tensor:
                angle_legs = torch.Tensor(angle_legs).view(-1)
            else:
                angle_legs = np.array(angle_legs).reshape(-1)
            self.orientation = orientation

        new_center = self.pos_front_wheel - angle_legs * self.length_front
        self.center_velocity_components = (new_center - self.pos_center) / self.timestep
        self.pos_center = new_center
        self.pos_rear_wheel = self.pos_front_wheel - angle_legs * self.length

        return self.pos_center

    def computeSteering(self, velocityComponentsCenter: Union[np.ndarray, torch.Tensor,
                                                              Tuple[float, float], List[float]]):

        orientationComponentsVehicle = (np.cos(self.orientation), np.sin(self.orientation))

        if isinstance(velocityComponentsCenter, torch.Tensor) and self.use_tensor:
            velocityComponentsCenter = velocityComponentsCenter.view(-1).cpu()
            orientationComponentsVehicle = torch.Tensor(orientationComponentsVehicle).view(-1)
        elif not isinstance(velocityComponentsCenter, torch.Tensor) and self.use_tensor:
            velocityComponentsCenter = torch.Tensor(velocityComponentsCenter).view(-1)
            orientationComponentsVehicle = torch.Tensor(orientationComponentsVehicle).view(-1)
        elif isinstance(velocityComponentsCenter, np.ndarray) and not self.use_tensor:
            velocityComponentsCenter = velocityComponentsCenter.reshape(-1)
            orientationComponentsVehicle = np.array(orientationComponentsVehicle).reshape(-1)
        elif isinstance(velocityComponentsCenter, torch.Tensor) and not self.use_tensor:
            velocityComponentsCenter = velocityComponentsCenter.cpu().data.numpy().reshape(-1)
            orientationComponentsVehicle = np.array(orientationComponentsVehicle)
        else:
            velocityComponentsCenter = np.array(velocityComponentsCenter).reshape(-1)
            orientationComponentsVehicle = np.array(orientationComponentsVehicle).reshape(-1)

        center_velocity = np.linalg.norm(velocityComponentsCenter)

        angle_orientation_velocity = map_angle(angleBetween(orientationComponentsVehicle, velocityComponentsCenter))
        dir_steering = np.sign(angle_orientation_velocity)

        # check if velocity and orientation components are collinear -> steering angle = 0
        if np.isclose(np.cross(velocityComponentsCenter, orientationComponentsVehicle), 0):
            steering_angle = 0
            front_velocity = center_velocity

        elif abs(angle_orientation_velocity) < np.pi / 2:
            # the perpendicular lines on orientationComponentsVehicle through pos_rear_wheel
            # and velocityComponentsCenter through center intersect in the center of rotation
            perpendicular_velocity = perpendicular2d(velocityComponentsCenter)
            perpendicular_orientation = perpendicular2d(orientationComponentsVehicle)

            # compute intersection, i.e. center of rotation
            vec_center_rear = self.pos_rear_wheel - self.pos_center
            increment_rear = - np.dot(velocityComponentsCenter, vec_center_rear) \
                             / np.dot(velocityComponentsCenter, perpendicular_orientation)
            increment_center = np.dot(orientationComponentsVehicle, vec_center_rear) \
                               / np.dot(orientationComponentsVehicle, perpendicular_velocity)
            vec_rear_center_of_rotation = increment_rear * perpendicular_orientation
            vec_center_center_of_rotation = increment_center * perpendicular_velocity

            center_of_rotation = self.pos_rear_wheel + vec_rear_center_of_rotation
            center_of_rotation_assert = self.pos_center + vec_center_center_of_rotation

            assert np.all(np.isclose(center_of_rotation, center_of_rotation_assert)), \
                "Computation should yield same results"

            # compute vector from center of rotation to front wheel
            vec_center_of_rotation_front = self.pos_front_wheel - center_of_rotation
            # orientation of the front wheel is perpendicular to this vector
            orientation_front_wheel = dir_steering * perpendicular2d(vec_center_of_rotation_front)
            if self.use_tensor:
                global_steering_angle = np.arctan2(*orientation_front_wheel.flip(0))
            else:
                global_steering_angle = np.arctan2(*orientation_front_wheel[::-1])
            steering_angle = map_angle(global_steering_angle - self.orientation)

            center_turn_radius = np.linalg.norm(vec_center_center_of_rotation)
            front_turn_radius = np.linalg.norm(vec_center_of_rotation_front)
            rear_turn_radius = np.linalg.norm(vec_rear_center_of_rotation)

            # angular velocity is the same for center and front wheel with rigid body
            angular_velocity = center_velocity / center_turn_radius
            front_velocity = angular_velocity * front_turn_radius

        else:
            # limit steering angle to max. 90°, otherwise kinematics collapses
            steering_angle = dir_steering * np.pi / 2
            front_velocity = center_velocity

        self.steering_angle = steering_angle
        self.velocity = front_velocity

        action = np.array((front_velocity, steering_angle))

        return action

    def plot_vehicle(self, axes=None, show=True):
        if axes is None:
            _, axes = plt.subplots(1)
        axes.clear()
        plt.xlim([0, 20])
        plt.ylim([0, 15])
        plt.gca().set_aspect('equal')

        # plot vehicle line
        axes.plot([self.pos_rear_wheel[0], self.pos_front_wheel[0]], [self.pos_rear_wheel[1], self.pos_front_wheel[1]],
                  'k-', lw=2)

        # plot front wheel
        plot_rect_around_center(axes, self.pos_front_wheel, self.orientation + self.steering_angle, 0.4, 0.2)
        # plot rear wheel
        plot_rect_around_center(axes, self.pos_rear_wheel, self.orientation, 0.4, 0.2)

        # plot center of mass
        axes.plot(*self.pos_center, 'go')

        if self.center_of_rotation is not None:
            # plot center of rotation
            axes.plot(*self.center_of_rotation, 'ro')

            # plot radii
            axes.plot([self.pos_front_wheel[0], self.center_of_rotation[0]],
                      [self.pos_front_wheel[1], self.center_of_rotation[1]], 'r--')
            axes.plot([self.pos_rear_wheel[0], self.center_of_rotation[0]],
                      [self.pos_rear_wheel[1], self.center_of_rotation[1]], 'r--')
            axes.plot([self.pos_center[0], self.center_of_rotation[0]], [self.pos_center[1],
                                                                         self.center_of_rotation[1]], 'g--')
        if self.center_velocity_components is not None:
            axes.arrow(self.pos_center[0], self.pos_center[1], self.center_velocity_components[0],
                       self.center_velocity_components[1], width=0.01, length_includes_head=True, color='b')
        if show:
            plt.show()
