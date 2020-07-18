import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import copy
from typing import Tuple, List, Union


def plot_rects_around_center(ax, centers, angles, width, height):
    ts = ax.transData
    lower_left_corners = np.column_stack((centers[:, 0] - width / 2, centers[:, 1] - height / 2))
    rect_list = []
    for corner, center, angle in zip(lower_left_corners, centers, angles):
        # perform rotation around center
        tr = patches.transforms.Affine2D().rotate_around(center[0], center[1], angle)
        rect_list.append(patches.Rectangle(corner, width, height, fill=False, color='black', transform=tr + ts))
    # add to plot
    ax.add_collection(PatchCollection(rect_list))


def rotate_around(p: Union[np.ndarray, torch.Tensor], c: Union[np.ndarray, torch.Tensor],
                  angle: Union[np.ndarray, torch.Tensor]):
    """
    rotate a point around another point by a given angle

    :param p: array/tensor, point to rotate
    :param c: array/tensor, center to rotate around
    :param angle: float, angle by which to rotate

    :return: p_new: array/tensor, rotated point (same type as p and c)
    """
    pshape = p.shape
    assert type(p) == type(c) and pshape == c.shape, "p and c both either need to be numpy arrays or torch tensors " \
                                                     "and of same shape"
    assert pshape[:2] == c.shape[:2] == angle.shape, "number of points to rotate must be consistent"

    p = p.reshape(-1, 2)
    c = c.reshape(-1, 2)
    angle = angle.reshape(-1)

    p1 = p - c
    ca, sa = np.cos(angle), np.sin(angle)
    rot_mat = ((ca, -sa), (sa, ca))

    if isinstance(p, torch.Tensor):
        rotation = torch.Tensor(rot_mat).permute(2, 0, 1)
        p_new = torch.einsum('ijk,ik->ij', rotation, p1) + c
    else:
        rotation = np.array(rot_mat).transpose((2, 0, 1))
        p_new = np.einsum('ijk,ik->ij', rotation, p1) + c

    p_new = p_new.reshape(pshape)

    return p_new


def map_angles(angles: Union[np.ndarray, torch.Tensor, float]):
    """
    maps angles to interval [-pi, pi)
    :param angles: float
    :return: mapped_angles: float
    """

    if isinstance(angles, (np.ndarray, torch.Tensor)):
        # map angle to range (0, 2 pi)
        angles = angles - (angles // (2 * np.pi)) * 2 * np.pi
        # map angles to range (-pi, pi)
        angles[angles > np.pi] -= 2 * np.pi
        # angle[angle <= -np.pi] += 2 * np.pi
    else:
        if angles > np.pi:
            angles -= 2 * np.pi
        elif angles <= -np.pi:
            angles += 2 * np.pi
    return angles


def perpendicular2d(a: Union[np.ndarray, torch.Tensor]):
    """
    Returns perpendicular vector(s) for 2D input vector(s)
    :param a: input array, either shape (2,) with (x,y) or shape (n,2) with (x,y) for each n
    :return: array with vectors perpendicular to a (same shape)
    """

    if isinstance(a, torch.Tensor):
        assert a.nelement() % 2 == 0, "Array needs to be 2D"
        perp_a = torch.empty_like(a)
    else:
        assert a.size % 2 == 0, "Array needs to be 2D"
        perp_a = np.empty_like(a)

    if len(a.shape) < 2:
        perp_a[0] = - a[1]
        perp_a[1] = a[0]
    else:
        perp_a[:, 0] = - a[:, 1]
        perp_a[:, 1] = a[:, 0]

    return perp_a


def angleBetween(v1: Union[np.ndarray, torch.Tensor, Tuple[float, float], List[float]],
                 v2: Union[np.ndarray, torch.Tensor, Tuple[float, float], List[float]]):
    """
    computes the angles between (arrays of) 2D vector pair v1 and v2
    """
    assert v1.shape[-1] == v2.shape[-1] == 2, "vectors need to be of the same length (2)"

    if len(v1.shape) < 2:
        v1_atan = np.arctan2(v1[1], v1[0])
    else:
        v1_atan = np.arctan2(v1[:, 1], v1[:, 0])
    if len(v2.shape) < 2:
        v2_atan = np.arctan2(v2[1], v2[0])
    else:
        v2_atan = np.arctan2(v2[:, 1], v2[:, 0])

    return v2_atan - v1_atan


class ModifiedAckermannModel(object):
    """
    A modified version of the Ackermann single track vehicle model.
    Model can deal with arbitrary steering angles (not limited to small angles) but does not include slip angle
    """

    EPS = np.finfo(float).eps

    def __init__(self, full_length: float, lf_ratio: float, use_tensor: bool = False,
                 centers: Union[np.ndarray, torch.Tensor, Tuple[float, float], List[float]] = None,
                 angles: Union[np.ndarray, torch.Tensor, float] = None, timestep: float = None):
        """
        :param full_length: float, length of the vehicle
        :param lf_ratio: float, ratio specifying position of the center of mass in relation to the rest of the vehicle
                (length_front_to_center / total_length)
        :param use_tensor: bool, if true uses torch.Tensor for attributes and arguments
        :param centers: (optional) tuple or array of x,y coordinates of the vehicle's center
                in relation to world coordinate system
        :param angles: (optional) float, angle of the vehicle base in relation to world coordinate system
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
        self.orientations = None  # type: Union[np.ndarray, torch.Tensor, None]
        self.pos_center = None  # type: Union[np.ndarray, torch.Tensor, None]
        self.pos_front_wheel = None  # type: Union[np.ndarray, torch.Tensor, None]
        self.pos_rear_wheel = None  # type: Union[np.ndarray, torch.Tensor, None]

        self.steering_angles = np.array((0.,))  # type: Union[np.ndarray, torch.Tensor, None]
        self.velocities = np.array((0.,))  # type: Union[np.ndarray, torch.Tensor, None]
        self.center_velocity_components = None  # type: Union[np.ndarray, torch.Tensor, None]
        self.steering_angles_global = self.orientations  # type: Union[np.ndarray, torch.Tensor, None]
        self.dir_steering = None  # type: Union[np.ndarray, torch.Tensor, None]
        self.turn_radii_center = None  # type: Union[np.ndarray, torch.Tensor, None]
        self.turn_radii_front = None  # type: Union[np.ndarray, torch.Tensor, None]
        self.turn_radii_rear = None  # type: Union[np.ndarray, torch.Tensor, None]
        self.centers_of_rotation = None  # type: Union[np.ndarray, torch.Tensor, None]

        if centers is not None and angles is not None:
            self.setPose(centers, angles)

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

    def setPose(self, centers: Union[np.ndarray, torch.Tensor, Tuple[float, float], List[float]], orientations: float):
        """
        set vehicle pose (center and orientation)

        :param centers: tuple or array of x,y coordinates
        :param orientations: float, angle of the vehicle base in relation to world coordinate system
        """
        if isinstance(centers, torch.Tensor) and self.use_tensor:
            centers = centers.cpu().view(-1, 2)
        elif not isinstance(centers, torch.Tensor) and self.use_tensor:
            centers = torch.Tensor(centers).view(-1, 2)
        elif isinstance(centers, np.ndarray) and not self.use_tensor:
            centers = centers.reshape(-1, 2)
        elif isinstance(centers, torch.Tensor) and not self.use_tensor:
            centers = centers.cpu().data.numpy().reshape(-1, 2)
        else:
            centers = np.array(centers).reshape(-1, 2)

        self.pos_center = centers

        if isinstance(orientations, torch.Tensor):
            orientations = orientations.cpu().view(-1, 1)
        elif isinstance(orientations, np.ndarray):
            orientations = orientations.reshape(-1, 1)
        else:
            orientations = np.array(orientations).reshape(-1, 1)

        orientations = map_angles(orientations)
        if self.use_tensor:
            if not isinstance(orientations, torch.Tensor):
                orientations = torch.Tensor(orientations)
            angle_legs = torch.cat((torch.sin(orientations), torch.cos(orientations)), -1)
        else:
            if isinstance(orientations, torch.Tensor):
                orientations = orientations.data.numpy()
            angle_legs = np.column_stack((np.cos(orientations), np.sin(orientations)))
        self.orientations = orientations

        self.pos_front_wheel = centers + angle_legs * self.length_front
        self.pos_rear_wheel = centers - angle_legs * self.length_rear

        # reset all other params
        self.steering_angles = np.array((0.,))  # type: Union[np.ndarray, torch.Tensor, None]
        self.velocities = np.array((0.,))  # type: Union[np.ndarray, torch.Tensor, None]
        self.center_velocity_components = None  # type: Union[np.ndarray, torch.Tensor, None]
        self.steering_angles_global = self.orientations  # type: Union[np.ndarray, torch.Tensor, None]
        self.dir_steering = None  # type: Union[np.ndarray, torch.Tensor, None]
        self.turn_radii_center = None  # type: Union[np.ndarray, torch.Tensor, None]
        self.turn_radii_front = None  # type: Union[np.ndarray, torch.Tensor, None]
        self.turn_radii_rear = None  # type: Union[np.ndarray, torch.Tensor, None]
        self.centers_of_rotation = None  # type: Union[np.ndarray, torch.Tensor, None]

    def computeCenterRotation(self, steering_angles: np.ndarray, mask: np.ndarray):
        """
        compute the center of rotation and radii for vehicle wheels and center for a singe track model

        :param steering_angles: np.array, front wheel angle in relation to vehicle base
        :param mask: np.array, boolean, True where steering_angle is not close to zero
        """

        assert not np.any(
            np.isclose(steering_angles[:, mask], 0)), "center of rotation can only be computed for actual steering angles"

        steering_angles_masked = steering_angles[:, mask]
        orientations = self.orientations
        abs_steering = np.abs(steering_angles_masked)
        R = np.linalg.norm((self.length / np.tan(abs_steering), self.length_rear), axis=0)

        dir_steering = np.sign(steering_angles_masked)
        tan_steering = np.tan(abs_steering)
        alphas = np.arctan2(self.length, tan_steering * self.length_rear)
        thetas = -np.pi + np.repeat(orientations, len(steering_angles_masked), axis=1) \
                 - dir_steering * alphas

        angle_legs = np.stack((np.cos(thetas), np.sin(thetas)), -1)
        if self.use_tensor:
            angle_legs = torch.Tensor(angle_legs)

        self.centers_of_rotation[:, mask] = np.tile(np.expand_dims(self.pos_center, 1),
                                                    (1, len(steering_angles_masked), 1)) \
                                            + angle_legs * np.expand_dims(R, axis=2)

        self.dir_steering[:, mask] = dir_steering
        self.steering_angles_global[:, mask] = np.repeat(orientations, len(steering_angles_masked),
                                                         axis=1) + steering_angles_masked
        self.turn_radii_center[:, mask] = R
        self.turn_radii_front[:, mask] = self.length / np.sin(abs_steering)
        self.turn_radii_rear[:, mask] = self.length / tan_steering

    def computeNextPosition(self, actions: Union[np.ndarray, Tuple[float, float], List[float]] = None) -> np.ndarray:
        """
        compute the next position of the vehicle and update all attributes

        :param actions: tuple or array of velocity and steering angle
        :return: new vehicle center position
        """

        assert self.timestep is not None, "to update the model's position setting a timestep by which to advance " \
                                          "is required first (use class method 'setTimeStep')"

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().data.numpy()

        velocities, steering_angles = np.hsplit(actions, 2) if actions is not None \
            else (self.velocities, self.steering_angles)
        velocities = velocities.reshape(1, -1)
        steering_angles = map_angles(steering_angles).reshape(1, -1)
        distances = velocities * self.timestep
        orientations = self.orientations
        self.steering_angles = np.repeat(steering_angles, len(orientations), 0)
        self.velocities = np.repeat(velocities, len(orientations), 0)

        self.turn_radii_center = np.empty((len(orientations), velocities.shape[1]))
        self.turn_radii_front = np.empty((len(orientations), velocities.shape[1]))
        self.turn_radii_rear = np.empty((len(orientations), velocities.shape[1]))
        self.centers_of_rotation = np.empty((len(orientations), velocities.shape[1], 2))
        self.dir_steering = np.empty((len(orientations), velocities.shape[1]))
        self.steering_angles_global = np.repeat(orientations, velocities.shape[1], axis=1)

        mask = np.isclose(steering_angles, 0).reshape(-1)

        # Handling driving straight by masking respective entries
        self.turn_radii_center[:, mask] = np.nan
        self.turn_radii_front[:, mask] = np.nan
        self.turn_radii_rear[:, mask] = np.nan
        self.centers_of_rotation[:, mask, :] = np.nan
        self.dir_steering[:, mask] = 0
        angle_legs = np.column_stack((np.cos(orientations), np.sin(orientations)))
        if self.use_tensor:
            angle_legs = torch.Tensor(angle_legs)
        new_pos_front_wheel = np.tile(np.expand_dims(self.pos_front_wheel, 1), (1, velocities.shape[1], 1))
        new_pos_front_wheel[:, mask] += np.einsum('ij,k->ikj', angle_legs, distances.reshape(-1)[mask])

        # Handling others by masking the opposite entries
        not_mask = np.bitwise_not(mask)
        self.computeCenterRotation(steering_angles, not_mask)
        angle_driven_front_wheel = self.dir_steering[:, not_mask] * distances[:, not_mask] \
                                   / self.turn_radii_front[:, not_mask]
        # new_center_velocity_components = np.empty((len(orientations), velocities.shape[1], 2))
        # new_center_velocity_components[:, mask] = angle_legs * velocities[mask]
        # new_center_velocity_components[:, not_mask] = velocities[not_mask] * (self.turn_radius_center[:, not_mask]
        #                                                                      / self.turn_radius_front[:, not_mask])

        # determine new position
        new_pos_front_wheel[:, not_mask] = rotate_around(new_pos_front_wheel[:, not_mask],
                                                         self.centers_of_rotation[:, not_mask], angle_driven_front_wheel)

        self.steering_angles_global[:, not_mask] += angle_driven_front_wheel
        orientations = np.repeat(orientations, velocities.shape[1], axis=1)
        orientations[:, not_mask] += angle_driven_front_wheel
        self.orientations = orientations

        angle_legs = np.stack((np.cos(orientations), np.sin(orientations)), -1)
        new_pos_center = new_pos_front_wheel - angle_legs * self.length_front
        self.center_velocity_components = (new_pos_center - np.tile(np.expand_dims(self.pos_center, 1),
                                                                    (1, velocities.shape[1], 1))) / self.timestep
        self.pos_rear_wheel = new_pos_front_wheel - angle_legs * self.length
        self.pos_front_wheel = new_pos_front_wheel
        self.pos_center = new_pos_center

        self.reshape_attributes()

        return self.pos_center

    def computeSteering(self, velocityComponentsCenter: Union[np.ndarray, torch.Tensor,
                                                              Tuple[float, float], List[float]]):

        orientationComponentsVehicle = np.stack((np.cos(self.orientations), np.sin(self.orientations))).reshape(-1, 2)

        if isinstance(velocityComponentsCenter, torch.Tensor) and self.use_tensor:
            velocityComponentsCenter = velocityComponentsCenter.cpu().view(-1, 2)
            orientationComponentsVehicle = torch.Tensor(orientationComponentsVehicle).view(-1, 2)
        elif not isinstance(velocityComponentsCenter, torch.Tensor) and self.use_tensor:
            velocityComponentsCenter = torch.Tensor(velocityComponentsCenter).view(-1, 2)
            orientationComponentsVehicle = torch.Tensor(orientationComponentsVehicle).view(-1, 2)
        elif isinstance(velocityComponentsCenter, np.ndarray) and not self.use_tensor:
            velocityComponentsCenter = velocityComponentsCenter.reshape(-1, 2)
        elif isinstance(velocityComponentsCenter, torch.Tensor) and not self.use_tensor:
            velocityComponentsCenter = velocityComponentsCenter.cpu().data.numpy().reshape(-1, 2)
        else:
            velocityComponentsCenter = np.array(velocityComponentsCenter).reshape(-1, 2)

        center_velocities = np.linalg.norm(velocityComponentsCenter, axis=1)

        angle_orientation_velocity = map_angles(angleBetween(orientationComponentsVehicle, velocityComponentsCenter))
        dir_steering = np.sign(angle_orientation_velocity)

        actions = np.empty((len(velocityComponentsCenter), 2))

        # check if velocity and orientation components are collinear -> steering angle = 0
        mask_no_turning = np.isclose(np.cross(velocityComponentsCenter, orientationComponentsVehicle), 0)
        actions[mask_no_turning] = np.column_stack((center_velocities[mask_no_turning], np.repeat(0., sum(mask_no_turning))))

        mask_valid_steering = abs(angle_orientation_velocity) < np.pi / 2
        mask_valid_turning = np.bitwise_and(np.bitwise_not(mask_no_turning), mask_valid_steering)

        # the perpendicular lines on orientationComponentsVehicle through pos_rear_wheel
        # and velocityComponentsCenter through center intersect in the center of rotation
        perpendicular_velocities = perpendicular2d(velocityComponentsCenter[mask_valid_turning])
        perpendicular_orientations = perpendicular2d(orientationComponentsVehicle)

        # compute intersection, i.e. center of rotation
        vec_center_rear = self.pos_rear_wheel - self.pos_center
        increments_rear = - np.dot(velocityComponentsCenter[mask_valid_turning], vec_center_rear.reshape(-1)) \
                          / np.dot(velocityComponentsCenter[mask_valid_turning], perpendicular_orientations.reshape(-1))
        increments_center = np.dot(orientationComponentsVehicle.reshape(-1), vec_center_rear.reshape(-1)) \
                            / np.dot(perpendicular_velocities, orientationComponentsVehicle.reshape(-1))
        vec_rear_center_of_rotation = increments_rear.reshape(-1, 1) * np.repeat(perpendicular_orientations,
                                                                  sum(mask_valid_turning), axis=0)
        vec_center_center_of_rotation = increments_center.reshape(-1, 1) * perpendicular_velocities

        centers_of_rotation = self.pos_rear_wheel + vec_rear_center_of_rotation
        centers_of_rotation_assert = self.pos_center + vec_center_center_of_rotation

        assert np.all(np.isclose(centers_of_rotation, centers_of_rotation_assert)), \
            "Computation should yield same results"

        # compute vector from center of rotation to front wheel
        vec_center_of_rotation_front = self.pos_front_wheel - centers_of_rotation
        # orientation of the front wheel is perpendicular to this vector
        orientations_front_wheel = dir_steering[mask_valid_turning].reshape(-1, 1) * perpendicular2d(vec_center_of_rotation_front)

        global_steering_angles = np.arctan2(orientations_front_wheel[:, 1], orientations_front_wheel[:, 0])

        steering_angles = map_angles(global_steering_angles - self.orientations)

        center_turn_radii = np.linalg.norm(vec_center_center_of_rotation, axis=1)
        front_turn_radii = np.linalg.norm(vec_center_of_rotation_front, axis=1)
        self.turn_radii_rear = np.linalg.norm(vec_rear_center_of_rotation, axis=1)
        self.turn_radii_center = center_turn_radii
        self.turn_radii_front = front_turn_radii

        # angular velocity is the same for center and front wheel with rigid body
        angular_velocities = center_velocities[mask_valid_turning] / center_turn_radii
        front_velocities = angular_velocities * front_turn_radii
        actions[mask_valid_turning] = np.column_stack((front_velocities, steering_angles.reshape(-1)))

        # limit steering angle to max. 90°, otherwise kinematics collapses
        mask_limited_steering = np.bitwise_not(mask_valid_steering)
        steering_angles = dir_steering[mask_limited_steering] * np.pi / 2
        velocities = center_velocities[mask_limited_steering]
        actions[mask_limited_steering] = np.column_stack((velocities, steering_angles))

        self.steering_angles = actions[:, 1]
        self.velocities = actions[:, 0]

        # self.reshape_attributes()

        return actions

    def reshape_attributes(self):
        self.pos_center = self.pos_center.reshape(-1, 2)
        self.pos_front_wheel = self.pos_front_wheel.reshape(-1, 2)
        self.pos_rear_wheel = self.pos_rear_wheel.reshape(-1, 2)
        self.velocities = self.velocities.reshape(-1, 1)
        self.orientations = self.orientations.reshape(-1, 1)
        self.steering_angles = self.steering_angles.reshape(-1, 1)
        self.steering_angles_global = self.steering_angles_global.reshape(-1, 1)
        self.center_velocity_components = self.center_velocity_components.reshape(-1, 2)
        self.dir_steering = self.dir_steering.reshape(-1, 1)
        self.turn_radii_center = self.turn_radii_center.reshape(-1, 1)
        self.turn_radii_front = self.turn_radii_front.reshape(-1, 1)
        self.turn_radii_rear = self.turn_radii_rear.reshape(-1, 1)
        self.centers_of_rotation = self.centers_of_rotation.reshape(-1, 2)

    def plot_vehicle(self, axes=None, show=True):
        if axes is None:
            _, axes = plt.subplots(1)
        axes.clear()
        plt.xlim([0, 20])
        plt.ylim([0, 15])
        plt.gca().set_aspect('equal')

        # plot vehicle line
        axes.plot([self.pos_rear_wheel[:, 0], self.pos_front_wheel[:, 0]],
                  [self.pos_rear_wheel[:, 1], self.pos_front_wheel[:, 1]],
                  'k-', lw=2)

        # plot front wheel
        plot_rects_around_center(axes, self.pos_front_wheel, self.orientations.ravel() + self.steering_angles.ravel(),
                                 0.4, 0.2)
        # plot rear wheel
        plot_rects_around_center(axes, self.pos_rear_wheel, self.orientations.ravel(),
                                 0.4, 0.2)

        # plot center of mass
        axes.plot(self.pos_center[:, 0], self.pos_center[:, 1], 'go')

        if self.centers_of_rotation is not None:
            mask = np.all(~np.isnan(self.centers_of_rotation), axis=1)
            # plot center of rotation
            axes.plot(self.centers_of_rotation[mask, 0], self.centers_of_rotation[mask, 1], 'ro')

            # plot radii
            axes.plot([self.pos_front_wheel[mask, 0], self.centers_of_rotation[mask, 0]],
                      [self.pos_front_wheel[mask, 1], self.centers_of_rotation[mask, 1]], 'r--')
            axes.plot([self.pos_rear_wheel[mask, 0], self.centers_of_rotation[mask, 0]],
                      [self.pos_rear_wheel[mask, 1], self.centers_of_rotation[mask, 1]], 'r--')
            axes.plot([self.pos_center[mask, 0], self.centers_of_rotation[mask, 0]],
                      [self.pos_center[mask, 1], self.centers_of_rotation[mask, 1]], 'g--')
        if self.center_velocity_components is not None:
            mask = np.all(~np.isnan(self.center_velocity_components), axis=1)
            axes.quiver(self.pos_center[mask, 0], self.pos_center[mask, 1],
                        self.center_velocity_components[mask, 0],
                        self.center_velocity_components[mask, 1], width=0.001, color='b')
        if show:
            plt.show()
