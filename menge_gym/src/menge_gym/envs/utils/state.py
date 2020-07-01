import torch
import numpy as np
from typing import Tuple, Union


class FullState(object):
    def __init__(self, state: np.ndarray):
        assert state.size % 11 == 0, "FullState needs to be composed of " \
                                     "p_x, p_y, angle_z, r, v_x, v_y, v_angular, g_x, g_y, r_goal and v_pref"
        state = state.reshape(-1, 11)
        self.state = state
        self.observable_state = state[:, :7]

        self.position = state[:, :2]
        self.orientation = state[:, 2]
        self.radius = state[:, 3]
        self.velocity = state[:, 4:6]
        self.angular_velocity = state[:, 6]
        self.goal_position = state[:, 7:9]
        self.goal_radius = state[:, 9]
        self.v_pref = state[:, 10]

    def __add__(self, other: np.ndarray):
        return other.reshape(-1, 11) + self.state

    def __str__(self):
        return ' '.join([str(x) for x in self.state])

    def __len__(self):
        return len(self.state)

    def to_tensor(self) -> torch.Tensor:
        return torch.Tensor(self.state)

    def get_observable_state(self):
        return ObservableState(self.state[:, :7])


class ObservableState(object):
    def __init__(self, state: np.ndarray):
        if len(state.shape) != 2 and (state.shape[1] != 8 or state.shape[1] != 7):
            if state.size % 7 == 0:
                state = state.reshape(-1, 7)
            elif state.size % 8 == 0:
                state = state.reshape(-1, 8)
            else:
                raise ValueError("ObservableState needs to be composed of "
                                 "p_x, p_y, angle_z, r, v_x, v_y, v_angular "
                                 "and optionally identifiers for each entitity")
        self.state = state[:, :7]

        self.position = state[:, :2]
        self.orientation = state[:, 2]
        self.radius = state[:, 3]
        self.velocity = state[:, 4:6]
        self.angular_velocity = state[:, 6]

        self.identifiers = state[:, 7] if state.shape[1] > 7 else None
        if self.identifiers is not None:
            assert len(np.unique(self.identifiers)) == len(self.identifiers), "Identifiers must be unique"
            self.identifiers = self.identifiers.astype(int).reshape(-1, 1)

    def __add__(self, other: np.ndarray):
        if other.size % 7 == 0:
            return other.reshape(-1, 7) + self.state
        elif other.size % 8 == 0:
            return other.reshape(-1, 8)[:, :7] + self.state
        else:
            raise ValueError("other needs to have same dimensions as state")

    def __str__(self):
        return ' '.join([str(x) for x in self.state])

    def __len__(self):
        return len(self.state)

    def set_identifiers(self, identifiers: np.ndarray):
        assert identifiers.size == len(self.state)
        self.identifiers = identifiers.astype(int).reshape(-1)

    def get_identifiers(self):
        return self.identifiers

    def to_tensor(self) -> Tuple[torch.Tensor, Union[None, torch.Tensor]]:
        if self.identifiers is None:
            return torch.Tensor(self.state), None
        else:
            return torch.Tensor(self.state), torch.from_numpy(self.identifiers)


class ObstacleState(object):
    def __init__(self, state: np.ndarray):
        assert state.size % 2 == 0, "ObstacleState needs to be composed of p_x and p_y"
        state = state.reshape(-1, 2)
        self.state = state
        self.position = state[:, :2]

    def __add__(self, other):
        return other.reshape(-1, 2) + self.state

    def __str__(self):
        return ' '.join([str(x) for x in self.state])

    def __len__(self):
        return len(self.state)

    def to_tensor(self) -> torch.Tensor:
        return torch.Tensor(self.state)


class JointState(object):
    def __init__(self, robot_state, human_states, obstacles):
        assert isinstance(robot_state, FullState)
        assert isinstance(human_states, ObservableState)
        assert isinstance(obstacles, ObstacleState)

        self.robot_state = robot_state
        self.human_states = human_states
        self.obstacles = obstacles

        self.whole_state = (robot_state, human_states, obstacles)

    def __getitem__(self, index):
        if index < 3:
            return self.whole_state[index]
        raise IndexError('Joint state only has 3 items (robot_state, human_states and obstacles)')

    def to_tensor(self, add_batch_size=False, device=None) \
            -> Tuple[torch.Tensor, Tuple[torch.Tensor, Union[None, torch.Tensor]], torch.Tensor]:

        robot_state_tensor = self.robot_state.to_tensor()
        human_states_tensor, human_identifiers = self.human_states.to_tensor()
        obstacle_tensor = self.obstacles.to_tensor()

        if add_batch_size:
            robot_state_tensor = robot_state_tensor.unsqueeze(0)
            human_states_tensor = human_states_tensor.unsqueeze(0)
            if human_identifiers is not None:
                human_identifiers = human_identifiers.unsqueeze(0)
            obstacle_tensor = obstacle_tensor.unsqueeze(0)

        if device is not None:
            robot_state_tensor = robot_state_tensor.to(device)
            human_states_tensor = human_states_tensor.to(device)
            if human_identifiers is not None:
                human_identifiers = human_identifiers.to(device)
            obstacle_tensor = obstacle_tensor.to(device)

        return robot_state_tensor, (human_states_tensor, human_identifiers), obstacle_tensor


def tensor_to_joint_state(state: Tuple[torch.Tensor,
                                       Union[Tuple[torch.Tensor, Union[None, torch.Tensor]],
                                             Tuple[torch.Tensor, Tuple[torch.Tensor, Union[None, torch.Tensor]]],
                                             torch.Tensor],
                                       Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]):
    robot_state, human_states, obstacles = state
    if isinstance(human_states, tuple):
        if isinstance(human_states[1], tuple):
            _, (human_states, human_identifiers) = human_states
        else:
            human_states, human_identifiers = human_states
    elif isinstance(human_states, torch.Tensor):
        human_identifiers = None
    else:
        raise NotImplementedError("human_states must be Tensor, tuple of Tensors (state, identifiers) "
                                  "or tuple of Tensor and tuple (mask, (state, identifiers))")

    if isinstance(obstacles, tuple):
        _, obstacles = obstacles
    elif not isinstance(obstacles, torch.Tensor):
        raise NotImplementedError("static_obs must be Tensor or tuple of Tensors (mask, state)")

    robot_state = robot_state.squeeze().data.numpy()
    robot_state = FullState(robot_state)
    human_states = human_states.squeeze(0).data.numpy()
    human_states = ObservableState(human_states)
    if human_identifiers is not None:
        human_identifiers = human_identifiers.squeeze(0).data.numpy()
        human_states.set_identifiers(human_identifiers)
    obstacle_states = obstacles.squeeze(0).data.numpy()
    obstacle_states = ObstacleState(obstacle_states)

    return JointState(robot_state, human_states, obstacle_states)
