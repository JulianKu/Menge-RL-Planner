import torch.nn as nn
from torch import sin, cos, from_numpy
import numpy as np
from crowd_nav.policy.helpers import mlp


class StatePredictor(nn.Module):
    def __init__(self, config, graph_model, time_step):
        """
        This function predicts the next state given the current state as input.
        It uses a graph model to encode the state into a latent space and predict each human's next state.
        """
        super().__init__()
        self.trainable = True
        self.kinematics = config.action_space.kinematics
        self.graph_model = graph_model
        self.human_motion_predictor = mlp(config.gcn.X_dim, config.model_predictive_rl.motion_predictor_dims)
        self.time_step = time_step
        self.motion_model = None

    def set_motion_model(self, motion_model):
        self.motion_model = motion_model

    def set_time_step(self, time_step: float):
        self.time_step = time_step
        if hasattr(self.motion_model, 'timestep'):
            self.motion_model.setTimeStep(time_step)

    def forward(self, state, actions, detach=False):
        """ Predict the next state tensor given current state as input.

        :return: tensor of shape (batch_size, # of agents, feature_size)
        """

        assert len(state[0].shape) == 3

        if isinstance(state[1], tuple):
            # human_states contain mask and zero padded state batch
            if isinstance(state[1][1], tuple):
                # human_states also contain identifiers for each human
                assert len(state[1][1][0].shape) == 3
                human_identifiers = state[1][1][1]
            else:
                assert len(state[1][1].shape) == 3
                human_identifiers = state[1][1]
        else:
            assert len(state[1].shape) == 3
            human_identifiers = None
        # state[2] = obstacles -> global position remains unchanged
        if isinstance(state[2], tuple):
            # obstacles contain mask and zero padded state batch
            assert len(state[2][1].shape) == 3
        else:
            assert len(state[2].shape) == 3

        state_embedding = self.graph_model(state)
        if detach:
            state_embedding = state_embedding.detach()
            if human_identifiers is not None:
                human_identifiers = human_identifiers.detach()
        if actions is None:
            # for training purpose
            next_robot_states = None
        else:
            next_robot_states = self.compute_next_state(state[0], actions)
        next_human_states = self.human_motion_predictor(state_embedding)[:, 1:, :]

        if next_robot_states is None:
            next_observations = [(next_robot_states, (next_human_states, human_identifiers), state[2])]
        else:
            # create observation for each next_robots_state
            next_observations = [(rob_state.squeeze(0), (next_human_states, human_identifiers), state[2])
                                 for rob_state in next_robot_states.split(1, dim=0)]
        return next_observations

    def compute_next_state(self, robot_state, actions):
        # currently it can not perform parallel computation
        # TODO: enable vectorized computation
        if robot_state.shape[0] != 1:
            raise NotImplementedError
        actions = actions.reshape(-1, 2)
        # px, py, theta, radius, vx, vy, theta_dot, gx, gy, g_r, v_pref
        next_states = robot_state.repeat(actions.shape[0], *[1] * len(robot_state.shape))

        action_velocities, action_angles = np.hsplit(actions, 2)
        if self.kinematics == 'holonomic':
            next_angles = next_states[..., 2]
            next_angles += action_angles.reshape(next_angles.shape)
            torch_velocities = from_numpy(action_velocities)
            vx = torch_velocities.view(next_angles.shape) * cos(next_angles)
            vy = torch_velocities.view(next_angles.shape) * sin(next_angles)
            next_states[..., 0] += vx * self.time_step
            next_states[..., 1] += vy * self.time_step
            next_states[..., 2] = next_angles
            next_states[..., 4] = vx
            next_states[..., 5] = vy
        elif self.kinematics == 'single_track':
            self.motion_model.setPose(robot_state[..., :2], robot_state[..., 2])
            self.motion_model.computeNextPosition(actions)
            next_states_shape = next_states.shape[:-1]
            next_states[..., :2] = self.motion_model.pos_center.clone().view(next_states_shape + (2,))
            next_states[..., 2] = self.motion_model.orientations.clone().view(next_states_shape)
            next_states[..., 4:6] = self.motion_model.center_velocity_components.clone().view(next_states_shape + (2,))
        else:
            raise NotImplementedError("other motion models not implemented yet")

        return next_states


class LinearStatePredictor(object):
    def __init__(self, config, time_step):
        """
        This function predicts the next state given the current state as input.
        It uses a graph model to encode the state into a latent space and predict each human's next state.
        """
        super().__init__()
        self.trainable = False
        self.kinematics = config.action_space.kinematics
        self.time_step = time_step
        self.motion_model = None

    def set_motion_model(self, motion_model):
        self.motion_model = motion_model

    def __call__(self, state, action):
        """ Predict the next state tensor given current state as input.

        :return: tensor of shape (batch_size, # of agents, feature_size)
        """
        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3
        # state[2] = obstacles -> global position remains unchanged
        assert len(state[2].shape) == 3

        next_robot_state = self.compute_next_state(state[0], action)
        next_human_states = self.linear_motion_approximator(state[1])

        next_observation = [next_robot_state, next_human_states, state[2]]
        return next_observation

    def compute_next_state(self, robot_state, action):
        # currently it can not perform parallel computation
        if robot_state.shape[0] != 1:
            raise NotImplementedError

        # px, py, vx, vy, radius, gx, gy, v_pref, theta
        next_state = robot_state.clone().squeeze()
        if self.kinematics == 'holonomic':
            # TODO: velocity update might need refinement (temporal dynamics)
            next_angle = next_state[2] + action[1]
            vx = action[0] * np.cos(next_angle)
            vy = action[0] * np.sin(next_angle)
            next_state[0] += vx * self.time_step
            next_state[1] += vy * self.time_step
            next_state[2] = next_angle
            next_state[4] = vx
            next_state[5] = vy
        elif self.kinematics == 'single_track':
            self.motion_model.setPose(next_state[:2], next_state[2])
            self.motion_model.computeNextPosition(action)
            next_state[:2] = self.motion_model.pos_center
            next_state[2] = self.motion_model.orientation
            next_state[4:6] = self.motion_model.center_velocity_components
        else:
            raise NotImplementedError("other motion models not implemented yet")

        return next_state.unsqueeze(0).unsqueeze(0)

    def linear_motion_approximator(self, human_states):
        """ approximate human states with linear motion, input shape : (batch_size, human_num, human_state_size)
        """
        # px, py, phi, radius, vx, vy, omega
        next_state = human_states.clone().squeeze()
        next_state[:, 0] = next_state[:, 0] + next_state[:, 4] * self.time_step
        next_state[:, 1] = next_state[:, 1] + next_state[:, 5] * self.time_step
        next_state[:, 2] = next_state[:, 2] + next_state[:, 6] * self.time_step

        return next_state.unsqueeze(0)
