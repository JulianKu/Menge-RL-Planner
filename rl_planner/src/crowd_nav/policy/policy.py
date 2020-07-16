import abc
import numpy as np
import torch
from menge_gym.envs.utils.state import JointState



class Policy(object):
    def __init__(self):
        """
        Base class for all policies, has an abstract method predict().
        """
        self.trainable = False
        self.phase = None
        self.model = None
        self.device = None
        self.last_state = None
        self.time_step = None
        # if agent is assumed to know the dynamics of real world
        self.env = None

    @abc.abstractmethod
    def configure(self, config):
        return

    def set_phase(self, phase):
        self.phase = phase

    def set_device(self, device):
        self.device = device

    def set_env(self, env):
        self.env = env

    def set_time_step(self, time_step):
        self.time_step = time_step

    def get_model(self):
        return self.model

    def save_model(self, file):
        torch.save(self.model.state_dict(), file)

    def load_model(self, file):
        self.model.load_state_dict(torch.load(file))

    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    @abc.abstractmethod
    def predict(self, state):
        """
        Policy takes state as input and output an action

        """
        return

    @staticmethod
    def reach_destination(state):
        robot_state = state.robot_state
        if np.linalg.norm(robot_state.position - robot_state.goal_position) \
                < robot_state.radius + robot_state.goal_radius:
            return True
        else:
            return False

    @staticmethod
    def compute_d_goal(state):
        if isinstance(state, JointState):
            robot_state = state.robot_state
            return np.linalg.norm(robot_state.position - robot_state.goal_position) \
                   + robot_state.radius[0] + robot_state.goal_radius[0]
        else:
            assert isinstance(state[0], torch.Tensor)
            robot_state = state[0].cpu().data.numpy().reshape(-1)
            return np.linalg.norm(robot_state[:2] - robot_state[7:9]) + robot_state[3] + robot_state[9]
