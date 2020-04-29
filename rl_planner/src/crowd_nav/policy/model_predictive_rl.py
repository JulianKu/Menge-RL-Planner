import logging
import torch
import numpy as np
from numpy.linalg import norm
from typing import Tuple
from crowd_nav.policy.policy import Policy
from crowd_nav.utils.utils import point_to_segment_dist
from crowd_nav.policy.state_predictor import StatePredictor, LinearStatePredictor
from crowd_nav.policy.graph_model import RGL
from crowd_nav.policy.value_estimator import ValueEstimator
from menge_gym.envs.utils.state import tensor_to_joint_state


class ModelPredictiveRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'ModelPredictiveRL'
        self.trainable = True
        self.multiagent_training = True
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.speed_sampling = None
        self.rotation_sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.action_space = None
        self.action_indices = None
        self.action_array = None
        self.rotation_constraint = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.robot_state_dim = 11
        self.human_state_dim = 7
        self.static_obs_dim = 2
        self.v_pref = 1
        self.reward = None
        self.share_graph_model = None
        self.value_estimator = None
        self.linear_state_predictor = None
        self.state_predictor = None
        self.planning_depth = None
        self.planning_width = None
        self.do_action_clip = None
        self.sparse_search = None
        self.sparse_speed_samples = 2
        self.sparse_rotation_samples = 8
        self.action_group_index = []
        self.traj = None

    def configure(self, config, env_config=None, action_space=(None, None)):

        if hasattr(env_config, "robot_v_pref"):
            self.v_pref = env_config.robot_v_pref
        self.action_space, self.action_array = action_space
        self.action_indices = np.array(np.meshgrid(np.arange(self.action_space.nvec[0]),
                                                   np.arange(self.action_space.nvec[1]))).T.reshape(-1, 2)

        self.set_common_parameters(config, env_config)
        self.planning_depth = config.model_predictive_rl.planning_depth
        self.do_action_clip = config.model_predictive_rl.do_action_clip
        if hasattr(config.model_predictive_rl, 'sparse_search'):
            self.sparse_search = config.model_predictive_rl.sparse_search
        self.planning_width = config.model_predictive_rl.planning_width
        self.share_graph_model = config.model_predictive_rl.share_graph_model
        self.linear_state_predictor = config.model_predictive_rl.linear_state_predictor

        if self.linear_state_predictor:
            self.state_predictor = LinearStatePredictor(config, self.time_step)
            graph_model = RGL(config, self.robot_state_dim, self.human_state_dim, self.static_obs_dim)
            self.value_estimator = ValueEstimator(config, graph_model)
            self.model = [graph_model, self.value_estimator.value_network]
        else:
            if self.share_graph_model:
                graph_model = RGL(config, self.robot_state_dim, self.human_state_dim, self.static_obs_dim)
                self.value_estimator = ValueEstimator(config, graph_model)
                self.state_predictor = StatePredictor(config, graph_model, self.time_step)
                self.model = [graph_model, self.value_estimator.value_network,
                              self.state_predictor.human_motion_predictor]
            else:
                graph_model1 = RGL(config, self.robot_state_dim, self.human_state_dim, self.static_obs_dim)
                self.value_estimator = ValueEstimator(config, graph_model1)
                graph_model2 = RGL(config, self.robot_state_dim, self.human_state_dim, self.static_obs_dim)
                self.state_predictor = StatePredictor(config, graph_model2, self.time_step)
                self.model = [graph_model1, graph_model2, self.value_estimator.value_network,
                              self.state_predictor.human_motion_predictor]

        logging.info('Planning depth: {}'.format(self.planning_depth))
        logging.info('Planning width: {}'.format(self.planning_width))
        logging.info('Sparse search: {}'.format(self.sparse_search))

        if self.planning_depth > 1 and not self.do_action_clip:
            logging.warning('Performing d-step planning without action space clipping!')

    def set_common_parameters(self, config, env_config=None):
        self.gamma = config.rl.gamma
        if hasattr(env_config, "robot_kinematics"):
            self.kinematics = env_config.robot_kinematics
        else:
            self.kinematics = config.action_space.kinematics
        if hasattr(env_config, "robot_speed_sampling"):
            self.speed_sampling = env_config.robot_speed_sampling
        else:
            self.speed_sampling = config.action_space.speed_sampling
        if hasattr(env_config, "robot_rotation_sampling"):
            self.rotation_sampling = env_config.robot_rotation_sampling
        else:
            self.rotation_sampling = config.action_space.rotation_sampling
        if hasattr(env_config, "num_speeds"):
            self.speed_samples = env_config.num_speeds
        else:
            self.speed_samples = config.action_space.speed_samples
        if hasattr(env_config, "num_angles"):
            self.rotation_samples = env_config.num_angles
        else:
            self.rotation_samples = config.action_space.rotation_samples
        if hasattr(env_config, "rotation_constraint"):
            self.rotation_constraint = env_config.rotation_constraint
        else:
            self.rotation_constraint = config.action_space.rotation_constraint
        self.reward = config.reward

    def set_device(self, device):
        self.device = device
        for model in self.model:
            model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_time_step(self, time_step):
        self.time_step = time_step
        self.state_predictor.time_step = time_step

    def get_normalized_gamma(self):
        return pow(self.gamma, self.time_step * self.v_pref)

    def get_model(self):
        return self.value_estimator

    def get_state_dict(self):
        if self.state_predictor.trainable:
            if self.share_graph_model:
                return {
                    'graph_model': self.value_estimator.graph_model.state_dict(),
                    'value_network': self.value_estimator.value_network.state_dict(),
                    'motion_predictor': self.state_predictor.human_motion_predictor.state_dict()
                }
            else:
                return {
                    'graph_model1': self.value_estimator.graph_model.state_dict(),
                    'graph_model2': self.state_predictor.graph_model.state_dict(),
                    'value_network': self.value_estimator.value_network.state_dict(),
                    'motion_predictor': self.state_predictor.human_motion_predictor.state_dict()
                }
        else:
            return {
                    'graph_model': self.value_estimator.graph_model.state_dict(),
                    'value_network': self.value_estimator.value_network.state_dict()
                }

    def get_traj(self):
        return self.traj

    def load_state_dict(self, state_dict):
        if self.state_predictor.trainable:
            if self.share_graph_model:
                self.value_estimator.graph_model.load_state_dict(state_dict['graph_model'])
            else:
                self.value_estimator.graph_model.load_state_dict(state_dict['graph_model1'])
                self.state_predictor.graph_model.load_state_dict(state_dict['graph_model2'])

            self.value_estimator.value_network.load_state_dict(state_dict['value_network'])
            self.state_predictor.human_motion_predictor.load_state_dict(state_dict['motion_predictor'])
        else:
            self.value_estimator.graph_model.load_state_dict(state_dict['graph_model'])
            self.value_estimator.value_network.load_state_dict(state_dict['value_network'])

    def save_model(self, file):
        torch.save(self.get_state_dict(), file)

    def load_model(self, file):
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint)

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """

        # TODO: transform state to tuple (maybe replace JointState class) and replace Action space
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.action_space is None:
            raise AttributeError('Action space needs to be defined in environment and set here via configure')

        if self.reach_destination(state):
            return np.array([0, 0], dtype=np.int32)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space.sample()
        else:
            max_action = None
            max_value = float('-inf')
            max_traj = None

            if self.do_action_clip:
                state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
                action_indices_clipped = self.action_clip(state_tensor, self.action_indices, self.action_array,
                                                          self.planning_width)
            else:
                action_indices_clipped = self.action_indices

            for action_idx in action_indices_clipped:
                action = self.action_array[tuple(action_idx)]
                state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
                next_state = self.state_predictor(state_tensor, action)
                max_next_return, max_next_traj = self.V_planning(next_state, self.planning_depth, self.planning_width)
                reward_est = self.estimate_reward(state, action)
                value = reward_est + self.get_normalized_gamma() * max_next_return
                if value > max_value:
                    max_value = value
                    max_action = action_idx
                    max_traj = [(state_tensor, action, reward_est)] + max_next_traj
            if max_action is None:
                raise ValueError('Value network is not well trained.')

        if self.phase == 'train':
            self.last_state = self.transform(state)
        else:
            self.traj = max_traj

        # TODO: check that max_action is np.array([idx_velocity, idx_angle]) for action_space
        return max_action

    def action_clip(self, state, action_indices, action_array, width, depth=1):
        values = []

        for action_idx in action_indices:
            action = action_array[tuple(action_idx)]
            next_state_est = self.state_predictor(state, action)
            next_return, _ = self.V_planning(next_state_est, depth, width)
            reward_est = self.estimate_reward(state, action)
            value = reward_est + self.get_normalized_gamma() * next_return
            values.append(value)

        if self.sparse_search:
            # self.sparse_speed_samples = 2
            # search in a sparse grained action space
            added_groups = set()
            max_indices = np.argsort(np.array(values))[::-1]
            clipped_action_indices = []
            for index in max_indices:
                # TODO: check if index indeed indexes action_indices right (index in range(102)
                if self.action_group_index[index] not in added_groups:
                    clipped_action_indices.append(action_indices[index])
                    added_groups.add(self.action_group_index[index])
                    if len(clipped_action_indices) == width:
                        break
        else:
            max_indices = np.argpartition(np.array(values), -width)[-width:]
            clipped_action_indices = [action_indices[i] for i in max_indices]

        # print(clipped_action_space)
        return clipped_action_indices

    def V_planning(self, state, depth, width):
        """ Plans n steps into future. Computes the value for the current state as well as the trajectories
        defined as a list of (state, action, reward) triples

        """

        current_state_value = self.value_estimator(state)
        if depth == 1:
            return current_state_value, [(state, None, None)]

        if self.do_action_clip:
            action_indices_clipped = self.action_clip(state, self.action_indices, self.action_array, width)
        else:
            action_indices_clipped = self.action_indices

        returns = []
        trajs = []

        for action_idx in action_indices_clipped:
            action = self.action_array[action_idx]
            next_state_est = self.state_predictor(state, action)
            reward_est = self.estimate_reward(state, action)
            next_value, next_traj = self.V_planning(next_state_est, depth - 1, self.planning_width)
            return_value = current_state_value / depth + (depth - 1) / depth * (self.get_normalized_gamma() * next_value + reward_est)

            returns.append(return_value)
            trajs.append([(state, action, reward_est)] + next_traj)

        max_index = np.argmax(returns)
        max_return = returns[max_index]
        max_traj = trajs[max_index]

        return max_return, max_traj

    def estimate_reward(self, state, action):
        """ If the time step is small enough, it's okay to model agent as linear movement during this period

        """
        if isinstance(state, list) or isinstance(state, tuple):
            state = tensor_to_joint_state(state)
        human_states = state.human_states
        robot_state = state.robot_state
        obstacles = state.obstacles

        # collision detection with other pedestrians

        # difference vector (distance) between robot and human (for all humans) at initial position
        d_0 = human_states.position - robot_state.position

        # compute robot velocity vector
        new_angle = robot_state.angle + action[1]
        if self.kinematics == 'holonomic':
            vx = action[0] * np.cos(new_angle)
            vy = action[0] * np.sin(new_angle)
        else:
            raise NotImplementedError("nonholonomic motion model not implemented yet")
        robot_velocity = np.array([vx, vy]).reshape(1, 2)

        # velocity difference between robot and human (for all humans)
        d_velocity = human_states.velocity - robot_velocity

        # difference vector (distance) between robot and human (for all humans) at advanced position
        d_1 = d_0 + d_velocity * self.time_step

        # linear interpolation between d_0 and d_1 gives all differences (distances)
        # between initial and advanced position
        # --> point on this line that is closest to origin (0, 0) yields minimal distance
        origin = np.array([0, 0])

        # vectorized distance formula for point and line segment
        d_min2human = point_to_segment_dist(d_0, d_1, origin) - human_states.radius - robot_state.radius
        d_min2human = np.min(d_min2human)

        # collision detection with obstacles
        if self.kinematics == 'holonomic':
            end_position = robot_state.position + robot_velocity * self.time_step
        else:
            raise NotImplementedError("nonholonomic motion model not implemented yet")
        d_min2obs = point_to_segment_dist(robot_state.position, end_position, obstacles.position) - robot_state.radius
        d_min2obs = np.min(d_min2obs)

        # check if reaching the goal
        reaching_goal = norm(end_position - robot_state.goal_position) < robot_state.radius + robot_state.goal_radius

        if d_min2human < 0:
            # collision with other human
            reward = self.reward.collision_penalty_crowd
        elif d_min2obs < 0:
            # collision with obstacle
            reward = self.reward.collision_penalty_obs
        elif reaching_goal:
            reward = self.reward.success_reward
        elif d_min2human < self.reward.discomfort_dist:
            # adjust the reward based on FPS
            reward = (d_min2human - self.reward.discomfort_dist) * self.reward.discomfort_penalty_factor \
                     * self.time_step
        elif d_min2obs < self.reward.clearance_dist:
            reward = (d_min2obs - self.reward.clearance_dist) * self.reward.clearance_penalty_factor * self.time_step
        else:
            reward = 0

        return reward

    def transform(self, state) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take the JointState to tensors

        :param state: JointState(robot_state, human_states, obstacles)
        :return: Tuple of tensors (robot_state_tensor, human_states_tensor, obstacle_tensor)
        """
        if isinstance(state, tuple):
            return state
        else:
            return state.to_tensor(device=self.device)
