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
from menge_gym.envs.utils.motion_model import ModifiedAckermannModel
from menge_gym.envs.utils.utils import DeviationWindow


class ModelPredictiveRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'ModelPredictiveRL'
        self.trainable = True
        self.multiagent_training = True
        self.kinematics = None
        self.motion_model = None  # type: Union[ModifiedAckermannModel, None]
        self.last_d_goal = None
        self.oscillation_window = None  # type: Union[DeviationWindow, None]
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
        self.sparse_speed_samples = 3
        self.sparse_rotation_samples = 4  # here: specifies samples per direction (left, right turn)
        self.action_group_index = []
        self.traj = None

    def configure(self, config, env_config=None, action_space=(None, None)):

        if hasattr(env_config, "robot_v_pref"):
            self.v_pref = env_config.robot_v_pref
        self.action_space, self.action_array = action_space
        num_speeds, num_angles = self.action_space.nvec
        self.action_indices = np.array(np.meshgrid(np.arange(num_speeds),
                                                   np.arange(num_angles))).T.reshape(-1, 2)
        action_group_index = np.empty_like(self.action_array[:, :, 0], dtype=int)
        # zero velocity actions get individual action group index
        action_group_index[0, :] = 0
        num_speed_samples = self.sparse_speed_samples
        speeds_per_group = int(np.ceil((num_speeds - 1) / (num_speed_samples - 1)))
        num_angle_samples = self.sparse_rotation_samples
        angles_per_group = int(np.ceil((num_angles - 1) / (2 * num_angle_samples)))
        median_angle = int(np.median(range(num_angles)))
        for s_sample in range(num_speed_samples - 1):
            # zero angle actions get individual action group index
            speed_idx = int(1 + s_sample * speeds_per_group)
            speed_group = 1 + s_sample * (2 * num_angle_samples + 1)
            action_group_index[speed_idx:, median_angle] = speed_group

            for a_sample in range(num_angle_samples):
                angle_idx_start = int(a_sample * angles_per_group)
                angle_idx_end = min(int((a_sample + 1) * angles_per_group), median_angle)
                action_group_index[speed_idx:, angle_idx_start:angle_idx_end] = speed_group + a_sample + 1
                angle_idx_start, angle_idx_end = num_angles - angle_idx_end, num_angles - angle_idx_start
                action_group_index[speed_idx:, angle_idx_start:angle_idx_end] = speed_group + num_angle_samples \
                                                                                + a_sample + 1
        self.action_group_index = action_group_index.reshape(-1)

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


        if isinstance(self.motion_model, ModifiedAckermannModel):
            state_predictor_motion_model = self.motion_model.copy()
            state_predictor_motion_model.use_tensor = True
        else:
            state_predictor_motion_model = None
        self.state_predictor.set_motion_model(state_predictor_motion_model)

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
            self.kinematics = config.robot.action_space.kinematics
        if hasattr(env_config, "robot_length"):
            robot_length = env_config.robot_length
        elif hasattr(config.robot, "length"):
            robot_length = config.robot.length
        else:
            if self.kinematics == "single_track":
                raise ValueError("robot length required in config for single_track motion_model")
            else:
                robot_length = None

        if hasattr(env_config, "robot_lf_ratio"):
            robot_lf_ratio = env_config.robot_lf_ratio
        elif hasattr(config.robot, "lf_ratio"):
            robot_lf_ratio = config.robot.lf_ratio
        else:
            if self.kinematics == "single_track":
                raise ValueError("lf_ratio required in config for single_track motion_model")
            else:
                robot_lf_ratio = None

        if self.kinematics == "single_track":
            self.motion_model = ModifiedAckermannModel(robot_length, robot_lf_ratio,
                                                       use_tensor=False, timestep=self.time_step)

        if hasattr(env_config, "robot_speed_sampling"):
            self.speed_sampling = env_config.robot_speed_sampling
        else:
            self.speed_sampling = config.robot.action_space.speed_sampling
        if hasattr(env_config, "robot_rotation_sampling"):
            self.rotation_sampling = env_config.robot_rotation_sampling
        else:
            self.rotation_sampling = config.robot.action_space.rotation_sampling
        if hasattr(env_config, "num_speeds"):
            self.speed_samples = env_config.num_speeds
        else:
            self.speed_samples = config.robot.action_space.speed_samples
        if hasattr(env_config, "num_angles"):
            self.rotation_samples = env_config.num_angles
        else:
            self.rotation_samples = config.robot.action_space.rotation_samples
        if hasattr(env_config, "rotation_constraint"):
            self.rotation_constraint = env_config.rotation_constraint
        else:
            self.rotation_constraint = config.robot.action_space.rotation_constraint

        if env_config is not None and hasattr(env_config, "reward"):
            self.reward = env_config.reward
        else:
            self.reward = config.reward
        self.oscillation_window = DeviationWindow(self.reward.oscillation_window_size)

    def set_device(self, device):
        self.device = device
        for model in self.model:
            model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_time_step(self, time_step):
        self.time_step = time_step
        self.state_predictor.set_time_step(time_step)
        if isinstance(self.motion_model, ModifiedAckermannModel):
            self.motion_model.setTimeStep(time_step)

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
        checkpoint = torch.load(file, map_location=self.device if self.device is not None else "cpu")
        self.load_state_dict(checkpoint)

    def set_last_d_goal(self, last_d_goal):
        self.last_d_goal = last_d_goal

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
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
            max_action_idx = self.action_space.sample()
            max_action = self.action_array[tuple(max_action_idx)]
        else:
            # max_action = None
            # max_value = float('-inf')
            # max_traj = None

            state_tensor = state.to_tensor(add_batch_size=True, device=self.device)

            if self.do_action_clip:
                action_indices_clipped = self.action_clip(state_tensor, self.oscillation_window, self.action_indices,
                                                          self.action_array, self.planning_width)
            else:
                action_indices_clipped = self.action_indices

            actions_clipped = self.action_array[action_indices_clipped[:, 0], action_indices_clipped[:, 1]]
            next_states = self.state_predictor(state_tensor, actions_clipped)
            reward_est = self.estimate_reward(state, actions_clipped, self.oscillation_window)

            d_goal = self.compute_d_goal(state_tensor)

            next_returns = [None] * len(actions_clipped)
            trajs = [None] * len(actions_clipped)

            for i, (action_idx, action) in enumerate(zip(action_indices_clipped, actions_clipped)):
                self.set_last_d_goal(d_goal)
                copied_osc_win = self.oscillation_window.copy()
                copied_osc_win(action[1])
                # TODO: multiprocessing here
                max_next_return, max_next_traj = self.V_planning(next_states[i], copied_osc_win, self.planning_depth,
                                                                 self.planning_width)
                next_returns[i] = float(max_next_return)
                trajs[i] = [(state_tensor, action, reward_est[i])] + max_next_traj

            next_returns = np.array(next_returns).reshape(reward_est.shape)
            values = reward_est + self.get_normalized_gamma() * next_returns

            max_index = np.argmax(values)
            max_action_idx = action_indices_clipped[max_index]
            max_action = actions_clipped[max_index]
            max_value = values[max_index]
            max_traj = [(state_tensor, max_action, reward_est[max_index])] + trajs[max_index]
            #
            # if max_action is None:
            #     raise ValueError('Value network is not well trained.')

        if self.phase == 'train':
            self.last_state = self.transform(state)
        else:
            self.traj = max_traj

        self.oscillation_window(max_action[1])

        return max_action_idx

    def action_clip(self, state, oscillation_window, action_indices, action_array, width, depth=1):
        
        # TODO: check d_last_goal for parallelized reward estimation
        d_goal = self.compute_d_goal(state)
        actions = action_array.reshape(-1, 2)
        next_states_est = self.state_predictor(state, actions)
        reward_est = self.estimate_reward(state, actions, oscillation_window)
        next_returns = []
        # TODO: try to parallelize
        for i, action in enumerate(actions):
            self.set_last_d_goal(d_goal)
            copied_osc_win = oscillation_window.copy()
            copied_osc_win(action[1])
            next_return, _ = self.V_planning(next_states_est[i], copied_osc_win, depth, width)
            next_returns.append(next_return)
        values = reward_est + self.get_normalized_gamma() * np.array(next_returns).reshape(reward_est.shape)
        values = values.reshape(-1)

        if self.sparse_search:
            # self.sparse_speed_samples = 2
            # search in a sparse grained action space
            added_groups = set()
            max_indices = np.argsort(np.array(values))[::-1]
            clipped_action_indices = []
            for index in max_indices:
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

    def V_planning(self, state, oscillation_window, depth, width):
        """ Plans n steps into future. Computes the value for the current state as well as the trajectories
        defined as a list of (state, action, reward) triples

        """

        current_state_value = self.value_estimator(state)
        if depth == 1:
            return current_state_value, [(state, None, None)]

        if self.do_action_clip:
            action_indices_clipped = self.action_clip(state, oscillation_window, self.action_indices,
                                                      self.action_array, width)
        else:
            action_indices_clipped = self.action_indices

        next_values = []
        trajs = []

        # TODO: check d_last_goal for parallelized reward estimation
        d_goal = self.compute_d_goal(state)
        actions = self.action_array[action_indices_clipped[:, 0], action_indices_clipped[:, 1]]
        next_states_est = self.state_predictor(state, actions)
        reward_est = self.estimate_reward(state, actions, oscillation_window)
        for i, (action_idx, action) in enumerate(zip(action_indices_clipped, actions)):
            self.set_last_d_goal(d_goal)
            updated_osc_win = oscillation_window.copy()
            updated_osc_win(action[1])
            next_value, next_traj = self.V_planning(next_states_est[i], updated_osc_win, depth - 1, self.planning_width)
            next_values.append(next_value)
            trajs.append([(state, action, reward_est[i])] + next_traj)

        next_values = np.array(next_values).reshape(reward_est.shape)
        returns = current_state_value / depth + (depth - 1) / depth * (self.get_normalized_gamma() *
                                                                       next_values + reward_est)

        max_index = np.argmax(returns)
        max_return = returns[max_index]
        max_traj = trajs[max_index]

        return max_return, max_traj

    def estimate_reward(self, state, actions, oscillation_window):
        """ If the time step is small enough, it's okay to model agent as linear movement during this period

        """
        if isinstance(state, list) or isinstance(state, tuple):
            state = tensor_to_joint_state(state)
        human_states = state.human_states
        robot_state = state.robot_state
        obstacles = state.obstacles

        # collision detection with other pedestrians

        # difference vector (distance) between robot and human (for all humans) at initial position
        d_0 = np.repeat((human_states.position - robot_state.position)[:, np.newaxis, :], len(actions), axis=1)
        actions = actions.reshape(-1, 2)
        num_actions = len(actions)
        # compute robot velocity vector
        action_velocities, action_angles = np.hsplit(actions, 2)
        if self.kinematics == 'holonomic':
            new_angles = robot_state.orientation + action_angles
            vx = action_velocities * np.cos(new_angles)
            vy = action_velocities * np.sin(new_angles)
            robot_velocities = np.column_stack((vx, vy))
        elif self.kinematics == 'single_track':
            self.motion_model.setPose(robot_state.position, robot_state.orientation)
            self.motion_model.computeNextPosition(actions)
            robot_velocities = self.motion_model.center_velocity_components
        else:
            raise NotImplementedError("other motion models not implemented yet")

        # velocity difference between robot and human (for all humans)
        human_velocities_expanded = np.repeat(human_states.velocity[:, np.newaxis, :], len(robot_velocities), axis=1)
        robot_velocities_expanded = np.repeat(robot_velocities[np.newaxis, :, :], len(human_states.velocity), axis=0)
        d_velocities = human_velocities_expanded - robot_velocities_expanded

        # difference vector (distance) between robot (for all actions) and human (for all humans)
        # at advanced position
        d_1 = d_0 + d_velocities * self.time_step

        # linear interpolation between d_0 and d_1 gives all differences (distances)
        # between initial and advanced position
        # --> point on this line that is closest to origin (0, 0) yields minimal distance
        origin = np.array([0, 0])

        # vectorized distance formula for point and line segment
        d_min2human = point_to_segment_dist(d_0, d_1, origin) \
                      - human_states.radius[:, np.newaxis] \
                      - robot_state.radius
        if d_min2human.size == 0:
            d_min2human = np.full(num_actions, np.inf)
        else:
            d_min2human = d_min2human.min(axis=0)

        # collision detection with obstacles
        if self.kinematics == 'holonomic':
            end_position = robot_state.position + robot_velocities * self.time_step
        elif self.kinematics == 'single_track':
            end_position = self.motion_model.pos_center
        else:
            raise NotImplementedError("other motion models not implemented yet")

        # TODO: not only check for circle collision (via radius) but also rectangle (spanned by radius + length)
        d_min2obs = point_to_segment_dist(np.repeat(robot_state.position, len(actions), axis=0),
                                          end_position, obstacles.position) - robot_state.radius
        if d_min2obs.size == 0:
            d_min2obs = np.full(num_actions, np.inf)
        else:
            d_min2obs = d_min2obs.min(0)

        oscillation_deviations = 1 / oscillation_window.size * (np.abs(actions[:, 1]
                                                                       - oscillation_window.get_last_item())
                                                                + oscillation_window.sum_except_one())
        oscillation_reward = - self.reward.oscillation_scale * oscillation_deviations

        d_goal = norm(end_position - robot_state.goal_position, axis=-1) - robot_state.radius + robot_state.goal_radius
        
        goal_approach_reward = 0
        if self.last_d_goal is not None:
            goal_approach_reward = self.reward.goal_approach_factor * (self.last_d_goal - d_goal)
        self.last_d_goal = d_goal

        rewards = np.zeros(num_actions)

        # collision with other human
        hum_collision_mask = d_min2human < 0
        # collision with obstacle
        obstacles_collision_mask = (d_min2obs < 0 & ~ hum_collision_mask)
        # reaching_goal
        goal_mask = (d_goal < 0 & ~ (hum_collision_mask | obstacles_collision_mask))
        # discomfortably close to humans
        discomfort_mask = ((d_min2human < self.reward.discomfort_dist)
                           & ~ (hum_collision_mask | obstacles_collision_mask | goal_mask))
        # or to obstacles
        clearance_mask = ((d_min2obs < self.reward.clearance_dist)
                          & ~ (hum_collision_mask | obstacles_collision_mask | goal_mask | discomfort_mask))

        rewards[hum_collision_mask] = self.reward.collision_penalty_crowd
        rewards[obstacles_collision_mask] = self.reward.collision_penalty_obs
        rewards[goal_mask] = self.reward.success_reward
        # adjust the reward based on FPS
        rewards[discomfort_mask] = (d_min2human[discomfort_mask] - self.reward.discomfort_dist) \
                                   * self.reward.discomfort_penalty_factor * self.time_step
        rewards[clearance_mask] = (d_min2obs[clearance_mask] - self.reward.clearance_dist) \
                                  * self.reward.clearance_penalty_factor * self.time_step

        rewards += oscillation_reward
        rewards += goal_approach_reward

        return rewards

    def transform(self, state) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take the JointState to tensors

        :param state: JointState(robot_state, human_states, obstacles)
        :return: Tuple of tensors (robot_state_tensor, human_states_tensor, obstacle_tensor)
        """
        if (isinstance(state, tuple) and isinstance(state[0], torch.Tensor)) or isinstance(state, torch.Tensor):
            return state
        else:
            return state.to_tensor(device=self.device)
