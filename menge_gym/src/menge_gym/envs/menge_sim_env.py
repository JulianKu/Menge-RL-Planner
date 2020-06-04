#!/usr/bin/env python3

import gym
from gym import spaces
import numpy as np
from os import path
import rospy as rp
import xml.etree.ElementTree as ElT
from geometry_msgs.msg import PoseArray, PoseStamped, Twist
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import UInt8, Float32
from env_config.config import Config
from MengeMapParser import MengeMapParser
from .utils.ros import obstacle2array, marker2array, ROSHandle
from .utils.params import goal2array, get_robot_initial_position, parseXML
from .utils.info import *
from .utils.tracking import Sort, KalmanTracker
from .utils.format import format_array
from .utils.state import FullState, ObservableState, ObstacleState, JointState
from .utils.motion_model import ModifiedAckermannModel
from typing import List, Union


class MengeGym(gym.Env):
    """
    Custom gym environment for the Menge_ROS simulation
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MengeGym, self).__init__()

        self.config = Config()

        # Environment variables
        self.config.time_limit = None
        self.config.time_step = None
        self.config.randomize_attributes = None
        self.config.human_num = None
        self.config.robot_sensor_range = None

        # Simulation scenario variables
        self.config.scenario_xml = None
        self.scenario_root = None
        self.scene_xml = None
        self.behavior_xml = None
        self.initial_robot_pos = None
        self.goals_array = None

        # Robot variables
        self.config.robot_config = None
        self.config.robot_kinematics = None
        self.config.robot_speed_sampling = None
        self.config.robot_rotation_sampling = None
        self.config.num_speeds = None
        self.config.num_angles = None
        self.config.rotation_constraint = None
        self.config.robot_v_pref = None
        self.config.robot_visibility = None
        self.config.robot_length = None
        self.config.robot_lf_ratio = None
        self.robot_motion_model = None  # type: Union[ModifiedAckermannModel, None]
        self.goal = None
        self.robot_const_state = None

        # Reward variables
        self.config.success_reward = None
        self.config.collision_penalty_crowd = None
        self.config.discomfort_dist = None
        self.config.discomfort_penalty_factor = None
        self.config.collision_penalty_obs = None
        self.config.clearance_dist = None
        self.config.clearance_penalty_factor = None

        # Observation variables
        self._crowd_poses = []  # type: List[np.ndarray]
        self._robot_poses = []  # type: List[np.ndarray]
        self._static_obstacles = np.array([], dtype=float).reshape(-1, 2)
        self.rob_tracker = None
        self.ped_tracker = None
        self.observation = None

        # Action variables
        self._velocities = None
        self._angles = None
        self.action_array = None
        self._action = None  # type: Union[None, np.ndarray]

        # Schedule variables
        self.phase = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None

        # ROS
        self.roshandle = None
        self._sim_pid = None
        self._pub_step = None
        self._pub_cmd_vel = None
        self.global_time = None
        self._prev_time = None

        # Random Seed
        self.seed = None

    def configure(self, config, seed=None):

        self.config = Config()

        # Environment
        self.config.time_limit = config.env.time_limit
        self.config.time_step = config.env.time_step

        # Simulation
        if hasattr(config.sim, 'scenario') and path.isfile(config.sim.scenario):
            self.config.scenario_xml = config.sim.scenario
        # if no scenario provided, make new from image + parameters
        else:
            print("No scenario specified in config or invalid path.")
            key = input("Do you want to use an existing scenario? (Y/N)")
            if key.lower() == 'y':
                for _ in range(3):
                    scenario = input("Provide path to the scenario xml file here: ")
                    if not path.isfile(scenario) or not scenario.endswith(".xml"):
                        print("Not a valid path to an xml file")
                    else:
                        self.config.scenario_xml = scenario
                        break
                else:
                    # only executed if for loop not ended with break statement
                    raise ValueError("No valid scenario file provided")
            elif key.lower() == 'n':
                key = input("Do you want to create a new scenario instead? (Y/N)")
                if key != 'y':
                    raise ValueError("No scenario xml specified in config or not pointing to valid xml file")
                # get map file from input
                print("Scenario file can be automatically generated from image file that shows a map of the environment")
                for _ in range(3):
                    map_img = input("Provide path to the image here: ")
                    if not path.isfile(map_img):
                        print("Not a valid path to a file")
                    else:
                        break
                else:
                    # only executed if for loop not ended with break statement
                    raise ValueError("No valid image file provided")

                print("To translate the image into a valid map of the environment the map's resolution is also required")
                for _ in range(3):
                    resolution = input("Map resolution (meters per pixel) [m/px] = ")
                    try:
                        resolution = float(resolution)
                        break
                    except ValueError:
                        print("Resolution needs to be an integer or float")
                else:
                    # only executed if for loop not ended with break statement
                    raise ValueError("No valid resolution provided")
                
                self.config.robot_config = {}
                for param in vars(config.robot):
                    if param == 'radius':
                        self.config.robot_config['r'] = config.robot.radius
                    elif param == 'v_pref':
                        self.config.robot_config['pref_speed'] = config.robot.v_pref
                    elif param == 'sensor_range':
                        self.config.robot_config['range_max'] = config.robot.sensor_range
                    elif param == 'fov':
                        self.config.robot_config['start_angle'] = -config.robot.fov / 2
                        self.config.robot_config['end_angle'] = config.robot.fov / 2
                    elif param == 'sensor_resolution':
                        self.config.robot_config['increment'] = config.robot.sensor_resolution
                kwargs = {}
                if hasattr(config.sim, 'human_num'):
                    kwargs['num_agents'] = config.sim.human_num
                if hasattr(config.sim, 'randomize_attributes'):
                    # randomize humans' radius and preferred speed
                    kwargs['randomize_attributes'] = config.sim.randomize_attributes
                    kwargs['num_samples'] = 3
                if self.config.robot_config:
                    kwargs['robot_config'] = self.config.robot_config

                img_parser = MengeMapParser(map_img, resolution)
                img_parser.full_process(**kwargs)
                self.config.scenario_xml = img_parser.output['base']
            else:
                raise ValueError("No scenario xml specified in config or not pointing to valid xml file")

        # get more parameters from scenario xml
        self._initialize_from_scenario()

        # Random Seed
        self.seed = seed
        self._set_seed()

        # setup pedestrian tracker
        self.ped_tracker = Sort(max_age=2, min_hits=2, d_max=2*self.config.robot_v_pref*self.config.time_step)

        # sample first goal
        self.sample_goal(exclude_initial=True)

        # Reward
        self.config.success_reward = config.reward.success_reward
        self.config.collision_penalty_crowd = config.reward.collision_penalty_crowd
        self.config.discomfort_dist = config.reward.discomfort_dist
        self.config.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.config.collision_penalty_obs = config.reward.collision_penalty_obs
        self.config.clearance_dist = config.reward.clearance_dist
        self.config.clearance_penalty_factor = config.reward.clearance_dist_penalty_factor

        # Robot
        v_max = self.config.robot_v_pref
        self.config.rotation_constraint = config.robot.rotation_constraint
        self.config.num_speeds = config.robot.action_space.speed_samples
        self.config.num_angles = config.robot.action_space.rotation_samples
        self.config.robot_speed_sampling = config.robot.action_space.speed_sampling
        self.config.robot_rotation_sampling = config.robot.action_space.rotation_sampling
        self.config.robot_visibility = config.robot.visible

        self.config.robot_kinematics = config.robot.action_space.kinematics
        if hasattr(config.robot, 'length'):
            self.config.robot_length = config.robot.length
        if hasattr(config.robot, 'lf_ratio'):
            self.config.robot_lf_ratio = config.robot.lf_ratio
        # set motion model if available
        if self.config.robot_kinematics == 'single_track':
            self.robot_motion_model = ModifiedAckermannModel(self.config.robot_length,
                                                             self.config.robot_lf_ratio, timestep=self.config.time_step)

        # action space
        # from paper RGL for CrowdNav --> 6 speeds [0, v_pref] and 16 headings [0, 2*pi)
        if self.config.robot_speed_sampling == 'exponential':
            # exponentially distributed speeds (distributed between 0 and v_max)
            self.config.num_speeds += 1  # to include 0 as well
            self._velocities = np.geomspace(1, v_max + 1, self.config.num_speeds, endpoint=True) - 1
        elif self.config.robot_speed_sampling == 'linear':
            self.config.num_speeds += 1  # to include 0 as well
            self._velocities = np.linspace(0, v_max, self.config.num_speeds, endpoint=True)
        else:
            raise NotImplementedError

        if self.config.robot_rotation_sampling == 'linear':
            # linearly distributed angles
            # make num_angles odd to ensure null action (0 --> not changing steering)
            self.config.num_angles = self.config.num_angles // 2 * 2 + 1
            # angles between -45° und +45° in contrast to paper between -pi and +pi
            self._angles = np.linspace(-self.config.rotation_constraint, self.config.rotation_constraint,
                                       self.config.num_angles, endpoint=True)
        elif self.config.robot_rotation_sampling == 'exponential':
            min_angle_increment = 1             # (in deg)
            min_angle_increment *= np.pi / 180  # (in rad)
            positive_angles = np.geomspace(min_angle_increment, self.config.rotation_constraint,
                                           self.config.num_angles//2, endpoint=True)
            self._angles = np.concatenate((-positive_angles[::-1], [0], positive_angles))
        else:
            raise NotImplementedError

        self.action_array = np.array(np.meshgrid(self._velocities, self._angles)).T.reshape(
            self.config.num_speeds, self.config.num_angles, -1)
        self.action_space = spaces.MultiDiscrete([self.config.num_speeds, self.config.num_angles])

        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': config.env.train_size, 'val': config.env.val_size,
                          'test': config.env.test_size}
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

    def _initialize_from_scenario(self):
        scenario_xml = self.config.scenario_xml
        scenario_dir = path.split(scenario_xml)[0]
        self.scenario_root = parseXML(scenario_xml)

        scene_xml = self.scenario_root.get('scene')
        if not path.isabs(scene_xml):
            self.scene_xml = path.join(scenario_dir, scene_xml)
        else:
            self.scene_xml = scene_xml
        assert path.isfile(self.scene_xml), 'Scene file specified in scenario_xml non-existent'

        scene_root = parseXML(self.scene_xml)
        # extract robot radius from scene_xml file
        self.config.robot_radius = float(scene_root.find("AgentProfile/Common[@external='1']").get('r'))
        # extract number of humans from scene_xml file
        self.config.human_num = len(scene_root.findall("AgentGroup/Generator/Agent")) - 1
        # extract robot pref_speed from scene_xml file
        robot_v_pref = scene_root.find("AgentProfile/Common[@external='1']").get('pref_speed')
        if robot_v_pref is not None:
            self.config.robot_v_pref = float(robot_v_pref)
        else:
            # pref_speed is inherited from other AgentProfile
            inherited_agt_profile = scene_root.find("AgentProfile/Common[@external='1']/..").get('inherits')
            self.config.robot_v_pref = float(scene_root.find("AgentProfile[@name='{}']/Common"
                                                             .format(inherited_agt_profile)).get('pref_speed'))

        behavior_xml = self.scenario_root.get('behavior')
        if not path.isabs(behavior_xml):
            self.behavior_xml = path.join(scenario_dir, behavior_xml)
        assert path.isfile(self.behavior_xml), 'Behavior file specified in scenario_xml non-existent'

        # extract goal set from behavior file
        behavior_root = parseXML(self.behavior_xml)
        goals = behavior_root.findall("GoalSet/Goal")
        self.goals_array = np.array([goal2array(goal) for goal in goals])

    def _set_seed(self):
        """set random seed in numpy, for simulation and write to scenario xml"""

        if self.seed is not None:

            np.random.seed(self.seed)

            self.scenario_root.set("random", str(self.seed))
            scenario_tree = ElT.ElementTree(self.scenario_root)
            scenario_tree.write(self.config.scenario_xml, xml_declaration=True, encoding='utf-8', method="xml")

    def sample_goal(self, exclude_initial: bool = False):
        """
        sample goal from available goals and set "goal" attribute accordingly

        :param exclude_initial:     bool, if True exclude a goal from sampling
                                          if the robot's initial position lies within this goal
        """
        goals_array = self.goals_array
        if exclude_initial:
            if self.initial_robot_pos is None:
                self.initial_robot_pos = get_robot_initial_position(self.scene_xml)
            # if initial robot position falls within goal, exclude this goal from sampling
            dist_rob_goals = np.linalg.norm(goals_array[:, :2] - self.initial_robot_pos, axis=1) - goals_array[:, 2] \
                             - self.config.robot_radius
            # mask out respective goal(s)
            mask = dist_rob_goals > 0
            goals_array = goals_array[mask]

        self.goal = goals_array[np.random.randint(len(goals_array))]
        # set constant part of the robot's state
        self.robot_const_state = np.concatenate((self.goal, [self.config.robot_v_pref])).reshape(-1, 4)

    def setup_ros_connection(self):
        rp.loginfo("Initializing ROS")
        self.roshandle = ROSHandle()

        # TODO: rviz config not loaded properly (workaround: start rviz separately via launch file etc.)
        # visualization = True
        # if visualization:
        #     # Get rviz configuration file from "menge_vis" package
        #     rviz_path = path.join(path.join(rospkg.RosPack().get_path("menge_vis"), "rviz"), "menge_ros.rviz")
        #     # Start rviz rosnode
        #     self.roshandle.start_rosnode('rviz', 'rviz', launch_cli_args={"d": rviz_path})

        rp.on_shutdown(self.close)

        # simulation controls
        rp.logdebug("Set up publishers and provided services")
        rp.init_node('MengeSimEnv', log_level=rp.DEBUG)
        self._pub_step = rp.Publisher('step', UInt8, queue_size=1, tcp_nodelay=True)
        self._pub_cmd_vel = rp.Publisher('cmd_vel', Twist, queue_size=1, tcp_nodelay=True, latch=True)

    def _crowd_expansion_callback(self, msg: MarkerArray):
        rp.logdebug('Crowd Expansion subscriber callback called')
        # transform MarkerArray message to numpy array (skip first marker as this is the "delete markers" action)
        marker_array = np.array(list(map(marker2array, msg.markers[1:])))
        self._crowd_poses.append(marker_array.reshape(-1, 4))
        rp.logdebug('Crowd callback done')

    def _static_obstacle_callback(self, msg: PoseArray):
        rp.logdebug('Static Obstacle subscriber callback called')
        # transform PoseArray message to numpy array
        self._static_obstacles = np.array(list(map(obstacle2array, msg.poses))).reshape(-1, 2)
        rp.logdebug('Static Obs callback done')

    def _robot_pose_callback(self, msg: PoseStamped):
        rp.logdebug('Robot Pose subscriber callback called')
        # extract 2D pose and orientation from message
        robot_pose = msg.pose
        x = robot_pose.position.x
        y = robot_pose.position.y
        # orientation quaternion
        q_z = robot_pose.orientation.z
        q_w = robot_pose.orientation.w
        phi = np.arctan2(2 * q_w * q_z, 1 - 2 * q_z * q_z)
        if phi < -np.pi:
            phi += 2*np.pi
        elif phi > np.pi:
            phi -= 2*np.pi

        # update list of robot poses + pointer to current position
        self._robot_poses.append(np.array([x, y, phi, self.config.robot_radius]).reshape(-1, 4))
        rp.logdebug('Robot Pose callback done')

    def _sim_time_callback(self, msg: Float32):
        rp.logdebug('Time message received')
        self.global_time = msg.data

    def _cmd_vel_pub(self):

        action = self._action
        if action is not None and hasattr(self.observation, 'robot_state'):
            robot_state = self.observation.robot_state
            velocity_action = self._velocities[action[0]]
            angle_action = self._angles[action[1]]

            if isinstance(self.robot_motion_model, ModifiedAckermannModel):
                # transform front wheel velocity and steering angle into center velocity and center velocity angle
                self.robot_motion_model.setPose(robot_state.position, robot_state.orientation[0])
                self.robot_motion_model.computeNextPosition(np.array((velocity_action, angle_action)))
                center_velocity_components = self.robot_motion_model.center_velocity_components
                velocity_action = np.linalg.norm(center_velocity_components)
                angle_action = np.arctan2(center_velocity_components[1], center_velocity_components[0])

            # in menge_ros the published angle defines the relative angle based on the agent's orientation
            angle_action -= robot_state.orientation[0]

            rp.logdebug('Setting action with vel={:.2f} and steering angle={:.2f}'.format(velocity_action,
                                                                                          angle_action))
        else:
            velocity_action = 0
            angle_action = 0

        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = velocity_action
        cmd_vel_msg.angular.z = angle_action
        self._pub_cmd_vel.publish(cmd_vel_msg)

    def step(self, action: np.ndarray):
        rp.logdebug("Performing step in the environment")

        self._take_action(action)

        reward, done, info = self._get_reward_done_info()

        if isinstance(info, Timeout) and not (self._robot_poses and self._crowd_poses):
            # in Timeout cases, crowd_poses and robot_poses might be missing
            self._crowd_poses.append(np.array([], dtype=float).reshape(-1, 4))
            self._robot_poses.append(np.array([], dtype=float).reshape(-1, 4))

        # in first iteration, initialize Kalman Tracker for robot
        if not self.rob_tracker:
            self.rob_tracker = KalmanTracker(self._robot_poses[0])

        # update velocities
        for (robot_pose, crowd_pose) in zip(self._robot_poses, self._crowd_poses):
            rp.logdebug("Robot Pose (Shape %r):\n %r" % (robot_pose.shape, robot_pose))
            rp.logdebug("Crowd Pose (Shape %r):\n %r" % (crowd_pose.shape, crowd_pose))

            ped_trackers = self.ped_tracker.update(crowd_pose)
            self.rob_tracker.predict()
            if np.any(robot_pose):
                self.rob_tracker.update(robot_pose)
        rob_tracker = self.rob_tracker.get_state()
        pedestrian_state = ObservableState(ped_trackers[ped_trackers[:, -1].argsort()])
        robot_state = FullState(np.concatenate((rob_tracker, self.robot_const_state), axis=1))
        obstacle_state = ObstacleState(self._static_obstacles)
        self.observation = JointState(robot_state, pedestrian_state, obstacle_state)
        ob = self.observation

        # reset last poses
        self._crowd_poses = []
        self._robot_poses = []
        self._static_obstacles = np.array([], dtype=float).reshape(-1, 2)

        return ob, reward, done, info

    def _take_action(self, action: np.ndarray):
        """
        execute one time step within the environment
        """
        rp.logdebug("Taking action")

        self._action = action

        # advance simulation by n steps (n=1)
        n_steps = 1
        target_time = self._prev_time + n_steps * self.config.time_step

        # wait for response from simulation
        counter = 0

        self._cmd_vel_pub()
        while not self._crowd_poses or not self._robot_poses:

            # handle simulation reaching time limit
            if self.global_time + self.config.time_step > self.config.time_limit:
                rp.logdebug("Simulation reached time limit")
                break

            # advance simulation
            time_diff = target_time - self.global_time
            if time_diff > 0:
                rp.logdebug('Simulation not done yet')
                
                rp.logdebug("Publishing {} steps".format(n_steps))
                self._pub_step.publish(UInt8(data=n_steps))

            rp.logdebug('Current Sim Time %.3f, previous sim time %.3f' % (self.global_time, self._prev_time))
            rp.logdebug('#Crowd %d, #Rob %d' %
                        (len(self._crowd_poses), len(self._robot_poses)))
            counter += 1
            rp.logdebug('Counter={}'.format(counter))
            if counter >= 10:
                # raise TimeoutError("Simulator node not responding")
                rp.logerr("Timeout reached, setting empty poses")
                self._crowd_poses.append(np.array([], dtype=float).reshape(-1, 4))
                self._robot_poses.append(np.array([], dtype=float).reshape(-1, 4))

        rp.logdebug('Simulation step(s) done')
        rp.logdebug('Current Sim Time %.3f, previous sim time %.3f' % (self.global_time, self._prev_time))
        self._prev_time = self.global_time
        self._action = None

    def _get_reward_done_info(self) -> (float, bool, object):
        """
        compute reward and other information from current state

        :return:
            reward, done, info
        """

        if self.global_time + self.config.time_step > self.config.time_limit:
            # handle reward, etc. for simulation reaching time limit
            reward = 0
            done = True
            info = Timeout()
            return reward, done, info
        else:
            # crowd_pose = [x, y, omega, r]
            recent_crowd_pose = self._crowd_poses[-1]

            # obstacle_position = [x, y]
            obstacle_position = self._static_obstacles

            # robot_pose = [x, y, omega]
            recent_robot_pose = self._robot_poses[-1]

            robot_radius = self.config.robot_radius
            goal = self.goal

            if np.any(recent_crowd_pose) and np.any(recent_robot_pose):
                crowd_distances = np.linalg.norm(recent_crowd_pose[:, :2] - recent_robot_pose[:, :2], axis=1)
                crowd_distances -= recent_crowd_pose[:, -1]
                crowd_distances -= robot_radius
            else:
                crowd_distances = np.array([])

            if np.any(obstacle_position) and np.any(recent_robot_pose):
                obstacle_distances = np.linalg.norm(obstacle_position - recent_robot_pose[:, :2], axis=1)
                obstacle_distances -= robot_radius
            else:
                obstacle_distances = np.array([])

            # compute distance to closest pedestrian
            if crowd_distances.size == 0:
                # if no pedestrian, set to infinity
                d_min_crowd = np.inf
            else:
                d_min_crowd = crowd_distances.min()

            # compute distance to closest static obstacle
            if obstacle_distances.size == 0:
                # if no obstacles, set to infinity
                d_min_obstacle = np.inf
            else:
                d_min_obstacle = obstacle_distances.min()

            if np.any(recent_robot_pose):
                d_goal = np.linalg.norm(recent_robot_pose[:, :2] - goal[:2]) - robot_radius - goal[-1]
            else:
                d_goal = np.inf

            # collision with crowd
            if d_min_crowd < 0:
                reward = self.config.collision_penalty_crowd
                done = True
                info = Collision('Crowd')
            # collision with obstacle
            elif d_min_obstacle < 0:
                reward = self.config.collision_penalty_obs
                done = True
                info = Collision('Obstacle')
            # goal reached
            elif d_goal < 0:
                reward = self.config.success_reward
                done = True
                info = ReachGoal()
            # too close to people
            elif d_min_crowd < self.config.discomfort_dist:
                # adjust the reward based on FPS
                reward = (d_min_crowd - self.config.discomfort_dist) * self.config.discomfort_penalty_factor \
                         * self.config.time_step
                done = False
                info = Discomfort(d_min_crowd)
            # too close to obstacles
            elif d_min_obstacle < self.config.clearance_dist:
                # adjust the reward based on FPS
                reward = (d_min_obstacle - self.config.clearance_dist) * self.config.clearance_penalty_factor \
                         * self.config.time_step
                done = False
                info = Clearance(d_min_obstacle)
            else:
                reward = 0
                done = False
                info = Nothing()
            rp.logdebug('Current reward={} ({})'.format(reward, info))
            return reward, done, info

    def reset(self, phase='test', test_case=None):
        """
        reset the state of the environment to an initial state

        :return: initial observation (ob return from step)
        """
        assert phase in ['train', 'val', 'test']
        self.phase = phase

        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0.0
        self._prev_time = 0

        base_seed = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                     'val': 0, 'test': self.case_capacity['val']}

        if self.case_counter[phase] >= 0:
            new_seed = base_seed[phase] + self.case_counter[phase]
            np.random.seed(new_seed)
            self.seed = new_seed
            if phase == 'test':
                rp.logdebug('current test seed is:{}'.format(new_seed))
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            raise NotImplementedError("Case counter < 0 were for debug purposes in original environment")

        if self._sim_pid is not None:
            rp.loginfo("Env reset - Shutting down simulation process")
            self.roshandle.terminateOne(self._sim_pid)

        rp.loginfo("Env reset - Starting new simulation process")
        cli_args = {'p': self.config.scenario_xml,
                    'd': self.config.time_limit,
                    't': self.config.time_step}
        self._sim_pid = self.roshandle.start_rosnode('menge_sim', 'menge_sim', cli_args)
        rp.sleep(5)

        rp.logdebug("Set up subscribers")
        rp.Subscriber("crowd_expansion", MarkerArray, self._crowd_expansion_callback, queue_size=50, tcp_nodelay=True)
        rp.Subscriber("laser_static_end", PoseArray, self._static_obstacle_callback, queue_size=50, tcp_nodelay=True)
        rp.Subscriber("pose", PoseStamped, self._robot_pose_callback, queue_size=50, tcp_nodelay=True)
        rp.Subscriber("menge_sim_time", Float32, self._sim_time_callback, queue_size=50, tcp_nodelay=True)

        # Sample new goal
        self.sample_goal(exclude_initial=True)

        # perform idle action and return observation
        return self.step(np.array([0, 0], dtype=np.int32))[0]

    def render(self, mode='human', close=False):
        """
        render environment information to screen
        """
        if close:
            self.close()
        states = self.observation  # type: JointState
        rob_state = states.robot_state.observable_state
        ped_states = states.human_states.state
        ped_identifiers = states.human_states.get_identifiers()
        if len(rob_state) or len(ped_states):
            rp.loginfo('Tracked Objects')
            combined_state = np.concatenate((rob_state, ped_states), axis=0)
            row_labels = ['human {}'.format(int(i)) for i in ped_identifiers]
            if len(rob_state):
                row_labels.insert(0, 'robot')
            state_str = format_array(combined_state,
                                     row_labels=row_labels,
                                     col_labels=['x', 'y', 'phi', 'r', 'x_dot', 'y_dot', 'omega_dot'])
            rp.loginfo('\n' + state_str)
        else:
            rp.logwarn("No objects tracked")
        rp.loginfo('\nNumber of static obstacles: %d\n' % len(self._static_obstacles))

    def close(self):
        """
        close the environment
        """

        rp.loginfo("Env close - Shutting down simulation process and killing roscore")
        self.roshandle.terminate()
