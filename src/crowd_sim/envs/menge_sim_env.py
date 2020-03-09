import gym
from gym import spaces
import numpy as np
import rospy as rp
import rosnode
from geometry_msgs.msg import PoseArray, PoseStamped, Twist
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Bool
from .utils.ros import pose2array, obstacle2array, marker2array, start_roslaunch_file
from .utils.info import *


class MengeGym(gym.Env):
    """
    Custom gym environment for the Menge_ROS simulation
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, scenario_xml):
        """

        :param scenario_xml: menge simulator scenario project file
        """

        super(MengeGym, self).__init__()
        rp.logdebug("Initializing environment")

        self.scenario_xml = scenario_xml
        self.timeout = 120
        self.time_step = 0.01  # do 10 ms step
        self.n_observations = 10
        v_max = 2
        num_speeds = 5
        num_angles = 16

        # from config
        self.success_reward = 1
        self.collision_penalty = -0.25
        self.discomfort_dist = 0.2
        self.discomfort_penalty_factor = 0.5

        rp.logdebug("Start Menge simulator launch file")
        launch_cli_args = {'project': self.scenario_xml,
                           'timeout': self.timeout,
                           'timestep': self.time_step}
        self._sim_process = start_roslaunch_file('menge_vis', 'menge.launch', launch_cli_args)
        self._sim_process.start()

        # simulation controls
        rp.logdebug("Set up publishers and subscribers")
        rp.init_node('MengeSimEnv')
        self._pub_run = rp.Publisher('run', Bool, queue_size=1)

        # observation space
        # always keep last n_observations
        self._crowd_poses = [None] * self.n_observations
        self._crowd_pose_i = 0
        self._static_obstacles = [None] * self.n_observations
        self._stat_obs_i = 0
        self._robot_poses = [None] * self.n_observations
        self._rob_pose_i = 0
        # rp.Subscriber("crowd_pose", PoseArray, self._crowd_pose_callback)
        rp.Subscriber("crowd_expansion", MarkerArray, self._crowd_expansion_callback)
        rp.Subscriber("laser_static_end", PoseArray, self._static_obstacle_callback)
        rp.Subscriber("pose", PoseStamped, self._robot_pose_callback)

        # action space
        # from paper RGL for CrowdNav --> 5 speeds (0, v_pref] and 16 headings [0, 2*pi)
        # exponentially distributed speeds
        self._velocities = np.logspace(0, np.log(v_max+1), num_speeds, endpoint=True, base=np.e) - 1
        # linearly distributed angles
        # make num_angles odd to ensure null action (0 --> not changing steering)
        num_angles = num_angles // 2 * 2 + 1
        self._angles = np.linspace(-np.pi, np.pi, num_angles, endpoint=True)
        self._action_space = spaces.MultiDiscrete([num_speeds, num_angles])

        self._cmd_vel_pub = rp.Publisher('/cmd_vel', Twist, queue_size=1)

        # initialize time
        self._run_duration = rp.Duration(self.time_step)

    # def _crowd_pose_callback(self, msg):
    #     # transform PoseArray message to numpy array
    #     rp.logdebug('Crowd Pose subscriber callback called')
    #     pose_array = np.array(list(map(pose2array, msg.poses)))
    #     # update list of crowd poses + pointer to current position
    #     crowd_pose_i = self._crowd_pose_i
    #     self._crowd_poses[crowd_pose_i] = pose_array
    #     self._crowd_pose_i = (crowd_pose_i + 1) % self.n_observations

    def _crowd_expansion_callback(self, msg):
        # transfor MarkerArray message to numpy array
        rp.logdebug('Crowd Expansion subscriber callback called')
        pose_array = np.array(list(map(marker2array, msg.markers)))
        # update list of crowd poses + pointer to current position
        crowd_pose_i = self._crowd_pose_i
        self._crowd_poses[crowd_pose_i] = pose_array
        self._crowd_pose_i = (crowd_pose_i + 1) % self.n_observations

    def _static_obstacle_callback(self, msg):
        rp.logdebug('Static Obstacle subscriber callback called')
        # transform PoseArray message to numpy array
        pose_array = np.array(list(map(obstacle2array, msg.poses)))
        # update list of static obstacle sets + pointer to current position
        obs_i = self._stat_obs_i
        self._crowd_poses[obs_i] = pose_array
        self._crowd_pose_i = (obs_i + 1) % self.n_observations

    def _robot_pose_callback(self, msg):
        rp.logdebug('Robot Pose subscriber callback called')
        # extract 2D pose and orientation from message
        robot_pose = msg.pose
        robot_x = robot_pose.position.x
        robot_y = robot_pose.position.y
        robot_q_w = robot_pose.orientation.w
        robot_q_z = robot_pose.orientation.z
        # update list of robot poses + pointer to current position
        rob_pose_i = self._rob_pose_i
        self._robot_poses[rob_pose_i] = np.array([robot_x, robot_y, robot_q_w, robot_q_z])
        self._rob_pose_i = (rob_pose_i + 1) % self.n_observations

    def step(self, action):
        rp.logdebug("performing step in the environment")

        # execute one time step within the environment
        # in menge_ros the published angle defines an angle increment

        vel_msg = Twist()
        vel_msg.linear.x = self._velocities[action[0]]   # vel_action
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = self._angles[action[1]]      # angle_action

        self._pub_run.publish(Bool(data=True))

        current_time = start_time = rp.Time.now()
        while current_time <= start_time + self._run_duration:
            self._cmd_vel_pub.publish(vel_msg)
            current_time = rp.Time.now()
        self._pub_run.publish(Bool(data=False))

        # TODO: implement distance to goal, collision, etc.

        if '/menge_sim' not in rosnode.get_node_names():
            reward = 0
            done = True
            info = Timeout()
        # elif collision:
        #     reward = self.collision_penalty
        #     done = True
        #     info = Collision()
        # elif reaching_goal:
        #     reward = self.success_reward
        #     done = True
        #     info = ReachGoal()
        # elif dmin < self.discomfort_dist:
        #     # adjust the reward based on FPS
        #     reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
        #     done = False
        #     info = Discomfort(dmin)
        # else:
        #     reward = 0
        #     done = False
        #     info = Nothing()

    def reset(self):
        # reset the state of the environment to an initial state
        rp.logdebug("Env reset - Shutting down simulation process")
        self._sim_process.shutdown()
        rp.logdebug("Env reset - Starting new simulation process")
        launch_cli_args = {'project': self.scenario_xml,
                           'timeout': self.timeout,
                           'timestep': self.time_step}
        self._sim_process = start_roslaunch_file('menge_vis', 'menge.launch', launch_cli_args)
        self._sim_process.start()

    def render(self, mode='human', close=False):
        # render environment to screen
        ...
