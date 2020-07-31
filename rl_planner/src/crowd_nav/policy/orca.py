import numpy as np
import rvo2
from crowd_nav.policy.policy import Policy
from crowd_nav.utils.utils import mahalanobis_dist_nd
from menge_gym.envs.utils.motion_model import ModifiedAckermannModel
from typing import Union


def obstacles2segments(obstacles, d_min=0.1):
    """
    connects consecutive obstacle points close to each other (distance <= d_min) to obstacle segments.
    """
    obstacle_segments = []
    segments = []
    seg_count = 0

    if obstacles:
        dist = np.linalg.norm(obstacles.position[:-1] - obstacles.position[1:], axis=1)
        # new segment begins where distance between two consecutive obstacle positions becomes to large (> d_min)
        mask = dist > d_min
        obstacle_segments = []
        segments = []
        seg_count = 0
        # create array that tells which segment an obstacle belongs to
        for new_segment in mask:
            obstacle_segments.append(seg_count)
            seg_count += 1 if new_segment else 0
        obstacle_segments.append(seg_count)
        obstacle_segments = np.array(obstacle_segments)
        # merge all obstacle positions from same segment into single obstacle
        for segment_idx in np.unique(obstacle_segments):
            obstacle_segment = obstacles.position[np.where(obstacle_segments == segment_idx)[0]]
            segments.append(list(map(tuple, obstacle_segment)))

    return segments


class ORCA(Policy):
    def __init__(self):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        super().__init__()
        self.name = 'ORCA'
        self.trainable = False
        self.multiagent_training = True
        self.kinematics = None
        self.motion_model = None  # type: Union[ModifiedAckermannModel, None]
        self.safety_space = 0
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.radius = 0.3
        self.max_speed = 1
        self.action_space = None
        self.action_array = None
        self.action_indices = None
        self.sim = None

    def configure(self, config, env_config=None, action_space=(None, None)):
        self.action_space, self.action_array = action_space
        self.action_indices = np.array(np.meshgrid(np.arange(self.action_space.nvec[0]),
                                                   np.arange(self.action_space.nvec[1]))).T.reshape(-1, 2)

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
                raise ValueError("lf_ratio required in config for single_track motion_model")
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
            self.motion_model = ModifiedAckermannModel(robot_length, robot_lf_ratio)

    def set_phase(self, phase):
        return

    def predict(self, state):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        robot_state, human_states, obstacles = state

        params = {'neighborDist': self.neighbor_dist,
                  'maxNeighbors': self.max_neighbors,
                  'timeHorizon': self.time_horizon,
                  'timeHorizonObst': self.time_horizon_obst}
        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None
        # required to turn individual obstacle points into line segments to being able to process them
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(timeStep=self.time_step,
                                           radius=self.radius,
                                           maxSpeed=self.max_speed,
                                           **params)
            self.sim.addAgent(tuple(*robot_state.position),
                              radius=robot_state.radius[0] + self.safety_space,
                              maxSpeed=robot_state.v_pref[0],
                              velocity=tuple(*robot_state.velocity),
                              **params)
            for i, human_position in enumerate(human_states.position):
                self.sim.addAgent(tuple(human_position),
                                  radius=human_states.radius[i] + self.safety_space,
                                  maxSpeed=self.max_speed,
                                  velocity=tuple(human_states.velocity[i]),
                                  **params)

        else:
            self.sim.setAgentPosition(0, tuple(*robot_state.position))
            self.sim.setAgentVelocity(0, tuple(*robot_state.velocity))
            for i, human_position in enumerate(human_states.position):
                self.sim.setAgentPosition(i + 1, tuple(human_position))
                self.sim.setAgentVelocity(i + 1, tuple(human_states.velocity[i]))

        # Set the preferred velocity in the direction of the goal.
        velocity = robot_state.goal_position - robot_state.position
        speed = np.linalg.norm(velocity)
        v_pref = robot_state.v_pref[0]
        # cap preferred velocity at maximum v_pref
        pref_vel = velocity * v_pref / speed if speed > v_pref else velocity

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel.reshape(-1)))
        for i, human_position in enumerate(human_states.position):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()

        target_velocity = self.sim.getAgentVelocity(0)

        if self.kinematics == "holonomic":
            target_velocity_magnitude = np.linalg.norm(target_velocity)
            target_angle = np.arctan2(target_velocity[1], target_velocity[0])
            action = np.array([target_velocity_magnitude, target_angle])

        elif self.kinematics == "single_track":
            self.motion_model.setPose(robot_state.position, robot_state.orientation[0])
            action = self.motion_model.computeSteering(target_velocity)
        else:
            raise NotImplementedError("Only holonomic and single track model are implemented already")

        # find action in action space that is closest
        dist_action_space = mahalanobis_dist_nd(self.action_array, action)
        closest_action = np.unravel_index(np.argmin(dist_action_space), dist_action_space.shape)
        self.last_state = state

        return closest_action
