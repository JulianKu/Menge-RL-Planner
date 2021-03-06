import os
import logging
import copy
import pickle
import torch
from tqdm import tqdm
from collections import Counter
from menge_gym.envs.utils.info import *
from crowd_nav.utils.utils import human_poses_from_traj, human_traj_ros_msg


class Explorer(object):
    def __init__(self, env, robot, device, writer, progress_file=None, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.writer = writer
        self.progress_file = progress_file
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.statistics = None
        self.current_episode = None
        self.saved_episodes = None

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None, epoch=None,
                       print_failure=False, visualize=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        discomfort = 0
        clearance = 0
        comfort_min_dist = []
        clearance_min_dist = []
        cumulative_rewards = []
        average_returns = []
        collision_cases = []
        collision_types = []
        timeout_cases = []

        if k != 1:
            pbar = tqdm(total=k)
        else:
            pbar = None

        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            step_no = 0
            self.current_episode = episode if episode is not None else i
            while not done:
                step_no += 1
                print("######################################")
                print("RUNNING EPISODE {}, STEP NUMBER {}".format(self.current_episode, step_no))
                print("######################################")
                action = self.robot.policy.predict(ob)
                ob, reward, done, info = self.env.step(action)
                if visualize:
                    traj_msg = None
                    if hasattr(self.robot.policy, "get_traj"):
                        traj = self.robot.policy.get_traj()
                        human_traj = map(human_poses_from_traj, traj)
                        traj_msg = human_traj_ros_msg(human_traj)
                    self.env.render(mode="ros", human_traj_prediction_msg=traj_msg)
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)

                if isinstance(info, InterfaceTimeout):
                    # env requires reset, because it went into an interface error
                    print("{} - Simulator Interface timed out, resetting env".format(info))
                    ob = self.env.reset(phase)
                    done = False
                    states = []
                    actions = []
                    rewards = []
                    step_no = 0
                if isinstance(info, Discomfort):
                    discomfort += 1
                    comfort_min_dist.append(info.min_dist)

                if isinstance(info, Clearance):
                    clearance += 1
                    clearance_min_dist.append(info.min_dist)

            # save replay buffer every 50th episode
            if (self.progress_file is not None) and (self.current_episode % 50 == 0) and self.current_episode != 0:
                self.save_memory()

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_types.append(info.partner)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.config.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.env.config.robot_v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            returns = []
            for step in range(len(rewards)):
                step_return = sum([pow(self.gamma, t * self.robot.time_step * self.env.config.robot_v_pref)
                                   * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)
            average_returns.append(average(returns))

            if pbar:
                pbar.update(1)
        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k

        collision_types = Counter(collision_types)
        collision_type_rates = {col_type: collision_types[col_type] * collision_rate / sum(collision_types.values())
                                for col_type in collision_types}
        for col_type in ["Crowd", "Obstacle"]:
            if col_type not in collision_types:
                collision_type_rates[col_type] = 0

        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.config.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        extra_info = extra_info + '' if epoch is None else extra_info + ' in epoch {} '.format(epoch)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f} (crowd: {:.2f}, obstacles: {:.2f}), '
                     'nav time: {:.2f}, total reward: {:.4f}, average return: {:.4f}'
                     .format(phase.upper(), extra_info, success_rate, collision_rate,
                             collision_type_rates['Crowd'], collision_type_rates['Obstacle'],
                             avg_nav_time, average(cumulative_rewards), average(average_returns)))
        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times)
            discomfort_rate = discomfort / total_time
            clearance_rate = clearance / total_time
            logging.info('Frequency of being too close to other bodies: {:.2f} (crowd: {:.2f}, obstacle: {:.2f}) and '
                         'average min separation distance (when too close): {:.2f} (crowd: {:.2f}, obstacle: {:.2f})'
                         .format(discomfort_rate + clearance_rate, discomfort_rate, clearance_rate,
                                 average(comfort_min_dist + clearance_min_dist), average(comfort_min_dist),
                                 average(clearance_min_dist)))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
        collision_type_rates.update({'Total': collision_rate})
        self.statistics = success_rate, collision_type_rates, avg_nav_time, average(cumulative_rewards), \
                          average(average_returns)

        return self.statistics

    def set_saved_episodes(self, episode: int):
        self.saved_episodes = episode

    def save_memory(self):
        assert self.progress_file is not None, "progress file needs to be set to save memory and current episode"
        print("Dump memory to file")
        episode = self.current_episode
        if self.saved_episodes is not None:
            episode += self.saved_episodes
        progress = {"memory": self.memory, "episode": episode}
        pickle.dump(progress, open(self.progress_file, "wb"))

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states[:-1]):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                next_state = self.target_policy.transform(states[i + 1])
                value = sum([pow(self.gamma, (t - i) * self.robot.time_step * self.env.config.robot_v_pref) * reward *
                             (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                next_state = states[i + 1]
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    value = 0
            value = torch.Tensor([value]).to(self.device)
            reward = torch.Tensor([rewards[i]]).to(self.device)

            if self.target_policy.name == 'ModelPredictiveRL':
                self.memory.push((state[0], state[1], state[2], value, reward,
                                  next_state[0], next_state[1], next_state[2]))
            else:
                self.memory.push((state, value, reward, next_state))

    def log(self, tag_prefix, global_step):
        sr, crs, time, reward, avg_return = self.statistics
        self.writer.add_scalar(tag_prefix + '/success_rate', sr, global_step)
        self.writer.add_scalars(tag_prefix + '/collision_rate', crs, global_step)
        self.writer.add_scalar(tag_prefix + '/time', time, global_step)
        self.writer.add_scalar(tag_prefix + '/reward', reward, global_step)
        self.writer.add_scalar(tag_prefix + '/avg_return', avg_return, global_step)


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
