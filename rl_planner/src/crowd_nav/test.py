#!/usr/bin/env python3

import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
import subprocess
from signal import SIGINT
from psutil import Process as psutilProcess
from datetime import datetime
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.robot import Robot
from crowd_nav.policy.orca import ORCA


def main(args):
    # set current working directory (cwd) to this script's location
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # configure logging and device
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    if args.model_dir is not None:
        if args.config is not None:
            config_file = args.config
        else:
            config_file = os.path.join(args.model_dir, 'config.py')
        if args.model_file is not None:
            model_weights = os.path.join(args.model_dir, args.model_file)
        elif args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
            logging.info('Loaded IL weights')
        elif args.rl:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_file = 'resumed_rl_model.pth'
            else:
                model_file = sorted(filter(lambda x: x.startswith("rl_model"), os.listdir(args.model_dir)))[-1]
            model_weights = os.path.join(args.model_dir, model_file)
            logging.info('Loaded RL weights from {}'.format(model_file))
        else:
            model_weights = os.path.join(args.model_dir, 'best_val.pth')
            logging.info('Loaded RL weights with best VAL')

    else:
        config_file = args.config

    spec = importlib.util.spec_from_file_location('config', config_file)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure environment
    env_config = config.EnvConfig(args.debug)

    new_scenario = args.test_scenario  # type: str
    if new_scenario is not None:
        assert os.path.exists(new_scenario) and new_scenario.endswith(".xml"), "specified scenario is invalid"
        env_config.sim.scenario = new_scenario
    else:
        new_scenario = env_config.sim.scenario

    if args.human_num is not None:
        env_config.sim.human_num = args.human_num

    env = gym.make("menge_gym:MengeGym-v0")
    env.configure(env_config)
    if hasattr(env, 'roshandle'):
        env.setup_ros_connection()

    # configure policy
    policy_config = config.PolicyConfig(args.debug)
    policy_name = policy_config.name
    policy = policy_factory[policy_name]()
    if args.planning_depth is not None:
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_depth = args.planning_depth
    if args.planning_width is not None:
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_width = args.planning_width
    if args.sparse_search:
        policy_config.model_predictive_rl.sparse_search = True

    policy_action_space = (env.action_space, env.action_array)
    policy.configure(policy_config, env.config, policy_action_space)

    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.load_model(model_weights)

    policy.set_phase(args.phase)
    policy.set_device(device)

    robot = Robot(env_config, 'robot')
    robot.time_step = env.config.time_step
    robot.set_policy(policy)

    train_config = config.TrainConfig(args.debug)
    epsilon_end = train_config.train.epsilon_end
    if not isinstance(robot.policy, ORCA):
        robot.policy.set_epsilon(epsilon_end)

    explorer = Explorer(env, robot, device, None, gamma=0.9)

    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if env.config.robot_visibility:
            robot.policy.safety_space = 0
        else:
            robot.policy.safety_space = args.safety_space
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)

    if args.visualize:
        # -a makes record all topics, -q suppresses console output
        record_expr = ["rosbag", "record", "-aq"]
        if args.bag_file is not None:
            rec_file = args.bag_file
            if not rec_file.endswith(".bag"):
                rec_file += ".bag"
        else:
            date = datetime.now().strftime("%d_%m_%Y")
            scenario_name = os.path.splitext(os.path.split(new_scenario)[1])[0]
            rec_file = "_".join((scenario_name, args.phase, policy_name, date)) + ".bag"
        rec_dir = args.bag_dir
        if not os.path.isdir(rec_dir):
            os.makedirs(rec_dir)
        rec_full_path = os.path.join(rec_dir, rec_file)
        record_expr.extend(["-O", rec_full_path])

        # # it would be possible to add additional args for rosbag record (e.g. max filesize, duration,
        # #                                                                including/excluding topics)
        # # see http://wiki.ros.org/rosbag/Commandline#record
        # add_args = []
        # for arg in add_args:
        #     record_expr.append(arg)

        # start rosbag record
        print("Start recording bag file")
        rosbag_proc = subprocess.Popen(record_expr)
        # TODO: CONTINUE HERE (add visualization flag to explorer and down + corresponding messages)
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True, visualize=True)
        # finish rosbag record
        print("Stop recording bag file")
        p = psutilProcess(rosbag_proc.pid)
        for child in p.children(recursive=True):
            # make sure all child processes are ended
            child.send_signal(SIGINT)
        rosbag_proc.wait()
        rosbag_proc.send_signal(SIGINT)
    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)

    env.close()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('-m', '--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-b', '--bag_file', type=str, default=None)
    parser.add_argument('--bag_dir', type=str, default='data/output')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--human_num', type=int, default=None)
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('--test_scenario', type=str, default=None)
    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--sparse_search', default=False, action='store_true')

    sys_args = parser.parse_args()

    main(sys_args)
