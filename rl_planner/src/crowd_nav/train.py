#!/usr/bin/env python3

import sys
import logging
import argparse
import os
import shutil
import importlib.util
import torch
import gym
import copy
import signal
import pickle
from tensorboardX import SummaryWriter
from crowd_nav.utils.robot import Robot
from crowd_nav.utils.trainer import VNRLTrainer, MPRLTrainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory


def iterable_to_device(iterable, device):
    new_iterable = []
    for i in iterable:
        res = None
        if isinstance(i, torch.Tensor):
            res = i.to(device)
        elif hasattr(i, "__iter__"):
            res = iterable_to_device(i, device)

        if res is not None:
            new_iterable.append(res)
    return tuple(new_iterable)


def set_random_seeds(seed):
    """
    Sets the random seeds for pytorch cpu and gpu
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def main(args):

    def signal_handler(signalNumber, frame):
        print("Received signal: {}".format(signal.Signals(signalNumber).name))
        try:
            if hasattr(explorer, "save_memory") and explorer.progress_file is not None:
                explorer.save_memory()
        except NameError:
            pass
        try:
            env.close()
        except NameError:
            pass
        sys.exit()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # set current working directory (cwd) to this script's location
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    set_random_seeds(args.randomseed)
    # configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        if args.restart:
            # do not overwrite existing progress on restart
            make_new_dir = False
        elif args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            key = input('Output directory already exists! Overwrite the folder? (y/n)')
            if key == 'y' and not args.resume:
                shutil.rmtree(args.output_dir)
            else:
                make_new_dir = False
    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.config, os.path.join(args.output_dir, 'config.py'))

        # # insert the arguments from command line to the config file
        # with open(os.path.join(args.output_dir, 'config.py'), 'r') as fo:
        #     config_text = fo.read()
        # search_pairs = {r"gcn.X_dim = \d*": "gcn.X_dim = {}".format(args.X_dim),
        #                 r"gcn.num_layer = \d": "gcn.num_layer = {}".format(args.layers),
        #                 r"gcn.similarity_function = '\w*'": "gcn.similarity_function = '{}'".format(args.sim_func),
        #                 r"gcn.layerwise_graph = \w*": "gcn.layerwise_graph = {}".format(args.layerwise_graph),
        #                 r"gcn.skip_connection = \w*": "gcn.skip_connection = {}".format(args.skip_connection)}
        #
        # for find, replace in search_pairs.items():
        #     config_text = re.sub(find, replace, config_text)
        #
        # with open(os.path.join(args.output_dir, 'config.py'), 'w') as fo:
        #     fo.write(config_text)

    args.config = os.path.join(args.output_dir, 'config.py')
    log_file = os.path.join(args.output_dir, 'output.log')
    il_weight_file = os.path.join(args.output_dir, 'il_model.pth')
    rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')
    progress_file = os.path.abspath(os.path.join(args.output_dir, 'progress.p'))
    spec = importlib.util.spec_from_file_location('config', args.config)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure logging
    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    logging.info('Current config content is :{}'.format(config))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)
    writer = SummaryWriter(log_dir=args.output_dir)

    # configure environment
    env_config = config.EnvConfig(args.debug)

    new_scenario = args.scenario  # type: str
    if new_scenario is not None:
        assert os.path.exists(new_scenario) and new_scenario.endswith(".xml"), "specified scenario is invalid"
        env_config.sim.scenario = new_scenario

    env = gym.make("menge_gym:MengeGym-v0")
    env.configure(env_config, args.randomseed)
    if hasattr(env, 'roshandle'):
        env.setup_ros_connection()

    # configure policy
    policy_config = config.PolicyConfig()
    policy = policy_factory[policy_config.name]()
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    policy_action_space = (env.action_space, env.action_array)
    policy.configure(policy_config, env.config, policy_action_space)
    policy.set_device(device)

    robot = Robot(env_config, 'robot')
    robot.time_step = env.config.time_step

    # read training parameters
    train_config = config.TrainConfig(args.debug)
    rl_learning_rate = train_config.train.rl_learning_rate
    train_batches = train_config.train.train_batches
    train_episodes = train_config.train.train_episodes
    sample_episodes = train_config.train.sample_episodes
    target_update_interval = train_config.train.target_update_interval
    evaluation_interval = train_config.train.evaluation_interval
    capacity = train_config.train.capacity
    epsilon_start = train_config.train.epsilon_start
    epsilon_end = train_config.train.epsilon_end
    epsilon_decay = train_config.train.epsilon_decay
    checkpoint_interval = train_config.train.checkpoint_interval

    # configure trainer and explorer
    memory = ReplayMemory(capacity)
    saved_episodes = 0

    if os.path.exists(progress_file):
        print("Loading progress file")
        progress = pickle.load(open(progress_file, "rb"))
        memory = progress["memory"]
        memory.memory = list(iterable_to_device(memory.memory, device))
        print("Memory length: {}".format(len(memory)))
        saved_episodes = progress["episode"]
        if saved_episodes is None:
            saved_episodes = 0
        print("Saved episodes: {}".format(saved_episodes))

    model = policy.get_model()
    batch_size = train_config.trainer.batch_size
    optimizer = train_config.trainer.optimizer
    if policy_config.name == 'model_predictive_rl':
        trainer = MPRLTrainer(model, policy.state_predictor, memory, device, policy, writer, batch_size, optimizer,
                              env.config.human_num,
                              reduce_sp_update_frequency=train_config.train.reduce_sp_update_frequency,
                              freeze_state_predictor=train_config.train.freeze_state_predictor,
                              detach_state_predictor=train_config.train.detach_state_predictor,
                              share_graph_model=policy_config.model_predictive_rl.share_graph_model)
    else:
        trainer = VNRLTrainer(model, memory, device, policy, batch_size, optimizer, writer)
    explorer = Explorer(env, robot, device, writer, progress_file, memory, policy.gamma, target_policy=policy)
    explorer.set_saved_episodes(saved_episodes)
    # imitation learning
    if args.resume:
        if not os.path.exists(rl_weight_file):
            logging.error('RL weights does not exist')
        policy.load_state_dict(torch.load(rl_weight_file))
        model = policy.get_model()
        rl_weight_file = os.path.join(args.output_dir, 'resumed_rl_model.pth')
        logging.info('Load reinforcement learning trained weights. Resume training')
    elif os.path.exists(il_weight_file):
        policy.load_state_dict(torch.load(il_weight_file))
        model = policy.get_model()
        logging.info('Load imitation learning trained weights.')
    else:
        il_episodes = train_config.imitation_learning.il_episodes - saved_episodes
        if il_episodes < 0:
            il_episodes = 0
        il_policy = train_config.imitation_learning.il_policy
        il_epochs = train_config.imitation_learning.il_epochs
        il_learning_rate = train_config.imitation_learning.il_learning_rate
        trainer.set_learning_rate(il_learning_rate)
        if env.config.robot_visibility:
            safety_space = 0
        else:
            raise NotImplementedError('Invisible Robot not implemented for Menge Sim')
            # safety_space = train_config.imitation_learning.safety_space
        il_policy = policy_factory[il_policy]()
        il_policy.configure(config, env.config, policy_action_space)
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = safety_space
        robot.set_policy(il_policy)
        explorer.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True)
        trainer.optimize_epoch(il_epochs)
        policy.save_model(il_weight_file)
        saved_episodes = 0
        logging.info('Finish imitation learning. Weights saved.')
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)

    trainer.update_target_model(model)

    # reinforcement learning
    policy.set_env(env)
    robot.set_policy(policy)
    trainer.set_learning_rate(rl_learning_rate)
    # fill the memory pool with some RL experience
    if args.resume:
        robot.policy.set_epsilon(epsilon_end)
        explorer.set_saved_episodes(saved_episodes)
        explorer.run_k_episodes(100, 'train', update_memory=True, episode=0)
    logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    best_val_reward = -1
    best_val_model = None

    # evaluate the model after imitation learning
    logging.info('Evaluate the model instantly after imitation learning on the validation cases')
    explorer.run_k_episodes(env.case_size['val'], 'val', episode=0)
    explorer.log('val', 0)

    if args.test_after_every_eval:
        explorer.run_k_episodes(env.case_size['test'], 'test', episode=0, print_failure=True)
        explorer.log('test', 0)

    explorer.set_saved_episodes(saved_episodes)
    episode = 0
    while (episode + saved_episodes) < train_episodes:
        if args.resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        robot.policy.set_epsilon(epsilon)

        # sample k episodes into memory and optimize over the generated memory
        explorer.run_k_episodes(sample_episodes, 'train', update_memory=True, episode=episode)
        explorer.log('train', episode)

        trainer.optimize_batch(train_batches, episode)
        episode += 1

        if episode % target_update_interval == 0:
            trainer.update_target_model(model)
        # evaluate the model
        if episode % evaluation_interval == 0:
            _, _, _, reward, _ = explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
            explorer.log('val', episode // evaluation_interval)

            if episode % checkpoint_interval == 0 and reward > best_val_reward:
                best_val_reward = reward
                best_val_model = copy.deepcopy(policy.get_state_dict())
        # test after every evaluation to check how the generalization performance evolves
            if args.test_after_every_eval:
                explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode, print_failure=True)
                explorer.log('test', episode // evaluation_interval)

        if episode != 0 and episode % checkpoint_interval == 0:
            current_checkpoint = episode // checkpoint_interval - 1
            save_every_checkpoint_rl_weight_file = rl_weight_file.split('.')[0] + '_' + str(current_checkpoint) + '.pth'
            policy.save_model(save_every_checkpoint_rl_weight_file)

    # # test with the best val model
    if best_val_model is not None:
        policy.load_state_dict(best_val_model)
        torch.save(best_val_model, os.path.join(args.output_dir, 'best_val.pth'))
        logging.info('Save the best val model with the reward: {}'.format(best_val_reward))
    explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode, print_failure=True)
    env.close()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='model_predictive_rl')
    parser.add_argument('--config', type=str, default='configs/modified_benchmark_config/mp_separate.py')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--restart', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--test_after_every_eval', default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=17)
    parser.add_argument('--scenario', type=str, default=None)

    # arguments for GCN
    # parser.add_argument('--X_dim', type=int, default=32)
    # parser.add_argument('--layers', type=int, default=2)
    # parser.add_argument('--sim_func', type=str, default='embedded_gaussian')
    # parser.add_argument('--layerwise_graph', default=False, action='store_true')
    # parser.add_argument('--skip_connection', default=True, action='store_true')

    sys_args = parser.parse_args()

    main(sys_args)
