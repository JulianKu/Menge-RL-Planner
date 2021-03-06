import logging
import abc
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial


class MPRLTrainer(object):
    def __init__(self, value_estimator, state_predictor, memory, device, policy, writer, batch_size, optimizer_str,
                 human_num,
                 reduce_sp_update_frequency, freeze_state_predictor, detach_state_predictor, share_graph_model):
        """
        Train the trainable model of a policy
        """
        self.value_estimator = value_estimator
        self.state_predictor = state_predictor
        self.device = device
        self.writer = writer
        self.target_policy = policy
        self.target_model = None
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer_str = optimizer_str
        self.reduce_sp_update_frequency = reduce_sp_update_frequency
        self.state_predictor_update_interval = human_num
        self.freeze_state_predictor = freeze_state_predictor
        self.detach_state_predictor = detach_state_predictor
        self.share_graph_model = share_graph_model
        self.v_optimizer = None
        self.s_optimizer = None

        # for value update
        self.gamma = 0.9
        self.time_step = 0.25
        self.v_pref = 1

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def set_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.v_optimizer = optim.Adam(self.value_estimator.parameters(), lr=learning_rate)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.Adam(self.state_predictor.parameters(), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.v_optimizer = optim.SGD(self.value_estimator.parameters(), lr=learning_rate, momentum=0.9)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.SGD(self.state_predictor.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError

        if self.state_predictor.trainable:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters()) +
                 list(self.state_predictor.named_parameters())]), self.optimizer_str))
        else:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters())]), self.optimizer_str))

    def optimize_epoch(self, num_epochs):
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True,
                                          collate_fn=partial(custom_collate, device=self.device))

        for epoch in range(num_epochs):
            epoch_v_loss = 0
            epoch_s_loss = 0
            logging.debug('{}-th epoch starts'.format(epoch))

            update_counter = 0
            for data in self.data_loader:
                robot_states, human_states, obstacles, values, _, _, next_human_states, _ \
                    = data
                joint_state = (robot_states, human_states, obstacles)

                # optimize value estimator
                self.v_optimizer.zero_grad()

                outputs = self.value_estimator(joint_state)
                values = values.to(self.device)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.v_optimizer.step()
                epoch_v_loss += loss.data.item()

                # optimize state predictor
                if self.state_predictor.trainable:
                    update_state_predictor = True
                    if update_counter % self.state_predictor_update_interval != 0:
                        update_state_predictor = False

                    if update_state_predictor:
                        self.s_optimizer.zero_grad()
                        _, (next_human_states_est, next_human_identifiers_est), _ = self.state_predictor(joint_state,
                                                                                                         None)
                        next_human_mask, (next_human_states, next_human_identifiers) = next_human_states

                        # if there is a mask, mask out padded values from predicted human state
                        human_mask = human_states[0] if isinstance(human_states, tuple) \
                                                        and isinstance(human_states[1], tuple) else 1
                        next_human_states_est = human_mask * next_human_states_est

                        if next_human_identifiers is not None and next_human_identifiers_est is not None:
                            # find matching indices between next_human_states and estimated next_human_states
                            compareview_id = next_human_identifiers.expand(
                                *next_human_identifiers.shape[:-1],
                                next_human_identifiers_est.shape[1]).permute(0, 2, 1)
                            identifier_mask = compareview_id == next_human_identifiers_est

                            # mask out all identifiers that were padded in one of the two frames
                            compareview_pad = next_human_mask.expand(*next_human_mask.shape[:-1],
                                                                     human_mask.shape[1]).permute(0, 2, 1)
                            pad_mask = (compareview_pad == 1) & (human_mask == 1)
                            batch_idx, human_est_idx, human_idx = torch.where(identifier_mask & pad_mask)

                            # compute loss only for humans that appear in two consecutive frames and are not masked out
                            loss = self.criterion(next_human_states_est[batch_idx, human_est_idx],
                                                  next_human_states[batch_idx, human_idx])
                        else:
                            loss = self.criterion(next_human_states_est,
                                                  next_human_states)
                        loss.backward()
                        self.s_optimizer.step()
                        epoch_s_loss += loss.data.item()
                    update_counter += 1

            logging.debug('{}-th epoch ends'.format(epoch))
            self.writer.add_scalar('IL/epoch_v_loss', epoch_v_loss / len(self.memory), epoch)
            self.writer.add_scalar('IL/epoch_s_loss', epoch_s_loss / len(self.memory), epoch)
            logging.info('Average loss in epoch %d: %.2E, %.2E', epoch, epoch_v_loss / len(self.memory),
                         epoch_s_loss / len(self.memory))

        return

    def optimize_batch(self, num_batches, episode):
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True,
                                          collate_fn=partial(custom_collate, device=self.device))
        v_losses = 0
        s_losses = 0
        batch_count = 0
        for data in self.data_loader:
            robot_states, human_states, obstacles, _, rewards, next_robot_states, next_human_states, next_obstacles \
                = data
            joint_state = (robot_states, human_states, obstacles)
            joint_next_state = (next_robot_states, next_human_states, next_obstacles)
            # optimize value estimator
            self.v_optimizer.zero_grad()
            outputs = self.value_estimator(joint_state)

            gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
            target_values = rewards + gamma_bar * self.target_model(joint_next_state)

            # values = values.to(self.device)
            loss = self.criterion(outputs, target_values)
            loss.backward()
            self.v_optimizer.step()
            v_losses += loss.data.item()

            # optimize state predictor
            if self.state_predictor.trainable:
                update_state_predictor = True
                if self.freeze_state_predictor:
                    update_state_predictor = False
                elif self.reduce_sp_update_frequency and batch_count % self.state_predictor_update_interval == 0:
                    update_state_predictor = False

                if update_state_predictor:
                    self.s_optimizer.zero_grad()
                    _, (next_human_states_est, next_human_identifiers_est), _ = self.state_predictor(joint_state, None,
                                                                                                     detach=self.detach_state_predictor)
                    next_human_mask, (next_human_states, next_human_identifiers) = next_human_states

                    # if there is a mask, mask out padded values from predicted human state
                    human_mask = human_states[0] if isinstance(human_states, tuple) \
                                                    and isinstance(human_states[1], tuple) else 1
                    next_human_states_est = human_mask * next_human_states_est

                    if next_human_identifiers is not None and next_human_identifiers_est is not None:
                        # find matching indices between next_human_states and estimated next_human_states
                        compareview_id = next_human_identifiers.expand(
                            *next_human_identifiers.shape[:-1],
                            next_human_identifiers_est.shape[1]).permute(0, 2, 1)
                        identifier_mask = compareview_id == next_human_identifiers_est

                        # mask out all identifiers that were padded in one of the two frames
                        compareview_pad = next_human_mask.expand(*next_human_mask.shape[:-1],
                                                                 human_mask.shape[1]).permute(0, 2, 1)
                        pad_mask = (compareview_pad == 1) & (human_mask == 1)
                        batch_idx, human_est_idx, human_idx = torch.where(identifier_mask & pad_mask)

                        # compute loss only for humans that appear in two consecutive frames and are not masked out
                        loss = self.criterion(next_human_states_est[batch_idx, human_est_idx],
                                              next_human_states[batch_idx, human_idx])
                    else:
                        loss = self.criterion(next_human_states_est,
                                              next_human_states)
                    loss.backward()
                    self.s_optimizer.step()
                    s_losses += loss.data.item()

            batch_count += 1
            if batch_count > num_batches:
                break

        average_v_loss = v_losses / num_batches
        average_s_loss = s_losses / num_batches
        logging.info('Average loss : %.2E, %.2E', average_v_loss, average_s_loss)
        self.writer.add_scalar('RL/average_v_loss', average_v_loss, episode)
        self.writer.add_scalar('RL/average_s_loss', average_s_loss, episode)

        return average_v_loss, average_s_loss


class VNRLTrainer(object):
    def __init__(self, model, memory, device, policy, batch_size, optimizer_str, writer):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.policy = policy
        self.target_model = None
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer_str = optimizer_str
        self.optimizer = None
        self.writer = writer

        # for value update
        self.gamma = 0.9
        self.time_step = 0.25
        self.v_pref = 1

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def set_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise NotImplementedError
        logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
            [name for name, param in self.model.named_parameters()]), self.optimizer_str))

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True, collate_fn=pad_batch)
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            logging.debug('{}-th epoch starts'.format(epoch))
            for data in self.data_loader:
                inputs, values, _, _ = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                values = values.to(self.device)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()
            logging.debug('{}-th epoch ends'.format(epoch))
            average_epoch_loss = epoch_loss / len(self.memory)
            self.writer.add_scalar('IL/average_epoch_loss', average_epoch_loss, epoch)
            logging.info('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches, episode=None):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True, collate_fn=pad_batch)
        losses = 0
        batch_count = 0
        for data in self.data_loader:
            inputs, _, rewards, next_states = data
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
            target_values = rewards + gamma_bar * self.target_model(next_states)

            loss = self.criterion(outputs, target_values)
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()
            batch_count += 1
            if batch_count > num_batches:
                break

        average_loss = losses / num_batches
        logging.info('Average loss : %.2E', average_loss)

        return average_loss


def custom_collate(batch, device):
    def pad(elems):
        # pad with zeros to unify length
        padded = torch.nn.utils.rnn.pad_sequence(elems, batch_first=True)
        # mask is 0 for padded elements, else 1
        mask = [torch.Tensor(t.shape[0] * [[1.]]).to(device) if t.size(0)
                else torch.Tensor([[0.]]).to(device) for t in elems]
        mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)

        return mask, padded

    robot, humans, obstacles, values, rewards, next_robot, next_humans, next_obstacles = zip(*batch)
    humans, h_id = zip(*humans)
    next_humans, next_h_id = zip(*next_humans)

    robot = torch.stack(robot, 0)
    # number of humans can change, thus needs padding
    humans = pad(humans)
    h_id = pad(h_id)
    # masks are the same for states and identifiers
    humans = (humans[0], (humans[1], h_id[1]))
    # number of obstacles can change, thus needs padding
    obstacles = pad(obstacles)
    values = torch.cat(values).unsqueeze(1)
    rewards = torch.cat(rewards).unsqueeze(1)
    next_robot = torch.stack(next_robot, 0)
    next_humans = pad(next_humans)
    next_h_id = pad(next_h_id)
    next_humans = (next_humans[0], (next_humans[1], next_h_id[1]))
    next_obstacles = pad(next_obstacles)

    return robot, humans, obstacles, values, rewards, next_robot, next_humans, next_obstacles


def pad_batch(batch):
    """
    args:
        batch - list of (tensor, label)
    return:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
    """
    def sort_states(position):
        # sort the sequences in the decreasing order of length
        sequences = sorted([x[position] for x in batch], reverse=True, key=lambda t: t.size()[0])
        packed_sequences = torch.nn.utils.rnn.pack_sequence(sequences)
        return torch.nn.utils.rnn.pad_packed_sequence(packed_sequences, batch_first=True)

    states = sort_states(0)
    values = torch.cat([x[1] for x in batch]).unsqueeze(1)
    rewards = torch.cat([x[2] for x in batch]).unsqueeze(1)
    next_states = sort_states(3)

    return states, values, rewards, next_states
