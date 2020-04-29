import logging
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu
from torch.nn import Parameter
from crowd_nav.policy.helpers import mlp


class RGL(nn.Module):
    def __init__(self, config, robot_state_dim, human_state_dim, static_obs_dim):
        """ The current code might not be compatible with models trained with previous version
        """
        super().__init__()
        self.multiagent_training = config.gcn.multiagent_training
        num_layer = config.gcn.num_layer
        X_dim = config.gcn.X_dim
        O_dim = config.gcn.O_dim
        wr_dims = config.gcn.wr_dims
        wh_dims = config.gcn.wh_dims
        ws_dims = config.gcn.ws_dims
        final_state_dim = config.gcn.final_state_dim
        similarity_function = config.gcn.similarity_function
        layerwise_graph = config.gcn.layerwise_graph
        skip_connection = config.gcn.skip_connection

        # design choice

        # 'gaussian', 'embedded_gaussian', 'cosine', 'cosine_softmax', 'concatenation'
        self.similarity_function = similarity_function
        self.robot_state_dim = robot_state_dim
        self.human_state_dim = human_state_dim
        self.static_obs_dim = static_obs_dim
        self.num_layer = num_layer
        self.X_dim = X_dim
        self.O_dim = O_dim
        self.layerwise_graph = layerwise_graph
        self.skip_connection = skip_connection

        logging.info('Similarity_func: {}'.format(self.similarity_function))
        logging.info('Layerwise_graph: {}'.format(self.layerwise_graph))
        logging.info('Skip_connection: {}'.format(self.skip_connection))
        logging.info('Number of layers: {}'.format(self.num_layer))

        self.w_r = mlp(robot_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(human_state_dim, wh_dims, last_relu=True)
        self.w_s = mlp(static_obs_dim, ws_dims, last_relu=True)

        if self.similarity_function == 'embedded_gaussian':
            self.w_a_dyn = Parameter(torch.randn(X_dim, X_dim))
            self.w_a_stat = Parameter(torch.randn(X_dim, O_dim))
        elif self.similarity_function == 'concatenation':
            self.w_a_dyn = mlp(2 * X_dim, [2 * X_dim, 1], last_relu=True)
            self.w_a_stat = mlp(X_dim + O_dim, [X_dim + O_dim, 1], last_relu=True)
        else:
            self.w_a_dyn = None
            self.w_a_stat = None

        # TODO: try other dim size
        embedding_dim_x = self.X_dim
        embedding_dim_o = self.O_dim
        self.Ws_dyn = torch.nn.ParameterList()
        self.Ws_stat = torch.nn.ParameterList()
        for i in range(self.num_layer):
            if i == 0:
                self.Ws_dyn.append(Parameter(torch.randn(self.X_dim, embedding_dim_x)))
                self.Ws_stat.append(Parameter(torch.randn(self.O_dim, embedding_dim_o)))
            elif i == self.num_layer - 1:
                self.Ws_dyn.append(Parameter(torch.randn(embedding_dim_x, final_state_dim)))
                self.Ws_stat.append(Parameter(torch.randn(embedding_dim_o, final_state_dim)))
            else:
                self.Ws_dyn.append(Parameter(torch.randn(embedding_dim_x, embedding_dim_x)))
                self.Ws_stat.append(Parameter(torch.randn(embedding_dim_o, embedding_dim_o)))

        # for visualization
        self.A = None

    def compute_similarity_matrix(self, X1, X2=None, w=None):

        if X2 is None:
            X2 = X1
        elif self.similarity_function == 'diagonal' and X1.size(1) != X2.size(1):
            raise ValueError("diagonal matrix must be square")

        if self.similarity_function == 'embedded_gaussian':
            if w is None:
                raise ValueError("for 'embedded_gaussian' a weight tensor is required")
            A = torch.matmul(torch.matmul(X1, w), X2.permute(0, 2, 1))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'gaussian':
            A = torch.matmul(X1, X2.permute(0, 2, 1))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'cosine':
            A = torch.matmul(X1, X2.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = torch.div(A, norm_matrix)
        elif self.similarity_function == 'cosine_softmax':
            A = torch.matmul(X1, X2.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = softmax(torch.div(A, norm_matrix), dim=2)
        elif self.similarity_function == 'concatenation':
            if w is None:
                raise ValueError("for 'concatenation' an mlp network is required")
            # all combinations of X1 indices and X2 indices
            indices = torch.stack([torch.arange(X1.size(1)).view(-1, 1).expand(X2.size(1), X1.size(1)).reshape(-1),
                                   torch.arange(X2.size(1)).repeat(X1.size(1))])
            selected_features_1 = torch.index_select(X1, dim=1, index=indices[0])
            selected_features_2 = torch.index_select(X2, dim=1, index=indices[2])
            selected_features = torch.stack([selected_features_1, selected_features_2], dim=2)
            pairwise_features = selected_features.reshape(-1, X1.size(1) * X2.size(1), X1.size(2) + X2.size(2))
            A = w(pairwise_features).reshape(-1, X1.size(1), X2.size(1))
            normalized_A = A
        elif self.similarity_function == 'squared':
            A = torch.matmul(X1, X2.permute(0, 2, 1))
            squared_A = A * A
            normalized_A = squared_A / torch.sum(squared_A, dim=2, keepdim=True)
        elif self.similarity_function == 'equal_attention':
            normalized_A = (torch.ones(X1.size(1), X2.size(1)) / X2.size(1)).expand(X1.size(0), X1.size(1), X2.size(1))
        elif self.similarity_function == 'diagonal':
            normalized_A = (torch.eye(X1.size(1), X2.size(1))).expand(X1.size(0), X1.size(1), X2.size(1))
        else:
            raise NotImplementedError

        return normalized_A

    def forward(self, state):
        """
        Embed current state tensor pair (robot_state, human_states) into a latent space
        Each tensor is of shape (batch_size, # of agent, features)
        :param state:
        :return:
        """

        robot_state, human_states, static_obs = state

        if isinstance(human_states, torch.Tensor):
            human_mask = 1
        else:
            raise NotImplementedError("human_states must be Tensor, tuple of Tensors (state, identifiers) "
                                      "or tuple of Tensor and tuple (mask, (state, identifiers))")

        if isinstance(static_obs, torch.Tensor):
            obs_mask = 1
        elif isinstance(static_obs, tuple):
            obs_mask, static_obs = static_obs
        else:
            raise NotImplementedError("static_obs must be Tensor or tuple of Tensors (mask, state)")

        # compute feature matrix X
        robot_state_embeddings = self.w_r(robot_state)
        human_state_embeddings = human_mask * self.w_h(human_states)  # mask out padded values from embedded human state
        obstacle_embeddings = obs_mask * self.w_s(static_obs)  # mask out padded values from embedded obstacle state

        X = torch.cat([robot_state_embeddings, human_state_embeddings], dim=1)

        # compute matrix A
        if not self.layerwise_graph:
            normalized_A_dyn = self.compute_similarity_matrix(X, w=self.w_a_dyn)
            normalized_A_stat = self.compute_similarity_matrix(X, obstacle_embeddings, w=self.w_a_stat)
            self.A_dyn = normalized_A_dyn[0, :, :].data.cpu().numpy()
            self.A_stat = normalized_A_stat[0, :, :].data.cpu().numpy()

        next_H = H = X
        for i in range(self.num_layer):
            if self.layerwise_graph:
                A_dyn = self.compute_similarity_matrix(H, w=self.w_a_dyn)
                A_stat = self.compute_similarity_matrix(H, obstacle_embeddings, w=self.w_a_stat)
                next_H = relu(torch.matmul(torch.matmul(A_dyn, H), self.Ws_dyn[i]) +
                              torch.matmul(torch.matmul(A_stat, obstacle_embeddings), self.Ws_stat[i]))
            else:
                next_H = relu(torch.matmul(torch.matmul(normalized_A_dyn, H), self.Ws_dyn[i]) +
                              torch.matmul(torch.matmul(normalized_A_stat, obstacle_embeddings), self.Ws_stat[i]))

            if self.skip_connection:
                next_H += H
            H = next_H

        return next_H


# class RGL(nn.Module):
#     def __init__(self, config, robot_state_dim, human_state_dim):
#         super().__init__()
#         self.multiagent_training = config.gcn.multiagent_training
#         num_layer = config.gcn.num_layer
#         X_dim = config.gcn.X_dim
#         wr_dims = config.gcn.wr_dims
#         wh_dims = config.gcn.wh_dims
#         final_state_dim = config.gcn.final_state_dim
#         gcn2_w1_dim = config.gcn.gcn2_w1_dim
#         similarity_function = config.gcn.similarity_function
#         layerwise_graph = config.gcn.layerwise_graph
#         skip_connection = config.gcn.skip_connection
#
#         # design choice
#
#         # 'gaussian', 'embedded_gaussian', 'cosine', 'cosine_softmax', 'concatenation'
#         self.similarity_function = similarity_function
#         logging.info('self.similarity_func: {}'.format(self.similarity_function))
#         self.robot_state_dim = robot_state_dim
#         self.human_state_dim = human_state_dim
#         self.num_layer = num_layer
#         self.X_dim = X_dim
#         self.layerwise_graph = layerwise_graph
#         self.skip_connection = skip_connection
#
#         self.w_r = mlp(robot_state_dim, wr_dims, last_relu=True)
#         self.w_h = mlp(human_state_dim, wh_dims, last_relu=True)
#
#         if self.similarity_function == 'embedded_gaussian':
#             self.w_a = Parameter(torch.randn(self.X_dim, self.X_dim))
#         elif self.similarity_function == 'concatenation':
#             self.w_a = mlp(2 * X_dim, [2 * X_dim, 1], last_relu=True)
#
#         if num_layer == 1:
#             self.w1 = Parameter(torch.randn(self.X_dim, final_state_dim))
#         elif num_layer == 2:
#             self.w1 = Parameter(torch.randn(self.X_dim, gcn2_w1_dim))
#             self.w2 = Parameter(torch.randn(gcn2_w1_dim, final_state_dim))
#         else:
#             raise NotImplementedError
#
#         # for visualization
#         self.A = None
#
#     def compute_similarity_matrix(self, X):
#         if self.similarity_function == 'embedded_gaussian':
#             A = torch.matmul(torch.matmul(X, self.w_a), X.permute(0, 2, 1))
#             normalized_A = softmax(A, dim=2)
#         elif self.similarity_function == 'gaussian':
#             A = torch.matmul(X, X.permute(0, 2, 1))
#             normalized_A = softmax(A, dim=2)
#         elif self.similarity_function == 'cosine':
#             A = torch.matmul(X, X.permute(0, 2, 1))
#             magnitudes = torch.norm(A, dim=2, keepdim=True)
#             norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
#             normalized_A = torch.div(A, norm_matrix)
#         elif self.similarity_function == 'cosine_softmax':
#             A = torch.matmul(X, X.permute(0, 2, 1))
#             magnitudes = torch.norm(A, dim=2, keepdim=True)
#             norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
#             normalized_A = softmax(torch.div(A, norm_matrix), dim=2)
#         elif self.similarity_function == 'concatenation':
#             indices = [pair for pair in itertools.product(list(range(X.size(1))), repeat=2)]
#             selected_features = torch.index_select(X, dim=1, index=torch.LongTensor(indices).reshape(-1))
#             pairwise_features = selected_features.reshape((-1, X.size(1) * X.size(1), X.size(2) * 2))
#             A = self.w_a(pairwise_features).reshape(-1, X.size(1), X.size(1))
#             normalized_A = A
#         elif self.similarity_function == 'squared':
#             A = torch.matmul(X, X.permute(0, 2, 1))
#             squared_A = A * A
#             normalized_A = squared_A / torch.sum(squared_A, dim=2, keepdim=True)
#         elif self.similarity_function == 'equal_attention':
#             normalized_A = (torch.ones(X.size(1), X.size(1)) / X.size(1)).expand(X.size(0), X.size(1), X.size(1))
#         elif self.similarity_function == 'diagonal':
#             normalized_A = (torch.eye(X.size(1), X.size(1))).expand(X.size(0), X.size(1), X.size(1))
#         else:
#             raise NotImplementedError
#
#         return normalized_A
#
#     def forward(self, state):
#         """
#         Embed current state tensor pair (robot_state, human_states) into a latent space
#         Each tensor is of shape (batch_size, # of agent, features)
#         :param state:
#         :return:
#         """
#         robot_state, human_states = state
#
#         # compute feature matrix X
#         robot_state_embedings = self.w_r(robot_state)
#         human_state_embedings = self.w_h(human_states)
#         X = torch.cat([robot_state_embedings, human_state_embedings], dim=1)
#
#         # compute matrix A
#         normalized_A = self.compute_similarity_matrix(X)
#         self.A = normalized_A[0, :, :].data.cpu().numpy()
#
#         # graph convolution
#         if self.num_layer == 0:
#             state_embedding = X
#         elif self.num_layer == 1:
#             h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1))
#             state_embedding = h1
#         else:
#             # compute h1 and h2
#             if not self.skip_connection:
#                 h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1))
#             else:
#                 h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1)) + X
#             if self.layerwise_graph:
#                 normalized_A2 = self.compute_similarity_matrix(h1)
#             else:
#                 normalized_A2 = normalized_A
#             if not self.skip_connection:
#                 h2 = relu(torch.matmul(torch.matmul(normalized_A2, h1), self.w2))
#             else:
#                 h2 = relu(torch.matmul(torch.matmul(normalized_A2, h1), self.w2)) + h1
#             state_embedding = h2
#
#         return state_embedding
