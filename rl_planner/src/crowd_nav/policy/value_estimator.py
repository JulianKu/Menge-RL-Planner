import torch.nn as nn
from crowd_nav.policy.helpers import mlp


class ValueEstimator(nn.Module):
    def __init__(self, config, graph_model):
        super().__init__()
        self.graph_model = graph_model
        self.value_network = mlp(config.gcn.X_dim, config.model_predictive_rl.value_network_dims)

    def forward(self, state):
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """

        assert len(state[0].shape) == 3

        if isinstance(state[1], tuple):
            # human_states contain mask and zero padded state batch
            if isinstance(state[1][1], tuple):
                # human_states also contain identifiers for each human
                assert len(state[1][1][0].shape) == 3
            else:
                assert len(state[1][0].shape) == 3
        else:
            assert len(state[1].shape) == 3
        # state[2] = obstacles -> global position remains unchanged
        if isinstance(state[2], tuple):
            # obstacles contain mask and zero padded state batch
            assert len(state[2][1].shape) == 3
        else:
            assert len(state[2].shape) == 3

        # only use the feature of robot node as state representation, disregarding identifiers
        state_embedding = self.graph_model(state)[:, 0, :]
        value = self.value_network(state_embedding)
        return value
