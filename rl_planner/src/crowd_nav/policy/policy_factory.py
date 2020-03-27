# from crowd_nav.policy.cadrl import CADRL
# from crowd_nav.policy.lstm_rl import LstmRL
# from crowd_nav.policy.sarl import SARL
# from crowd_nav.policy.gcn import GCN
from crowd_nav.policy.model_predictive_rl import ModelPredictiveRL
from crowd_nav.policy.orca import ORCA
# from crowd_nav.policy.linear import Linear
# from crowd_nav.policy.socialforce import SocialForce


def none_policy():
    return None

policy_factory = dict()
# policy_factory['cadrl'] = CADRL
# policy_factory['lstm_rl'] = LstmRL
# policy_factory['sarl'] = SARL
# policy_factory['gcn'] = GCN
policy_factory['model_predictive_rl'] = ModelPredictiveRL
# policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
# policy_factory['socialforce'] = SocialForce
policy_factory['none'] = none_policy
