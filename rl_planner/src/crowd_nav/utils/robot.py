from crowd_nav.policy.policy_factory import policy_factory


class Robot(object):
    def __init__(self, config, section):
        self.policy = policy_factory[getattr(config, section).policy]()
        self.time_step = None
        self.kinematics = self.policy.kinematics if self.policy is not None else None

    def set_policy(self, policy):
        if self.time_step is None:
            raise ValueError('Time step is None')
        policy.set_time_step(self.time_step)
        self.policy = policy
        self.kinematics = policy.kinematics
