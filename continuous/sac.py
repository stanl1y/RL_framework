from base_agent import base_agent


class sac(base_agent):
    def __init__(self, observation_dim, action_dim, action_lower=-1, action_upper=1):
        if action_lower == -1 and action_upper == 1:
            activation = "tanh"
        elif action_lower == 0 and action_upper == 1:
            activation = "sigmoid"
        super().__init__(observation_dim, action_dim, activation)
