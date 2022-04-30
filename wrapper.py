import gym
import numpy as np
from gym.wrappers import FilterObservation, FlattenObservation


class BasicWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def get_observation_dim(self):
        return self.env.observation_space.shape[0]

    def get_action_dim(self):
        return self.env.action_space.shape[0]


class GymRoboticWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = FlattenObservation(
            FilterObservation(env, ["observation", "desired_goal"])
        )

    def get_observation_dim(self):
        return self.env.observation_space.shape[0]

    def get_action_dim(self):
        return self.env.action_space.shape[0]



class HERWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def get_observation_dim(self):
        observation_space = self.env.observation_space
        return (
            observation_space["desired_goal"].shape[0]
            + observation_space["observation"].shape[0]
        )

    def get_action_dim(self):
        return self.env.action_space.shape[0]
