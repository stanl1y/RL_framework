import gym
import numpy as np


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
        self.env = env

    def get_observation_dim(self):
        observation_space = self.env.observation_space
        return (
            observation_space["achieved_goal"].shape[0]
            + observation_space["desired_goal"].shape[0]
            + observation_space["observation"].shape[0]
        )

    def get_action_dim(self):
        return self.env.action_space.shape[0]

    def reset(self):
        state = self.env.reset()
        state = np.append(
            state["observation"],
            [state["achieved_goal"], state["desired_goal"]],
        )
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = np.append(
            next_state["observation"],
            [next_state["achieved_goal"], next_state["desired_goal"]],
        )
        return next_state, reward, done, info
