import gym
from wrapper import *
def get_env(env_name,wrapper_type):
    env=gym.make(env_name)
    if wrapper_type=="basic":
        return BasicWrapper(env)
    elif wrapper_type=="gym_robotic":
        return GymRoboticWrapper(env)
    else:
        raise TypeError(f"env wrapper type : {wrapper_type} not supported")