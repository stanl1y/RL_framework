import gym
from wrapper import *
def get_env(env_name,wrapper_type):
    if env_name in ['RoboSumo-Ant-vs-Ant-v0','RoboSumo-Ant-vs-Bug-v0','RoboSumo-Ant-vs-Spider-v0','RoboSumo-Bug-vs-Ant-v0','RoboSumo-Bug-vs-Bug-v0','RoboSumo-Bug-vs-Spider-v0','RoboSumo-Spider-vs-Ant-v0','RoboSumo-Spider-vs-Bug-v0','RoboSumo-Spider-vs-Spider-v0']
        import robosumo.envs
    env=gym.make(env_name)
    if wrapper_type=="basic":
        return BasicWrapper(env)
    elif wrapper_type=="gym_robotic":
        return GymRoboticWrapper(env)
    elif wrapper_type=="her":
        return HERWrapper(env)
    else:
        raise TypeError(f"env wrapper type : {wrapper_type} not supported")