import gym
from .wrapper import *
def get_env(env_name,wrapper_type):
    if env_name in ["highway-v0","merge-v0","roundabout-v0","parking-v0","intersection-v0","racetrack-v0"]:
        '''
        highway_env:
        https://github.com/eleurent/highway-env
        '''
        import highway_env
    env=gym.make(env_name)
    if wrapper_type=="basic":
        return BasicWrapper(env)
    elif wrapper_type=="gym_robotic":
        return GymRoboticWrapper(env)
    elif wrapper_type=="her":
        return HERWrapper(env)
    else:
        raise TypeError(f"env wrapper type : {wrapper_type} not supported")