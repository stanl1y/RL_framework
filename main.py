import argparse
from rl_algo import get_rl_agent
from envs import get_env
from replay_buffer import get_replay_buffer
from training_stage import get_training_stage


def get_config():
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument(
        "--env",
        type=str,
        default="cont_env_toy",
        help="Gym environment name, default: CartPole-v0",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        help="which RL algo",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="./bc_weight",
        help="Path of model weight",
    )
    parser.add_argument(
        "--episodes", type=int, default=256, help="Number of episodes, default: 100"
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Saves the network every x epochs, default: 25",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size, default: 256"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="dimension of hidden layer 256"
    )
    parser.add_argument(
        "-o",
        "--out_of_dist",
        help="whether test out of distribution",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = get_config()
    agent = get_rl_agent(config)
    env = get_env(config.env)
    storage = get_replay_buffer()
    training_fn=get_training_stage()
    training_fn.train(agent,env,storage)
