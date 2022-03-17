import argparse
import yaml
from rl_algo import get_rl_agent
from envs import get_env
from replay_buffer import get_replay_buffer
from training_stage import get_training_stage


def get_config():
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument(
        "--env",
        type=str,
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
        help="Path of model weight",
    )
    parser.add_argument("--episodes", type=int, help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, help="Seed, default: 1")
    parser.add_argument(
        "--save_every",
        type=int,
        help="Saves the network every x epochs, default: 25",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size, default: 256")
    parser.add_argument("--hidden_dim", type=int, help="dimension of hidden layer 256")
    # parser.add_argument(
    #     "-o",
    #     "--out_of_dist",
    #     help="whether test out of distribution",
    #     default=False,
    #     action="store_true",
    # )
    args = parser.parse_args()
    with open(f"{args.algo}.yml", "r") as f:
        hyper = yaml.safe_load(f)
    hyper.update(vars(args))
    return hyper


if __name__ == "__main__":
    config = get_config()
    print(str(config))
    # agent = get_rl_agent(config)
    # env = get_env(config.env)
    # storage = get_replay_buffer()
    # training_fn=get_training_stage()
    # training_fn.train(agent,env,storage)
