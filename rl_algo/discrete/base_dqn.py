import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class base_dqn:
    def __init__(
        self,
        observation_dim,
        action_num,
        hidden_dim=256,
        network_type="dueling",
        noisy_network=False,
        soft_update_target=True,
        gamma=0.99,
        optim="adam",
        lr=3e-4,
        tau=0.01,
        batch_size=256,
    ):
        self.observation_dim = observation_dim
        self.action_num = action_num
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()

        """actor"""
        self.q_network = self.get_new_network(network_type)  # vanilla, dueling

        if optim == "adam":
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        else:
            raise TypeError(f"optimizer type : {optim} not supported")

        self.best_q_network = copy.deepcopy(self.q_network)
        self.best_optimizer = copy.deepcopy(self.optimizer)

        self.q_network_target = copy.deepcopy(self.q_network)

        self.previous_checkpoint_path = None
        self.update_target = self.soft_update_target if soft_update_target else self.hard_update_target

    def get_new_network(self, network_type):
        if network_type == "vanilla":
            return VanillaDQN(
                self.observation_dim,
                self.hidden_dim,
                self.action_num,
            ).to(device)
        else:
            return DuelingDQN(
                self.observation_dim,
                self.hidden_dim,
                self.action_num,
            ).to(device)

    def soft_update_target(self):
        for i, j in zip(
            self.q_network_target.parameters(), self.q_network.parameters()
        ):
            i.data = (1 - self.tau) * i.data + self.tau * j.data

    def hard_update_target(self):
        self.q_network_target.load_state_dict(self.q_network.state_dict())

    def cache_weight(self):
        self.best_q_network = copy.deepcopy(self.q_network)
        self.best_optimizer = copy.deepcopy(self.optimizer)

    def save_weight(self, best_testing_reward, algo, env_id, episodes):
        path = f"./trained_model/{algo}/{env_id}/"
        if not os.path.isdir(path):
            os.makedirs(path)
        data = {
            "episodes": episodes,
            "dqn_state_dict": self.best_q_network.state_dict(),
            "dqn_optimizer_state_dict": self.best_optimizer.state_dict(),
            "reward": best_testing_reward,
        }

        file_path = os.path.join(
            path, f"episode{episodes}_reward{round(best_testing_reward,3)}.pt"
        )
        torch.save(data, file_path)
        try:
            os.remove(self.previous_checkpoint_path)
        except:
            pass
        self.previous_checkpoint_path = file_path

    def load_weight(self, algo=None, env_id=None, path=None):
        if path is None:
            assert algo is not None and env_id is not None
            path = f"./trained_model/{algo}/{env_id}/"
            assert os.path.isdir(path)
            onlyfiles = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            ]
            path = onlyfiles[0]
        else:
            assert os.path.isfile(path)

        checkpoint = torch.load(path)

        self.q_network.load_state_dict(checkpoint["dqn_state_dict"])
        self.best_q_network.load_state_dict(checkpoint["dqn_state_dict"])
        self.optimizer.load_state_dict(checkpoint["dqn_optimizer_state_dict"])
        self.best_optimizer.load_state_dict(checkpoint["dqn_optimizer_state_dict"])

    def act(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class VanillaDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4_value = nn.Linear(hidden_dim, 1)
        self.fc4_advantage = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x_v = self.fc4_value(x)
        x_a = self.fc4_advantage(x)
        return x_v + (x_a - torch.mean(x_a, axis=1, keepdim=True))