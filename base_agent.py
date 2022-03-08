import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class base_agent:
    def __init__(
        self,
        observation_dim,
        action_dim,
        activation="tanh",
        critic_num=1,
        hidden_dim=256,
        policy_type="stochastic",
        actor_target=True,
        critic_target=True,
        gamma=0.99,
        actor_optim="adam",
        critic_optim="adam",
        actor_lr=3e-4,
        critic_lr=3e-4,
        tau=0.01
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.critic_num = critic_num
        self.gamma = gamma
        self.tau=tau
        self.critic_criterion = nn.MSELoss()

        """actor"""
        self.actor = self.get_new_actor(activation, policy_type)
        if actor_optim == "adam":
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=actor_lr
            )
        else:
            raise TypeError(f"optimizer type : {critic_optim} not supported")
        if actor_target:
            self.actor_target = copy.deepcopy(self.actor)
        else:
            self.actor_target = None

        """critic"""
        self.critic = [self.get_new_critic() for _ in range(self.critic_num)]
        if critic_optim == "adam":
            self.critic_optimizer = [
                torch.optim.Adam(model.parameters(), lr=critic_lr)
                for model in self.critic
            ]
        else:
            raise TypeError(f"optimizer type : {critic_optim} not supported")
        if self.critic_num == 1:
            self.critic = self.critic[0]
            self.critic_optimizer = self.critic_optimizer[0]
        if critic_target:
            self.critic_target = copy.deepcopy(self.critic)
        else:
            self.critic_target = None

    def get_new_actor(self, activation, policy_type):
        if policy_type == "stochastic":
            return StochasticPolicyNet(
                self.observation_dim, self.hidden_dim, self.action_dim, activation
            ).to(device)
        else:
            return DeterministicPolicyNet(
                self.observation_dim, self.hidden_dim, self.action_dim, activation
            ).to(device)

    def get_new_critic(self):
        return CriticNet(
            self.observation_dim + self.action_dim, self.hidden_dim, 1, None
        ).to(device)

    def soft_update_target(self):
        if self.actor_target is not None:
            for i, j in zip(self.actor_target.parameters(), self.actor.parameters()):
                i.data = (1 - self.tau) * i.data + self.tau * j.data

        if self.critic_target is not None:
            for idx in range(self.critic_num):
                for i, j in zip(
                    self.critic_target[idx].parameters(), self.critic[idx].parameters()
                ):
                    i.data = (1 - self.tau) * i.data + self.tau * j.data

    def hard_update_target(self):
        if self.actor_target is not None:
            self.actor_target.load_state_dict(self.actor.state_dict())

        if self.critic_target is not None:
            for idx in range(self.critic_num):
                self.critic_target[idx].load_state_dict(self.critic[idx].state_dict())

    def act(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class DeterministicPolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        if activation == "tanh":
            self.output_activation = nn.Tanh()
        elif activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = None

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


class StochasticPolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, output_dim)
        self.std = nn.Linear(hidden_dim, output_dim)
        if activation == "tanh":
            self.output_activation = nn.Tanh()
        elif activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = None

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.mu(x)
        std = self.std(x)
        dist = torch.distributions.normal.Normal(mu, std)
        return mu, std, dist


class CriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, s, a):
        x = torch.cat((s, a), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
