import torch
import torch.nn as nn
import torch.nn.functional as F


class base_agent:
    def __init__(
        self,
        observation_dim,
        action_dim,
        activation="tanh",
        critic_num=1,
        hidden_dim=256,
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.actor = self.get_actor(activation)
        self.critic = [self.get_critic() for _ in range(critic_num)]

    def get_actor(self, activation):
        return Net(self.observation_dim, self.hidden_dim, self.action_dim, activation)

    def get_critic(self):
        return Net(self.observation_dim + self.action_dim, self.hidden_dim, 1, None)

    def act(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class Net(nn.Module):
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
