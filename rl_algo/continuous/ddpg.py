from .base_agent import base_agent
from exploration.ounoise import OUNoise
import torch.nn as nn
import copy
import torch
import os
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ddpg(base_agent):
    def __init__(
        self,
        observation_dim,
        action_dim,
        action_lower=-1,
        action_upper=1,
        hidden_dim=256,
        gamma=0.99,
        actor_optim="adam",
        critic_optim="adam",
        actor_lr=3e-4,
        critic_lr=3e-4,
        tau=0.01,
        batch_size=256,
        use_ounoise=True,
    ):

        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
            action_lower=action_lower,
            action_upper=action_upper,
            critic_num=1,
            hidden_dim=hidden_dim,
            policy_type="deterministic",
            actor_target=True,
            critic_target=True,
            gamma=gamma,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            tau=tau,
            batch_size=batch_size,
        )
        self.ounoise = (
            OUNoise(
                action_dimension=action_dim,
                mu=0,
                scale=self.action_scale,
            )
            if use_ounoise
            else None
        )

    def act(self, state, testing=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state)
        if not testing:
            if self.ounoise is not None:
                action = action[0].cpu().numpy() + self.ounoise.noise()
            else:
                action = action[0].cpu().numpy() + np.random.normal(
                    self.action_bias, self.action_scale / 2, self.action_dim
                )
            return np.clip(action, self.action_lower, self.action_upper)
        else:
            return action[0].cpu().numpy()

    def update_critic(self, state, action, reward, next_state, done):

        """compute target value"""
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q_val = self.critic_target(next_state, next_action)
            target_value = reward + self.gamma * (1 - done) * target_q_val

        """compute loss and update"""
        pred = self.critic(state, action)
        critic_loss = self.critic_criterion(pred, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def update_actor(self, state):
        action_tilda = self.actor(state)
        q_val = self.critic(state, action_tilda)
        actor_loss = -q_val.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss

    def update(self, storage):

        """sample data"""
        state, action, reward, next_state, done = storage.sample(self.batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)

        """update model"""
        critic_loss = self.update_critic(state, action, reward, next_state, done)
        actor_loss = self.update_actor(state)
        self.soft_update_target()
        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }  # this dict will be recorded by wandb

    def cache_weight(self):
        self.best_actor.load_state_dict(self.actor.state_dict())
        self.best_actor_optimizer.load_state_dict(self.actor_optimizer.state_dict())

        self.best_critic.load_state_dict(self.critic.state_dict())
        self.best_critic_optimizer.load_state_dict(self.critic_optimizer.state_dict())

    def save_weight(self, best_testing_reward, algo, env_id, episodes):
        dir_path = f"./trained_model/{algo}/{env_id}/"
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        file_path = os.path.join(
            dir_path, f"episode{episodes}_reward{round(best_testing_reward,3)}.pt"
        )

        if file_path == self.previous_checkpoint_path:
            return

        data = {
            "episodes": episodes,
            "actor_state_dict": self.best_actor.cpu().state_dict(),
            "actor_optimizer_state_dict": self.best_actor_optimizer.state_dict(),
            "reward": best_testing_reward,
            "critic_state_dict": self.best_critic.cpu().state_dict(),
            "critic_optimizer_state_dict": self.best_critic_optimizer.state_dict(),
        }

        torch.save(data, file_path)
        try:
            os.remove(self.previous_checkpoint_path)
        except:
            pass
        self.previous_checkpoint_path = file_path

    def load_weight(self, algo="ddpg", env_id=None, path=None):
        if path is None:
            assert env_id is not None
            path = f"./trained_model/{algo}/{env_id}/"
            assert os.path.isdir(path)
            onlyfiles = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            ]
            path = onlyfiles[-1]
        else:
            assert os.path.isfile(path)

        checkpoint = torch.load(path)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor = self.actor.to(device)
        self.best_actor.load_state_dict(checkpoint["actor_state_dict"])
        self.best_actor = self.best_actor.to(device)
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.best_actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer_state_dict"]
        )
        self.critic_actor.load_state_dict(checkpoint["actor_state_dict"])

        self.critic.load_state_dict(checkpoint[f"critic_state_dict"])
        self.critic = self.critic.to(device)
        self.critic_target.load_state_dict(checkpoint[f"critic_state_dict"])
        self.critic_target = self.critic_target.to(device)
        self.best_critic.load_state_dict(checkpoint[f"critic_state_dict"])
        self.best_critic = self.best_critic.to(device)
        self.critic_optimizer.load_state_dict(
            checkpoint[f"critic_optimizer_state_dict"]
        )
        self.best_critic_optimizer.load_state_dict(
            checkpoint[f"critic_optimizer_state_dict"]
        )
