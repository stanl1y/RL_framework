from .base_agent import base_agent
from exploration.ounoise import OUNoise
import torch.nn as nn
import copy
import torch
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class sac(base_agent):
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
        alpha_lr=3e-4,
        tau=0.01,
        batch_size=256,
        use_ounoise=False,
        log_alpha_init=0,
    ):

        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
            action_lower=action_lower,
            action_upper=action_upper,
            critic_num=2,
            hidden_dim=hidden_dim,
            policy_type="stochastic",
            actor_target=False,
            critic_target=True,
            gamma=gamma,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            tau=tau,
            batch_size=batch_size,
        )
        self.target_entropy = -action_dim
        self.log_alpha = nn.Parameter(torch.ones(1).to(device) * log_alpha_init)
        self.log_alpha.requires_grad = True
        self.alpha_lr = alpha_lr
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.best_log_alpha_optimizer = copy.deepcopy(self.log_alpha_optimizer)
        self.ounoise = (
            OUNoise(
                action_dimension=action_dim,
                mu=0,
                scale=self.action_scale,
            )
            if use_ounoise
            else None
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, state, testing=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, mu = self.actor.sample(state)
        if not testing:
            if self.ounoise is not None:
                return action[0].cpu().numpy() + self.ounoise.noise()
            else:
                return action[0].cpu().numpy()
        else:
            return mu[0].cpu().numpy()

    def update_critic(self, state, action, reward, next_state, done):

        """compute target value"""
        with torch.no_grad():
            next_action, next_log_prob, next_mu = self.actor.sample(next_state)
            target_q_val = [
                critic_target(next_state, next_action)
                for critic_target in self.critic_target
            ]
            target_value = reward + self.gamma * (1 - done) * (
                torch.min(target_q_val[0], target_q_val[1]) - self.alpha * next_log_prob
            )

        """compute loss and update"""
        q_val = [critic(state, action) for critic in self.critic]
        critic_loss = [self.critic_criterion(pred, target_value) for pred in q_val]

        for i in range(2):
            self.critic_optimizer[i].zero_grad()
            critic_loss[i].backward()
            self.critic_optimizer[i].step()
        return {
            "critic0_loss": critic_loss[0],
            "critic1_loss": critic_loss[1],
        }

    def update_actor(self, state):
        action, log_prob, mu = self.actor.sample(state)
        q_val = [critic(state, action) for critic in self.critic]
        entropy_loss = -self.alpha.detach() * log_prob
        actor_loss = (-(torch.min(q_val[0], q_val[1]) + entropy_loss)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        """update alpha"""
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return {
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
            "entropy_loss": entropy_loss.mean(),
            "log_prob": log_prob.mean(),
        }

    def update(self, storage):

        """sample data"""
        state, action, reward, next_state, done = storage.sample(self.batch_size)
        if not storage.to_tensor:
            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            reward = torch.FloatTensor(reward)
            next_state = torch.FloatTensor(next_state)
            done = torch.FloatTensor(done)
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        done = done.to(device)

        """update model"""
        critic_loss = self.update_critic(state, action, reward, next_state, done)
        actor_loss = self.update_actor(state)
        self.soft_update_target()
        return {**critic_loss, **actor_loss}


    def cache_weight(self):
        self.best_actor.load_state_dict(self.actor.state_dict())
        self.best_actor_optimizer.load_state_dict(self.actor_optimizer.state_dict())
        for idx in range(self.critic_num):
            self.best_critic[idx].load_state_dict(self.critic[idx].state_dict())
            self.best_critic_optimizer[idx].load_state_dict(
                self.critic_optimizer[idx].state_dict()
            )
        self.best_log_alpha = self.log_alpha
        self.best_log_alpha_optimizer.load_state_dict(
            self.log_alpha_optimizer.state_dict()
        )

    def save_weight(
        self,
        best_testing_reward,
        algo,
        env_id,
        episodes,
        log_name="",
        delete_prev_weight=True,
    ):
        dir_path = f"./trained_model/{algo}/{env_id}/"
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        file_path = os.path.join(
            dir_path,
            f"episode{episodes}_reward{round(best_testing_reward,3)}{log_name}.pt",
        )

        if file_path == self.previous_checkpoint_path:
            return

        data = {
            "episodes": episodes,
            "actor_state_dict": self.best_actor.cpu().state_dict(),
            "actor_optimizer_state_dict": self.best_actor_optimizer.state_dict(),
            "log_alpha_state_dict": self.best_log_alpha.cpu(),
            "log_alpha_optimizer_state_dict": self.best_log_alpha_optimizer.state_dict(),
            "reward": best_testing_reward,
        }

        for idx, (model, optimizer) in enumerate(
            zip(self.best_critic, self.best_critic_optimizer)
        ):
            data[f"critic_state_dict{idx}"] = model.cpu().state_dict()
            data[f"critic_optimizer_state_dict{idx}"] = optimizer.state_dict()

        torch.save(data, file_path)
        if delete_prev_weight:
            try:
                os.remove(self.previous_checkpoint_path)
            except:
                pass
            self.previous_checkpoint_path = file_path

    def load_weight(self, algo="sac", env_id=None, path=None):
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

        self.log_alpha = checkpoint["log_alpha_state_dict"]
        self.log_alpha.requires_grad = False
        self.log_alpha = self.log_alpha.to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.log_alpha_optimizer.load_state_dict(
            checkpoint["log_alpha_optimizer_state_dict"]
        )

        for idx in range(self.critic_num):
            self.critic[idx].load_state_dict(checkpoint[f"critic_state_dict{idx}"])
            self.critic[idx] = self.critic[idx].to(device)
            self.critic_target[idx].load_state_dict(
                checkpoint[f"critic_state_dict{idx}"]
            )
            self.critic_target[idx] = self.critic_target[idx].to(device)
            self.best_critic[idx].load_state_dict(checkpoint[f"critic_state_dict{idx}"])
            self.best_critic[idx] = self.best_critic[idx].to(device)
            self.critic_optimizer[idx].load_state_dict(
                checkpoint[f"critic_optimizer_state_dict{idx}"]
            )
            self.best_critic_optimizer[idx].load_state_dict(
                checkpoint[f"critic_optimizer_state_dict{idx}"]
            )
