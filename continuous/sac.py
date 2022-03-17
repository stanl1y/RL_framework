from base_agent import base_agent
import torch

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
    ):
        if action_lower == -1 and action_upper == 1:
            activation = "tanh"
        elif action_lower == 0 and action_upper == 1:
            activation = "sigmoid"
        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
            activation=activation,
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
        )
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, state, testing=False):
        state = torch.tensor(state).to(device)
        if not testing:
            return self.actor.forward_and_sample(state)
        else:
            mu, std = self.actor.forward(state)
            return mu

    def update_critic(self, state, action, reward, next_state, done):

        """compute target value"""
        with torch.no_grad():
            mu, std, dist = self.actor(next_state)
            next_action = dist.sample()
            next_log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_q_val = [
                critic_target(next_state, next_action)
                for critic_target in self.critic_target
            ]
            target_value = reward + self.gamma * (1 - done) * (
                min(target_q_val[0], target_q_val[1]) - self.alpha * next_log_prob
            )

        """compute loss and update"""
        q_val = [critic(state, action) for critic in self.critic]
        critic_loss = [self.critic_criterion(pred, target_value) for pred in q_val]

        for i in range(2):
            self.critic_optimizer[i].zero_grad()
            critic_loss[i].backward()
            self.critic_optimizer[i].step()

    def update_actor(self, state):

        mu, std, dist = self.actor(state)
        action_tilda = dist.rsample()
        q_val = [critic(state, action_tilda) for critic in self.critic]
        log_prob = dist.log_prob(action_tilda).sum(-1, keepdim=True)
        entropy_loss = -self.alpha.detach() * log_prob
        actor_loss = (-(min(q_val[0], q_val[1]) + entropy_loss)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        """update alpha"""
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, storage, batch_size=128):

        """sample data"""
        state, action, reward, next_state, done = storage.sample(batch_size)
        state = torch.from_numpy(state).to(device)
        action = torch.from_numpy(action).to(device)
        reward = torch.from_numpy(reward).to(device)
        next_state = torch.from_numpy(next_state).to(device)
        done = torch.from_numpy(done).to(device)

        """update model"""
        self.update_critic(state, action, reward, next_state, done)
        self.update_actor(state)
        self.soft_update_target()
