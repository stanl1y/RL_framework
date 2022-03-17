from continuous import sac


def get_rl_agent(env, config):
    algo = config.algo
    if type == "sac":
        return sac(
            observation_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            action_lower=min(env.action_space.low),
            action_upper=max(env.action_space.high),
            hidden_dim=config.hidden_dim,
            gamma=config.gamma,
            actor_optim=config.actor_optim,
            critic_optim=config.critic_optim,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            alpha_lr=config.alpha_lr,
            tau=config.tau
        )
