from continuous import *


def get_rl_agent(env, config):
    if config.algo == "sac":
        return sac(
            observation_dim=env.get_observation_dim(),
            action_dim=env.get_action_dim(),
            action_lower=min(env.action_space.low),
            action_upper=max(env.action_space.high),
            hidden_dim=config.hidden_dim,
            gamma=config.gamma,
            actor_optim=config.actor_optim,
            critic_optim=config.critic_optim,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            alpha_lr=config.alpha_lr,
            tau=config.tau,
            batch_size=config.batch_size,
            use_ounoise=config.use_ounoise,
            log_alpha_init=config.log_alpha_init,
        )
    elif config.algo == "ddpg":
        return ddpg(
            observation_dim=env.get_observation_dim(),
            action_dim=env.get_action_dim(),
            action_lower=min(env.action_space.low),
            action_upper=max(env.action_space.high),
            hidden_dim=config.hidden_dim,
            gamma=config.gamma,
            actor_optim=config.actor_optim,
            critic_optim=config.critic_optim,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            tau=config.tau,
            batch_size=config.batch_size,
            use_ounoise=config.use_ounoise,
        )
    elif config.algo == "td3":
        return td3(
            observation_dim=env.get_observation_dim(),
            action_dim=env.get_action_dim(),
            action_lower=min(env.action_space.low),
            action_upper=max(env.action_space.high),
            hidden_dim=config.hidden_dim,
            gamma=config.gamma,
            actor_optim=config.actor_optim,
            critic_optim=config.critic_optim,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            tau=config.tau,
            batch_size=config.batch_size,
            use_ounoise=config.use_ounoise,
            policy_update_delay=config.policy_update_delay
        )
    else:
        raise TypeError(f"rl agent type : {config.algo} not supported")
