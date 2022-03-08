from continuous import sac


def get_rl_agent(config):
    type=config.type
    if type=="sac":
        return sac(observation_dim,
            action_dim,
            hidden_dim,
            action_lower=-1,
            action_upper=1,
            gamma=0.99,
            actor_optim="adam",
            critic_optim="adam",
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            tau=0.01,
            init_temperature=0.1,
        )
