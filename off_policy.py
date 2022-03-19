import wandb
class off_policy_training_stage:
    def __init__(self, config):
        self.episodes = config.episodes
        self.buffer_warmup = config.buffer_warmup
        self.buffer_warmup_step = config.buffer_warmup_step
        wandb.init(project="RL_Implementation", name=f"{config.algo}_{config.env}", config=config)

    def test(self,agent,env):
        state = env.reset()
        done = False
        total_reward=0
        while not done:
            action = agent.act(state, testing=True)
            next_state, reward, done, info = env.step(action)
            total_reward+=reward
            state = next_state
        return total_reward
    def train(self, agent, env, storage):
        if self.buffer_warmup:
            state = env.reset()
            done = False
            while len(storage) < self.buffer_warmup_step:
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                storage.store(state, action, reward, next_state, done)
                if done:
                    state = env.reset()
                    done = False
                else:
                    state = next_state
        for i in range(self.episodes):
            state = env.reset()
            done = False
            total_reward=0
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                total_reward+=reward
                storage.store(state, action, reward, next_state, done)
                loss_info=agent.update(storage)
                wandb.log(loss_info,commit=False)
                state = next_state
            wandb.log({"training_reward":total_reward,"episode_num":i,"buffer_size":len(storage)})

            if i%2==0:
                testing_reward=self.test(agent,env)
                wandb.log({"testing_reward":testing_reward,"testing_episode_num":i})


