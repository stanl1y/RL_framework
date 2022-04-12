import wandb


class off_policy_training_stage:
    def __init__(self, config):
        self.episodes = config.episodes
        self.buffer_warmup = config.buffer_warmup
        self.buffer_warmup_step = config.buffer_warmup_step
        self.algo = config.algo
        self.env_id = config.env
        self.save_weight_period = config.save_weight_period
        self.continue_training=config.continue_training
        wandb.init(
            project="RL_Implementation",
            name=f"{self.algo}_{self.env_id}",
            config=config,
        )

    def test(self, agent, env):
        total_reward = 0
        for i in range(3):
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state, testing=True)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = next_state
        total_reward /= 3
        return total_reward

    def start(self, agent, env, storage):
        self.train(agent, env, storage)

    def train(self, agent, env, storage):
        if self.continue_training:
            agent.load_weight(self.env_id)
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
        best_testing_reward = -1e7
        best_episode = 0
        for i in range(self.episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                storage.store(state, action, reward, next_state, done)
                loss_info = agent.update(storage)
                wandb.log(loss_info, commit=False)
                state = next_state
            wandb.log(
                {
                    "training_reward": total_reward,
                    "episode_num": i,
                    "buffer_size": len(storage),
                }
            )

            if i % 5 == 0:
                testing_reward = self.test(agent, env)
                if testing_reward > best_testing_reward:
                    agent.cache_weight()
                    best_testing_reward = testing_reward
                    best_episode = i
                wandb.log({"testing_reward": testing_reward, "testing_episode_num": i})
            if i % self.save_weight_period == 0:
                agent.save_weight(
                    best_testing_reward, self.algo, self.env_id, best_episode
                )
        agent.save_weight(best_testing_reward, self.algo, self.env_id, best_episode)
