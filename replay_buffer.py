import numpy as np
import os
import pickle


def get_replay_buffer(env, config):
    if config.buffer_type == "normal":
        return normal_replay_buffer(
            size=config.buffer_size,
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
        )
    else:
        raise TypeError(f"replay buffer type : {config.buffer_type} not supported")


class normal_replay_buffer:
    def __init__(self, size, state_dim, action_dim):
        self.size = size
        self.storage_index = 0
        self.states = np.empty((size, state_dim))
        self.actions = np.empty((size, action_dim))
        self.rewards = np.empty((size, 1))
        self.next_states = np.empty((size, state_dim))
        self.dones = np.empty((size, 1))

    def store(self, s, a, r, ss, d):
        index = self.storage_index % self.size
        self.states[index] = s
        self.actions[index] = a
        self.rewards[index] = r
        self.next_states[index] = ss
        self.dones[index] = d
        self.storage_index += 1

    def sample(self, batch_size):
        index = np.random.randint(min(self.storage_index, self.size), size=batch_size)
        return (
            self.states[index],
            self.actions[index],
            self.rewards[index],
            self.next_states[index],
            self.dones[index],
        )

    def write_storage(self, based_on_transition_num, expert_data_num, algo, env_id):
        path = f"./saved_expert_transition/{env_id}/{algo}/"
        if not os.path.isdir(path):
            os.makedirs(path)
        data = {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "dones": self.dones,
        }
        if based_on_transition_num:
            file_name=f"transition_num{expert_data_num}.pkl"
        else:
            file_name=f"episode_num{expert_data_num}.pkl"

        with open(os.path.join(path, file_name), "wb") as handle:
            pickle.dump(data, handle)

    def __len__(self):
        return min(self.storage_index, self.size)
