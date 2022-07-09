import numpy as np
import os
import pickle

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
        save_idx=min(self.storage_index, self.size)
        print(save_idx)
        data = {
            "states": self.states[:save_idx],
            "actions": self.actions[:save_idx],
            "rewards": self.rewards[:save_idx],
            "next_states": self.next_states[:save_idx],
            "dones": self.dones[:save_idx],
        }
        if based_on_transition_num:
            file_name=f"transition_num{expert_data_num}.pkl"
        else:
            file_name=f"episode_num{expert_data_num}.pkl"
        print(os.path.join(path, file_name))
        with open(os.path.join(path, file_name), "wb") as handle:
            pickle.dump(data, handle)

    def __len__(self):
        return min(self.storage_index, self.size)
