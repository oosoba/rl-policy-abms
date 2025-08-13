import numpy as np
import torch
from abc import ABC, abstractmethod
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import os

from pytorch_arch.networks import SensoriumNetwork, ActionPolicyNetwork, ValueNetwork

class PytorchAgent(ABC):
    def __init__(self, name, env, latent_dim=128, max_episode_length=400, tboard_path="./log_torch"):
        self.name = name
        self.env = env
        self.max_episode_length = max_episode_length
        self.latent_dim = latent_dim

        self.model_path = f"{tboard_path}/train_{self.name}"
        os.makedirs(self.model_path, exist_ok=True)
        self.summary_writer = SummaryWriter(self.model_path)

        if hasattr(self.env, "state_space_size") and hasattr(self.env, "action_space_size"):
            self.s_size = self.env.state_space_size
            self.a_size = self.env.action_space_size
        else: # Fallback for standard Gym envs
            self.s_size = self.env.observation_space.shape[0]
            # Assuming a discrete action space for now, which is consistent with the original project
            self.a_size = 1 # The original project mostly deals with binary action spaces

        self.sensorium = SensoriumNetwork(self.s_size, self.latent_dim)
        self.actor = ActionPolicyNetwork(self.latent_dim, self.a_size)
        self.value_net = ValueNetwork(self.latent_dim)

        self.models = {'sensorium': self.sensorium, 'actor': self.actor, 'value_net': self.value_net}

        self.episode_buffer = {
            'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []
        }
        self.last_total_return = 0.0
        self.total_epoch_count = 0
        self.agent_age = 0

    def reset_buffer(self):
        self.episode_buffer = {
            'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []
        }

    def episode_length(self):
        return len(self.episode_buffer['states'])

    @abstractmethod
    def act(self, state):
        pass

    def play(self, terminal_reward=-10.0):
        self.reset_buffer()
        self.last_total_return = 0.
        self.agent_age += 1
        d = False
        reset_output = self.env.reset()
        if isinstance(reset_output, tuple):
            s, _ = reset_output
        else:
            s = reset_output

        while (self.episode_length() < self.max_episode_length) and not d:
            s_tensor = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            act_pn = self.act(s_tensor)

            # In the original envs, step returns s1, r, d, {}, so rest is empty
            s1, r, d, *rest = self.env.step(act_pn)
            if d and self.episode_length() < self.max_episode_length -1:
                r = terminal_reward

            self.episode_buffer['states'].append(s)
            self.episode_buffer['actions'].append(act_pn)
            self.episode_buffer['rewards'].append(float(r))
            self.episode_buffer['next_states'].append(s1)
            self.episode_buffer['dones'].append(d)
            self.last_total_return += float(r)
            s = s1

    @abstractmethod
    def train(self, gamma):
        pass

    def work(self, num_epochs, gamma=0.95):
        print(f"Starting agent {self.name}")
        for tk in range(int(num_epochs)):
            print(f"\rEpoch no.: {tk+1}/{num_epochs}", end="")
            self.total_epoch_count += 1

            self.play()

            if self.episode_length() > 0:
                stats = self.train(gamma=gamma)
                self.log_summary(tk, stats)
                if (tk+1) % 100 == 0:
                    print(f"\nEpoch {tk+1}: Return = {self.last_total_return:.2f}, Stats = {stats}")
                    self.save_models()


    def log_summary(self, step, stats):
        self.summary_writer.add_scalar('Perf/Recent Reward', self.last_total_return, step)
        for key, value in stats.items():
            self.summary_writer.add_scalar(key, value, step)
        self.summary_writer.flush()

    def save_models(self):
        for name, model in self.models.items():
            torch.save(model.state_dict(), f"{self.model_path}/{name}_model.pth")
        print(f"\nSaved models to {self.model_path}")

    def load_models(self):
        for name, model in self.models.items():
            model.load_state_dict(torch.load(f"{self.model_path}/{name}_model.pth"))
        print(f"Loaded models from {self.model_path}")
