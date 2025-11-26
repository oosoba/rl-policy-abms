import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Bernoulli

from pytorch_arch.agent_base import PytorchAgent

# This function is different from the one in rf.py, as it uses TD(lambda)-style returns
# which is what the original PPO and AC implementations used.
def calc_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    Calculate Generalized Advantage Estimation (GAE).
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = 0
    for t in reversed(range(len(rewards) - 1)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        advantages[t] = last_adv = delta + gamma * lam * last_adv
    returns = advantages + values[:-1] # In the buffer, states and next_states have same length
    return returns, advantages


class PytorchAgentPPO(PytorchAgent):
    def __init__(self, name, env, config, **kwargs):
        super(PytorchAgentPPO, self).__init__(name, env, **kwargs)

        self.ent_decay = config.get('ent_decay', 5e-3)
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.clip_range = config.get('clip_range', 0.2)
        lrp = config.get('lrp', 1e-3)

        self.optimizer = optim.Adam(
            list(self.sensorium.parameters()) + list(self.actor.parameters()) + list(self.value_net.parameters()),
            lr=lrp
        )
        self.models['optimizer'] = self.optimizer

    def act(self, state_tensor):
        self.sensorium.eval()
        self.actor.eval()
        with torch.no_grad():
            z = self.sensorium(state_tensor)
            logits = self.actor(z)
            probs = torch.sigmoid(logits)
            action = Bernoulli(probs).sample().item()
        self.sensorium.train()
        self.actor.train()
        return int(action)

    def train(self, gamma):
        states = torch.from_numpy(np.array(self.episode_buffer['states'])).float()
        actions = torch.from_numpy(np.array(self.episode_buffer['actions'])).float().unsqueeze(1)
        rewards = np.array(self.episode_buffer['rewards'])

        # Calculate old log probs with the policy that generated the data
        with torch.no_grad():
            z_old = self.sensorium(states)
            logits_old = self.actor(z_old)
            old_log_probs = -torch.nn.functional.binary_cross_entropy_with_logits(
                logits_old, actions, reduction='none'
            )
            values_old = self.value_net(z_old).squeeze().numpy()

        # Append the value of the last state for GAE calculation
        with torch.no_grad():
            last_state = torch.from_numpy(self.episode_buffer['next_states'][-1]).float().unsqueeze(0)
            last_value = self.value_net(self.sensorium(last_state)).item()

        values_for_gae = np.append(values_old, last_value)

        returns, advantages = calc_gae(rewards, values_for_gae, gamma)
        returns = torch.from_numpy(returns).float().unsqueeze(1)
        advantages = torch.from_numpy(advantages).float().unsqueeze(1)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        # PPO inner loop
        for _ in range(self.ppo_epochs):
            z = self.sensorium(states)
            logits = self.actor(z)
            values = self.value_net(z)

            # Critic loss
            critic_loss = torch.nn.functional.mse_loss(values, returns)

            # Actor loss
            new_log_probs = -torch.nn.functional.binary_cross_entropy_with_logits(
                logits, actions, reduction='none'
            )

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages

            probs = torch.sigmoid(logits)
            entropy = Bernoulli(probs).entropy().mean()

            actor_loss = -torch.min(surr1, surr2).mean() - self.ent_decay * entropy

            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss

            # Optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return {
            'Losses/Actor_Loss': actor_loss.item(),
            'Losses/Critic_Loss': critic_loss.item(),
            'Losses/Total_Loss': total_loss.item(),
            'Losses/Entropy': entropy.item()
        }
