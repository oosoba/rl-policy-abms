import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Bernoulli

from pytorch_arch.agent_base import PytorchAgent

# From original code
def discount(rewards, gamma=0.99, window=None):
    """
    computes the discounted reward sum (returns) from a list of rewards
    """
    if window is None:
        return _discount_rewards(rewards, gamma)
    else:
        rwds = np.array(rewards)
        n = len(rwds)
        dr = np.zeros_like(rwds)
        for i in range(n):
            i0 = max(0, i - window + 1)
            dr[i] = _discount_rewards(rwds[i0:i + 1], gamma)[0]
        return dr

def _discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class PytorchAgentRF(PytorchAgent):
    def __init__(self, name, env, lrp=1e-2, reg_gamma=1e-1, **kwargs):
        super(PytorchAgentRF, self).__init__(name, env, **kwargs)

        self.optimizer = optim.Adam(
            list(self.sensorium.parameters()) + list(self.actor.parameters()),
            lr=lrp,
            weight_decay=reg_gamma
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
        rewards = self.episode_buffer['rewards']

        # Calculate discounted returns
        discounted_returns = torch.from_numpy(discount(rewards, gamma)).float().unsqueeze(1)

        # Forward pass
        z = self.sensorium(states)
        logits = self.actor(z)

        # Calculate policy loss (REINFORCE loss)
        # Using BCEWithLogitsLoss which combines a Sigmoid layer and the BCELoss in one single class.
        # This is equivalent to -(G_t * log(pi(a_t|s_t)))
        log_probs = -torch.nn.functional.binary_cross_entropy_with_logits(
            logits, actions, reduction='none'
        )
        policy_loss = -torch.mean(discounted_returns * log_probs)

        # Calculate entropy for reporting
        probs = torch.sigmoid(logits)
        entropy = Bernoulli(probs).entropy().mean()

        # Backward pass and optimization
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return {
            'Losses/Policy_Loss': policy_loss.item(),
            'Losses/Entropy': entropy.item()
        }
