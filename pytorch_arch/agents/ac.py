import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Bernoulli

from pytorch_arch.agent_base import PytorchAgent
from pytorch_arch.agents.rf import discount # Re-use the discount function from rf.py

class PytorchAgentAC(PytorchAgent):
    def __init__(self, name, env, lrp=1e-3, lrv=1e-2, reg_gamma=1e-1, ent_decay=5e-3, **kwargs):
        super(PytorchAgentAC, self).__init__(name, env, **kwargs)

        self.ent_decay = ent_decay

        # Separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(
            list(self.sensorium.parameters()) + list(self.actor.parameters()),
            lr=lrp,
            weight_decay=reg_gamma
        )
        self.critic_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=lrv,
            weight_decay=reg_gamma
        )
        self.models['actor_optimizer'] = self.actor_optimizer
        self.models['critic_optimizer'] = self.critic_optimizer

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

        # Calculate discounted returns (targets for the critic)
        returns = torch.from_numpy(discount(rewards, gamma)).float().unsqueeze(1)

        # Forward pass
        z = self.sensorium(states)
        values = self.value_net(z)
        logits = self.actor(z)

        # Critic Loss
        critic_loss = torch.nn.functional.mse_loss(values, returns)

        # Critic optimization
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Loss
        advantage = (returns - values).detach() # Detach to stop gradients from flowing to the critic

        log_probs = -torch.nn.functional.binary_cross_entropy_with_logits(
            logits, actions, reduction='none'
        )

        probs = torch.sigmoid(logits)
        entropy = Bernoulli(probs).entropy().mean()

        actor_loss = -torch.mean(advantage * log_probs) - self.ent_decay * entropy

        # Actor optimization
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            'Losses/Actor_Loss': actor_loss.item(),
            'Losses/Critic_Loss': critic_loss.item(),
            'Losses/Entropy': entropy.item()
        }
