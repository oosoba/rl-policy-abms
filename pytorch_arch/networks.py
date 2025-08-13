import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class SensoriumNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim=128, h_dims=(20, 20, 20, 20)):
        super(SensoriumNetwork, self).__init__()
        self.latent_dim = latent_dim

        layers = []
        in_dim = input_dim
        for h_dim in h_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, self.latent_dim)

        self.apply(init_weights)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = F.relu(self.output_layer(x))
        return x

class ActionPolicyNetwork(nn.Module):
    def __init__(self, latent_dim, action_dim, h_dims=(20, 20, 20, 20)):
        super(ActionPolicyNetwork, self).__init__()

        layers = []
        in_dim = latent_dim
        for h_dim in h_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, action_dim)

        self.apply(init_weights)

    def forward(self, x):
        x = self.hidden_layers(x)
        # The output is logits for a binary action space, so no activation here.
        # Sigmoid will be applied outside to get probabilities.
        return self.output_layer(x)

class ValueNetwork(nn.Module):
    def __init__(self, latent_dim, h_dims=(20, 20, 20, 20)):
        super(ValueNetwork, self).__init__()

        layers = []
        in_dim = latent_dim
        for h_dim in h_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)

        self.apply(init_weights)

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)
