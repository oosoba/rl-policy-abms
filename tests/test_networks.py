import torch
import pytest

from pytorch_arch.networks import SensoriumNetwork, ActionPolicyNetwork, ValueNetwork

def test_sensorium_network():
    input_dim = 10
    latent_dim = 128
    net = SensoriumNetwork(input_dim, latent_dim)

    # Test with a single input
    input_tensor = torch.randn(1, input_dim)
    output = net(input_tensor)
    assert output.shape == (1, latent_dim)

    # Test with a batch of inputs
    batch_size = 16
    input_tensor = torch.randn(batch_size, input_dim)
    output = net(input_tensor)
    assert output.shape == (batch_size, latent_dim)

def test_action_policy_network():
    latent_dim = 128
    action_dim = 1
    net = ActionPolicyNetwork(latent_dim, action_dim)

    # Test with a single input
    input_tensor = torch.randn(1, latent_dim)
    output = net(input_tensor)
    assert output.shape == (1, action_dim)

    # Test with a batch of inputs
    batch_size = 16
    input_tensor = torch.randn(batch_size, latent_dim)
    output = net(input_tensor)
    assert output.shape == (batch_size, action_dim)

def test_value_network():
    latent_dim = 128
    net = ValueNetwork(latent_dim)

    # Test with a single input
    input_tensor = torch.randn(1, latent_dim)
    output = net(input_tensor)
    assert output.shape == (1, 1)

    # Test with a batch of inputs
    batch_size = 16
    input_tensor = torch.randn(batch_size, latent_dim)
    output = net(input_tensor)
    assert output.shape == (batch_size, 1)
