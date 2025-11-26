import pytest
import gymnasium as gym

from pytorch_arch.agents.rf import PytorchAgentRF
from pytorch_arch.agents.ac import PytorchAgentAC
from pytorch_arch.agents.ppo import PytorchAgentPPO

@pytest.fixture
def cartpole_env():
    return gym.make('CartPole-v0')

def test_rf_agent_instantiation(cartpole_env):
    agent = PytorchAgentRF(name="test_rf", env=cartpole_env)
    assert agent is not None

def test_ac_agent_instantiation(cartpole_env):
    agent = PytorchAgentAC(name="test_ac", env=cartpole_env)
    assert agent is not None

def test_ppo_agent_instantiation(cartpole_env):
    config = {'ppo_epochs': 2}
    agent = PytorchAgentPPO(name="test_ppo", env=cartpole_env, config=config)
    assert agent is not None
    assert agent.ppo_epochs == 2

def test_ppo_agent_train_step(cartpole_env):
    config = {'ppo_epochs': 2}
    agent = PytorchAgentPPO(name="test_ppo_train", env=cartpole_env, config=config, max_episode_length=10)

    # Run a short episode
    agent.play()

    # Check if the buffer has been filled
    assert agent.episode_length() > 0

    # Run a single training step
    stats = agent.train(gamma=0.99)

    # Check if the stats are reported
    assert 'Losses/Actor_Loss' in stats
    assert 'Losses/Critic_Loss' in stats
    assert 'Losses/Total_Loss' in stats
    assert 'Losses/Entropy' in stats

    # Check that the loss is a float
    assert isinstance(stats['Losses/Total_Loss'], float)
