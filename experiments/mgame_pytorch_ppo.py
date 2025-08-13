import sys
import os

# Add the root directory to the Python path
# This is necessary to import modules from the other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pytorch_arch.agents.ppo import PytorchAgentPPO
from minoritygame.minority_env import MinorityGame1vN_env

def main():
    print("Setting up the Minority Game environment...")
    env = MinorityGame1vN_env(nagents=101, m=5, s=2)

    config = {
        'lrp': 1e-3,
        'ent_decay': 1e-3,
        'ppo_epochs': 5,
        'clip_range': 0.2,
    }

    print("Creating the PyTorch PPO agent...")
    agent = PytorchAgentPPO(
        name="ppo_mgame_test",
        env=env,
        config=config,
        max_episode_length=200,
        latent_dim=32 # smaller latent dim for this problem
    )

    num_epochs = 200
    print(f"Starting training for {num_epochs} epochs...")
    agent.work(num_epochs=num_epochs, gamma=0.95)

    print("\nExperiment finished successfully!")
    print(f"Final average return: {agent.last_total_return}")

if __name__ == '__main__':
    main()
