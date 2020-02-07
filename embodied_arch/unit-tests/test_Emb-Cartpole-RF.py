# coding: utf-8
import importlib
import sys

import gym
import numpy as np
import tensorflow as tf

import embodied_arch.embodied as emg

sys.path.append('./..')
importlib.reload(emg)

# ## Cartpole Benchmark Setup
tf.reset_default_graph()
env = gym.make('CartPole-v0')
cprf = emg.EmbodiedAgentRF(
    name="cartpole-emb",
    env_=env,
    space_size=(4, 1)
)
print(cprf, cprf.s_size, cprf.a_size)

saver = tf.train.Saver(max_to_keep=1)  # n_epochs = 1000
sess = tf.InteractiveSession()
cprf.init_graph(sess)

num_episodes = 40
n_epochs = 101

# Verify step + play set up
state = cprf.env.reset()
print(state, cprf.act(state, sess))
cprf.env.step(cprf.act(state, sess))
cprf.play(sess)
len(cprf.episode_buffer)

# ## Baseline
print('\n\nBaselining untrained pnet...')
uplen0 = []
for k in range(num_episodes):
    cprf.play(sess)
    uplen0.append(cprf.last_total_return)  # uplen0.append(len(cprf.episode_buffer))
    if k % 20 == 0:
        print("\rEpisode {}/{}".format(k, num_episodes), end="")
base_perf = np.mean(uplen0)
print("\nCartpole stays up for an average of {} steps".format(base_perf))

# ## Train
# Train pnet on cartpole episodes
print('Training...')
cprf.work(sess, saver, num_epochs=n_epochs)

# ## Test: Test pnet!
print('\n\nTesting...')
uplen = []
for k in range(num_episodes):
    cprf.play(sess)
    uplen.append(cprf.last_total_return)  # uplen.append(len(cprf.episode_buffer))
    if k % 20 == 0:
        print("\rEpisode {}/{}".format(k, num_episodes), end="")
trained_perf = np.mean(uplen)

# Evaluate
print("\nCartpole stays up for an average of {} steps compared to baseline {} steps".format(
    trained_perf, base_perf))

# fig, axs = plt.subplots(2, 1, sharex=True)
# sns.boxplot(uplen0, ax=axs[0])
# axs[0].set_title('Baseline Episode Lengths')
# sns.boxplot(uplen, ax=axs[1])
# axs[1].set_title('Trained Episode Lengths')


sess.close()
