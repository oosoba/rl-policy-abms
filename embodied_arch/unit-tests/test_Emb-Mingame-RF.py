# coding: utf-8
import importlib
import sys

import numpy as np
import tensorflow as tf

import embodied_arch.embodied as emg
import minoritygame.minority_agent as minority_agent
from minoritygame.minority_env import MinorityGame1vN_env

sys.path.append('./..')

importlib.reload(minority_agent)
importlib.reload(emg)

num_episodes = 75
n_epochs = 501  # 101  # 5001

importlib.reload(minority_agent)
importlib.reload(emg)

tf.reset_default_graph()
menv = MinorityGame1vN_env(nagents=301, m=2, s=2, mrl=4, p=0.5)
embrf = emg.EmbodiedAgentRF(
    name="mgRF",
    env_=menv,
    alpha=1.e-1
)

# embrf = emg.EmbodiedAgentRFBaselined(
#     name="mgRFB",
#     env_=menv,
#     alpha_p=1.e-1
# )
print(menv.state_space_size, menv.action_space_size)

embrf.max_episode_length = 50  # 101  # dangerous... may incentivize finite n behavior
print(embrf, embrf.s_size, embrf.a_size)

sess = tf.InteractiveSession()
embrf.init_graph(sess)  # note tboard log dir

# Verify step + play set up
state = embrf.env.reset()
print(state, embrf.act(state, sess))

embrf.env.step(embrf.act(state, sess))
embrf.play(sess)
print(embrf.last_total_return)

# ### Pre-test Agent
print('Baselining untrained pnet...', flush=True)
rwd_mg0 = []
for k in range(num_episodes):
    embrf.play(sess)
    rwd_mg0.append(float(embrf.last_total_return) / embrf.max_episode_length)
    if k % int(num_episodes / 5) == 0:
        print("\rEpisode {}/{}".format(k, num_episodes), end="")
base_perf_mg = np.mean(rwd_mg0)
print("\nAgent wins an average of {} pct".format(100.0 * base_perf_mg),
      flush=True)

# ### Train Agent w/ Algo on Experience Tuples
# Train pnet on mingame episodes
print('Training...')
saver = tf.train.Saver(max_to_keep=1)
embrf.work(sess, saver, num_epochs=n_epochs)

# ### Post-test Agent# Test pnet!
print('Testing...', flush=True)
rwd_mg = []
for k in range(num_episodes):
    embrf.play(sess)
    rwd_mg.append(float(embrf.last_total_return) / embrf.max_episode_length)
    if k % int(num_episodes / 5) == 0:
        print("\rEpisode {}/{}".format(k, num_episodes), end="")
trained_perf_mg = np.mean(rwd_mg)
print("\nAgent wins an average of {} pct \ncompared to baseline of {} pct".format(
    100 * trained_perf_mg, 100 * base_perf_mg), flush=True)

# fig, axs = plt.subplots(2, 1, sharex=True)
# sns.boxplot(rwd_mg0, ax = axs[0])
# axs[0].set_title('Baseline Mean Success Percentage')
# sns.boxplot(rwd_mg, ax = axs[1])
# axs[1].set_title('Trained Mean Success Percentage')

sess.close()
