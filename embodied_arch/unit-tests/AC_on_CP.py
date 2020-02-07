# coding: utf-8
import importlib

# sys.path.append('.')
import embodied as emb
import embodied_AC as emg
import gym
import numpy as np
import tensorflow as tf

importlib.reload(emg)

num_episodes = 100
n_epochs = 10001
emb._every_ = int(n_epochs / 25)

# ## Cartpole Benchmark Setup
tf.reset_default_graph()
env = gym.make('CartPole-v0')
cprf = emg.EmbodiedAgentAC(name="ac-cp", env_=env,
                           alpha_p=5.e-3, latentDim=24)
print(cprf, cprf.s_size, cprf.a_size)
saver = tf.train.Saver(max_to_keep=1)
sess = tf.InteractiveSession()
cprf.init_graph(sess)

## Verify step + play set up
state = cprf.env.reset()
print(state, cprf.act(state, sess))
cprf.env.step(cprf.act(state, sess))
cprf.play(sess)
len(cprf.episode_buffer)

# # ## Baseline
# print('Baselining untrained pnet...', flush=True)
# uplen0 = []
# for k in range(num_episodes):
#     cprf.play(sess)
#     uplen0.append(cprf.last_total_return) # uplen0.append(len(cprf.episode_buffer))
#     if k%5 == 0: print("\rEpisode {}/{}".format(k, num_episodes),end="")
# base_perf = np.mean(uplen0)
# print("\nCartpole stays up for an average of {} steps".format(base_perf))

base_perf = 27.
print("\nCartpole stays up for an average of 25-30 steps")

# ## Train
# Train pnet on cartpole episodes
print('Training...', flush=True)
saver = tf.train.Saver(max_to_keep=1)
cprf.work(sess, saver, num_epochs=n_epochs)

# ## Test
# Test pnet!
print('Testing...', flush=True)
uplen = []
for k in range(num_episodes):
    cprf.play(sess)
    uplen.append(cprf.last_total_return)
    if k % 5 == 0: print("\rEpisode {}/{}".format(k, num_episodes), end="")
trained_perf = np.mean(uplen)
print("\nCartpole stays up for an average of "
      "{} steps compared to baseline {} steps".format(
    trained_perf, base_perf),
    flush=True)

# ## Evaluate
# fig, axs = plt.subplots(2, 1, sharex=True)
# sns.boxplot(uplen0, ax = axs[0])
# axs[0].set_title('Baseline Episode Lengths')
# sns.boxplot(uplen, ax = axs[1])
# axs[1].set_title('Trained Episode Lengths')

sess.close()
