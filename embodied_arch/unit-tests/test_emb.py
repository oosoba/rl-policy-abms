# coding: utf-8
import importlib
import sys

import tensorflow as tf

import embodied_arch.embodied as emg
import minoritygame.minority_agent as minority_agent
from minoritygame.minority_env import MinorityGame1vN_env

sys.path.append('./..')
importlib.reload(minority_agent)
importlib.reload(emg)

n_epochs = 11
tf.reset_default_graph()

menv = MinorityGame1vN_env(nagents=301, m=2, s=2, mrl=3, p=0.5)
embrf = emg.EmbodiedAgentRF(
    name="embMRF-test",
    env_=menv
)
embrf.max_episode_length = 25

# env = gym.make('CartPole-v0')
# embrf = emg.EmbodiedAgentRFBaselined(name="cp-emb", env_=env, space_size = (4,1))
# alpha_p=5.e-2, alpha_v=1.e-1

sess = tf.InteractiveSession()
embrf.init_graph(sess)

## Verify step + play set up
state = embrf.env.reset()
embrf.env.step(embrf.act(state, sess))
embrf.play(sess)

# Train pnet on mingame episodes
print('Training...')
saver = tf.train.Saver(max_to_keep=1)
embrf.work(sess, saver, num_epochs=n_epochs)
sess.close()
