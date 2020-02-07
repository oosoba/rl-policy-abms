import importlib
import sys

import tensorflow as tf

sys.path.append('./../../minoritygame')

from minoritygame.minority_env import MinorityGame1vN_env
import embodied_arch.embodied as emg

importlib.reload(emg)

n_epochs = 51
tf.reset_default_graph()
menv = MinorityGame1vN_env(33, 4, 4, 0.5)
embrf = emg.EmbodiedAgentRF(name="mingame-RF-test", env_=menv)
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
