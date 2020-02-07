import embodied_arch.embodied_indep as emi
import minoritygame.minority_multienv as MGME
import importlib
import sys
import warnings

import numpy as np
import tensorflow as tf

print("Tensorflow version:", tf.__version__)
warnings.filterwarnings("ignore")

sys.path.append('./embodied_arch')
sys.path.append('./minoritygame')

tf.reset_default_graph()
importlib.reload(MGME)
importlib.reload(emi)
exos = (np.random.sample(33) < 0.3)  # np.sum(exos)
menv = MGME.MinorityGame_Multiagent_env(
    nagents=33, m=3, s=4,
    exo_actorsQ=exos
)

embrf = emi.EmbodiedAgent_IRF(
    name="mgRF",
    env_=menv,
    alpha=5.e-2
)

num_episodes, n_epochs = (15, 101)
embrf.max_episode_length = 30  # 101
print(embrf, embrf.s_size, embrf.a_size)

sess = tf.InteractiveSession()
embrf.init_graph(sess)  # note tboard log dir

saver = tf.train.Saver(max_to_keep=1)
embrf.work(sess, num_epochs=n_epochs, saver=saver)
