import itertools, importlib, sys, warnings

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ML libs
import tensorflow as tf
print("Tensorflow version:", tf.__version__)
warnings.filterwarnings("ignore")

sys.path.append('./embodied_arch')
sys.path.append('./minoritygame')
import minoritygame.minority_multienv as MGME
import embodied as emg
import embodied_indep as emi


importlib.reload(MGME)
importlib.reload(emg)
importlib.reload(emi)



plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (18,7)

log_path = './log/mingame'
#tensorboard --logdir=mingame_worker_1:'./log/train_rf_mingame_worker'
tf.reset_default_graph()
importlib.reload(MGME)
importlib.reload(emi)
#exos = (np.random.sample(33) < 0.3)  # np.sum(exos)
exos = np.zeros(301)
exos[-1] = 1
#exos[-2] = 1
exos = np.bool_(exos)
menv = MGME.MinorityGame_Multiagent_env(nagents=301, m=2, s=2, mrl=3,
                                                exo_actorsQ=exos
                                            )


print(menv.actor_count, menv.actor_index, menv.actor_exoQ)
print(menv.state_space_size, menv.action_space_size)

embrf = emi.EmbodiedAgent_IRF(
        name="mgRF",
        env_=menv,
        alpha=5.e-2
    )

#Needs to match parameters of trained neural net.  Really need to expose those params.  

num_episodes = 15
n_epochs = 400

embrf.max_episode_length = 500 #101  # dangerous... may incentivize finite n behavior
print(embrf, embrf.s_size, embrf.a_size)

sess = tf.InteractiveSession()
embrf.init_graph(sess)

def get_sensorium_dict():
    d = {}
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        if "sensorium" in var.name:
            key  = var.name.split(":")[0]
            d[key]=var
    return d

def set_sensorium(sess, path):
    tf.train.Saver(get_sensorium_dict()).restore(sess, path)

def set_actor_net(actor_name, sess, path):
    d = {}
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        if actor_name in var.name:
            if "actor" in var.name:
                newkey = var.name.replace("actor", "action")
            if "critic" in var.name:
                newkey = var.name.replace("critic", "value_fxn")
            newkey = newkey.replace(actor_name +"/", "")
            newkey = newkey.split(":")[0]
            d[newkey] = var
            
    saver = tf.train.Saver(d)
    saver.restore(sess, path)
    return  saver

set_sensorium(sess, "./log/train_mgRF/model-261.cptk")
set_actor_net("player_0", sess, "./log/train_mgRF/model-261.cptk")

state = embrf.env.reset()
print(state, embrf.act(state, sess))

embrf.env.step(embrf.act(state, sess))
