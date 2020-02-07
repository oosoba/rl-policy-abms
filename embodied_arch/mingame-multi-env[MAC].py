#!/usr/bin/env python
import importlib
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# ML libs
import tensorflow as tf  # from tensorflow.python import debug as tf_debug

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '3' to block all including error msgs
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

sys.path.append('./embodied_arch')
import embodied_arch.embodied_central_Qcritic as emac

importlib.reload(emac)

sys.path.append('./embodied_arch')
sys.path.append('./minoritygame')
import minoritygame.minority_multienv as MGME
from embodied_misc import ActionPolicyNetwork, ValueNetwork, SensoriumNetworkTemplate

importlib.reload(MGME)
log_path = './log/mingame'
# tensorboard --logdir=mingame_worker_1:'./log/train_rf_mingame_worker'


n_agents = 61
exos = (np.random.sample(n_agents) < 0.3)  # np.sum(exos)
exos = [True, True, False, False]
menv = MGME.MinorityGame_Multiagent_env(
    m=3, s=4,
    exo_actorsQ=exos
)
print(len(menv.actor_exoQ), menv.actor_count, menv.actor_index)
print(menv.state_space_size, menv.action_space_size)
print(len(menv.actor_exoQ), (len(menv.agents), menv.nagents))

# ## Setup MARL
num_episodes, n_epochs, max_len = (25, 501, 30)
actor = lambda s: ActionPolicyNetwork(s, hSeq=(8,), gamma_reg=2.)
value = lambda s: ValueNetwork(s, hSeq=(8,), gamma_reg=1.)
sensor = lambda st, out_dim: SensoriumNetworkTemplate(st, hSeq=(16, 8,), out_dim=out_dim, gamma_reg=5.)

tf.reset_default_graph()
importlib.reload(emac)
empopn = emac.EmbodiedAgent_MAC(
    name="mgMAC_k-vs-N-k", env_=menv,
    alpha_p=1., alpha_v=1e-2, alpha_q=1e-1,
    actorNN=actor, valueNN=value,
    sensorium=sensor,
    max_episode_length=max_len
)
# empopn = emac.EmbodiedAgent_MAC( 
#     name=agent_name,
#     env_=MinorityGame_Multiagent_env,
#     latentDim=_zdim_,
#     space_size=(_s_size_, _a_size_),
#     sensorium=SensoriumNetworkTemplate,
#     actorNN=ActionPolicyNetwork,
#     valueNN=ValueNetwork,
#     QNN=QsaNetwork,
#     alpha_p=5e-2, alpha_v=1e-1, alpha_q=1e-1,
#     _every_=_every_, recover=None,
#     max_episode_length=_max_len_
# )
# print(empopn, empopn.s_size, empopn.a_size)

sess = tf.InteractiveSession()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
empopn.init_graph(sess)  # note tboard log dir

## Verify step + play set up
state = empopn.env.reset()
print(state, empopn.act(state, sess))
empopn.play(sess)
print(np.sum(np.array(empopn.episode_buffer['rewards']), axis=0))
# np.array(empopn.episode_buffer['rewards'])
print(empopn.last_total_returns)
print(emac.summarize_np(empopn.last_total_returns))

## Test Q_global trainer
empopn.play(sess)
empopn.episode_buffer['rewards']
empopn.last_total_returns
tmp = empopn.train_eval_QC(sess)
print(tmp)

empopn.play(sess)
empopn.train_eval_QC(sess)

# ### Pre-test Agent
print('Baselining untrained pnet...')
rwd_mg0 = []
for k in range(num_episodes):
    empopn.play(sess, terminal_reward=0.)
    rwd_mg0.append(empopn.last_total_returns)
    if k % int(num_episodes / 5) == 0: print("\rEpisode {}/{}".format(k, num_episodes), end="")
base_perf_mg = np.mean(np.array(rwd_mg0) / float(empopn.max_episode_length))
print("\nAgent wins an average of {} pct".format(100.0 * base_perf_mg))
base_per_agent = np.mean(np.array(rwd_mg0) / float(empopn.max_episode_length), axis=0)

# ## Train MARL Agents
## Pre Train QC
obs = []
for ct in range(1250):
    empopn.play(sess)
    tmp = empopn.train_eval_QC(sess)
    obs.append(np.mean(tmp, axis=0))
    print('\r\tIteration {}: Value loss({})'.format(ct, np.mean(tmp, axis=0)), end="")
plt.plot(obs)

# ### Train Agent w/ Algo on Experience Tuples
# Train pnet on mingame episodes
print('Training...')
saver = tf.train.Saver(max_to_keep=1)
empopn.work(sess, num_epochs=n_epochs, saver=saver)

# ### Post-test Agent
# Test pnet!
print('Testing...')
rwd_mg = []
for k in range(num_episodes):
    empopn.play(sess, terminal_reward=0.)
    rwd_mg.append(empopn.last_total_returns)
    if k % int(num_episodes / 5) == 0: print("\rEpisode {}/{}".format(k, num_episodes), end="")
trained_perf_mg = np.mean(np.array(rwd_mg) / float(empopn.max_episode_length))
print("\nAgent wins an average of {} pct compared to baseline of {} pct".format(
    100 * trained_perf_mg, 100 * base_perf_mg))

trained_per_agent = np.mean(np.array(rwd_mg) / float(empopn.max_episode_length), axis=0)

fig, axs = plt.subplots(2, 1, sharex=True)
sns.violinplot(np.array(rwd_mg0) / empopn.max_episode_length - 0.5, ax=axs[0])
axs[0].set_title('Baseline Mean Success Rates')
sns.violinplot(np.array(rwd_mg) / empopn.max_episode_length - 0.5, ax=axs[1])
axs[1].set_title('Trained Mean Success Rate')

print("\nAgent wins an average of {} pct \ncompared to baseline of {} pct".format(
    100 * trained_perf_mg, 100 * base_perf_mg))


fig, axs = plt.subplots(2, 1)
cmp = sns.boxplot(base_per_agent - trained_per_agent, ax=axs[0])
axs[0].set_title('Distribution: Success Rate Per-Agent Improvement');
axs[1].bar(x=range(sum(exos)), height=(base_per_agent - trained_per_agent))