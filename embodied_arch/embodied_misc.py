import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_probability as tfp
from scipy import signal

'''
init: Feb2019
Author: OO
Requirements:
    - slim + tfp 0.6.0 <--- tf>=1.13
Mods:
    - Defaulting to deep-ish nets for actor+value
    - Actor NNs return distribution *params*, not distributions
        - Bernoulli: probabs or logits (final activation=none)
'''

sys.path.append(".")

# ML libs
# using tfp 0.5.0, compatible with tf 1.12
# latest tfp 0.6.0 requires tf>=1.13
tfd = tfp.distributions

# networks: default parameters
_reg_gamma_ = 1.e-1
_keep_prob_ = 0.8

# wgts_init = tf.truncated_normal_initializer(stddev=.25)
wgts_init = tf.glorot_uniform_initializer()
bias_init = tf.glorot_uniform_initializer()
# bias_init = tf.truncated_normal_initializer(stddev=.01)
trainingQ = True

_zdim_ = 128
_g_thresh_ = 20.0  # gradient clip threshold

# _hSeq_tri_ = (_zdim_, 16, 64, 128, 64, 16, _zdim_)
_hSeq_tri_ = (20, 20, 20, 20)


def maximize(optimizer, value, **kwargs):
    return optimizer.minimize(-value, **kwargs)


_sched_win_ = int(1e5)  # limit of cyclic LR factor


def cyc_learning_spd(sch_type, start, hlife):
    _lambda_ = 50  ## wavelength of sawtooth in cyclic alpha signal. large=>slow
    if sch_type == "constant" or sch_type not in ['exp', 'log']:
        lrps = start * np.ones(_sched_win_)
    else:
        freq = 2 * np.pi / _lambda_
        cycs = 2 * start * (1 + signal.sawtooth(freq * np.arange(1, _sched_win_), width=0))
        cycs = cycs[:_sched_win_]
        if sch_type == "exp":
            modulation = np.hstack([atten * np.ones(hlife) for
                                    atten in 2 ** -np.arange(0, np.ceil(_sched_win_ / hlife))])
        elif sch_type == "log":
            modulation = 1 / np.log(np.arange(1.002, _sched_win_) * hlife)
        modulation = modulation[:len(cycs)]
        lrps = cycs * modulation
        min_lr = 0.05 * start
        if np.min(lrps) < 0: lrps = (lrps - np.min(lrps)) + min_lr
        # lrps[0] = start
        # lrps = np.clip(lrps, a_min=(1e-6*start), a_max=(1.-_eps_))
    return lrps


### Helper fxns specific to Bernoulli/discrete binary policies
def policy_entropy(tf_logits):
    a = tf_logits - tf.reduce_max(tf_logits)
    exp_a = tf.exp(tf_logits)
    z = tf.reduce_sum(exp_a, axis=-1, keepdims=True)
    p = exp_a / z
    return tf.reduce_sum(p * (tf.log(z) - a))


# bernoulli(p) log-likelihood function
bernoulli_LL = (lambda x, p: x * tf.log(p) + (tf.ones_like(x) - x) * tf.log(tf.ones_like(p) - p))
# bernoulli_LL = (lambda x, p: x * tf.log(p) + (tf.ones_like(x) - x) * tf.log(tf.ones_like(p) - p))

# entropy functional for bernoulli(p)
bernoulli_H = (lambda p: -p * tf.log(p) - (tf.ones_like(p) - p) * tf.log(tf.ones_like(p) - p))


# bernoulli_H = (lambda p: -p * tf.log(p) - (tf.ones_like(p) - p) * tf.log(tf.ones_like(p) - p))


# def summarize(vec):  # vec = tf.stack(list(vec_d.values()), axis=1)
#     vec_ = tf.where(
#         tf.is_nan(vec),
#         tf.zeros_like(vec),
#         vec)
#     return tf.reduce_min(vec_), tf.reduce_mean(vec_), tf.reduce_max(vec_)

def summarize(vec):  # vec = tf.stack(list(vec_d.values()), axis=1)
    return tf.reduce_min(vec), tf.reduce_mean(vec), tf.reduce_max(vec)


def summarize_np(vec):
    return np.min(vec), np.mean(vec), np.max(vec)


def calc_advantages(returns, vals,
                    gamma=0.99, bootstrap_value=0.0,
                    standardize=True):
    nsteps = len(returns)
    advantages = np.zeros(nsteps, dtype=np.float32)
    vals = np.squeeze(vals)
    last_value = bootstrap_value
    for t in reversed(range(nsteps)):
        advantages[t] = returns[t] + gamma * last_value - vals[t]  # delta in S-B
        # lambda: elig.trace factor = gamma * lambda * last_advantage
        last_value = vals[t]
    if standardize:  # standardize advantages
        advantages -= np.mean(advantages)
        advantages /= np.std(advantages)
    return advantages


def calc_V_TD_target(returns, vals, gamma=0.99, bootstrap_value=0.0):
    nsteps = len(returns)
    targets = np.zeros(nsteps, dtype=np.float32)
    vals = np.squeeze(vals)
    last_value = bootstrap_value
    for t in reversed(range(nsteps)):
        targets[t] = returns[t] + gamma * last_value
        # lambda: elig.trace factor = gamma * lambda * last_advantage
        last_value = vals[t]
    return targets


def calc_Q_TD_target(returns, Qs, gamma=0.99, bootstrap_value=0.0):
    ''' assuming vector-form Q-net'''
    nsteps, ag_count = np.shape(returns)
    targets = np.zeros_like(returns, dtype=np.float32)
    next_Q = bootstrap_value * np.ones(ag_count, dtype=np.float32)  # terminal Q-value
    for t in reversed(range(nsteps)):
        targets[t] = returns[t] + gamma * next_Q
        next_Q = Qs[t]
    return targets


###
def clipGrads(gvs_, thresh=_g_thresh_):
    gs, vs = zip(*gvs_)
    gs, _ = tf.clip_by_global_norm(gs, thresh)
    return list(zip(gs, vs))


def checkRollout(buf):
    '''DEPRECATED... using list-of-dict buffers now.
    Sanity-check + normalize on SARS rollout buffer'''
    assert 4 in buf.shape, "Rollout is not the correct shape"
    if buf.shape[0] == 4:
        buf = buf.T
    return buf


def checkRolloutDict(bufDict):
    assert all(np.diff([len(buf) for _, buf in bufDict.items()]) == 0), \
        "Rollout is not the correct shape"
    rollout_len = len(list(bufDict.values())[2])
    return rollout_len


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def SensoriumNetworkTemplate(state, hSeq=None, out_dim=_zdim_,
                             gamma_reg=_reg_gamma_):
    '''Agent's Sensorium: Template/Sample Network'''
    if hSeq is None:
        hSeq = _hSeq_tri_
    regularizer = slim.l2_regularizer(gamma_reg)
    hidden = slim.fully_connected(
        slim.flatten(state),
        num_outputs=hSeq[0],
        activation_fn=tf.nn.relu,
        weights_regularizer=regularizer,
        weights_initializer=wgts_init,
        biases_initializer=bias_init
    )
    # hidden = slim.dropout(hidden, keep_prob=_keep_prob_, is_training=trainingQ)
    if (len(hSeq) > 1):
        hidden = slim.stack(hidden,
                            slim.fully_connected,
                            list(hSeq[1:]),
                            activation_fn=tf.nn.relu,
                            weights_regularizer=regularizer,
                            weights_initializer=wgts_init,
                            biases_initializer=bias_init
                            )
    latent_sensed = slim.fully_connected(
        hidden,
        num_outputs=out_dim,
        activation_fn=tf.nn.relu,
        weights_initializer=wgts_init,
        biases_initializer=bias_init
    )
    return latent_sensed


def ValueNetwork(sense, hSeq=None, gamma_reg=_reg_gamma_):
    '''Agent's Value Policy Network'''
    if hSeq is None:
        hSeq = _hSeq_tri_
    regularizer = slim.l2_regularizer(gamma_reg)
    hidden = slim.fully_connected(
        slim.flatten(sense),
        num_outputs=hSeq[0],
        activation_fn=tf.nn.relu,
        weights_regularizer=regularizer,
        weights_initializer=wgts_init,
        biases_initializer=bias_init
    )
    if (len(hSeq) > 1):
        hidden = slim.stack(hidden,
                            slim.fully_connected,
                            list(hSeq[1:]),
                            activation_fn=tf.nn.relu,
                            weights_regularizer=regularizer,
                            weights_initializer=wgts_init,
                            biases_initializer=bias_init
                            )
    val = slim.fully_connected(
        hidden, 1,
        activation_fn=None,
        weights_initializer=wgts_init,
        biases_initializer=bias_init
    )
    # this is a regression network...
    return val


def ActionPolicyNetwork(sense, hSeq=None, gamma_reg=_reg_gamma_):
    '''Agent's Action Policy Network (example)
        This one is for a binary action space'''
    if hSeq is None:
        hSeq = _hSeq_tri_
    regularizer = slim.l2_regularizer(gamma_reg)
    hidden = slim.fully_connected(
        slim.flatten(sense),
        num_outputs=hSeq[0],
        activation_fn=tf.nn.relu,
        weights_regularizer=regularizer,
        weights_initializer=wgts_init,
        biases_initializer=bias_init
    )
    if (len(hSeq) > 1):
        hidden = slim.stack(hidden,
                            slim.fully_connected,
                            list(hSeq[1:]),
                            activation_fn=tf.nn.relu,
                            weights_regularizer=regularizer,
                            weights_initializer=wgts_init,
                            biases_initializer=bias_init
                            )
    apn = slim.fully_connected(
        hidden, 1,
        activation_fn=None,  # tf.nn.sigmoid,
        weights_initializer=wgts_init,
        biases_initializer=bias_init
    )  # activation_fn=None for alogits
    return apn


def ActionPolicyNetworkBox(sense, hSeq=_hSeq_tri_, gamma_reg=_reg_gamma_):
    '''Agent's Action Policy Network (example)
        This one is for a continuous box action space (e.g. MountainCarContinuous)'''
    regularizer = slim.l2_regularizer(gamma_reg)
    hidden = slim.fully_connected(
        slim.flatten(sense),
        num_outputs=hSeq[0],
        activation_fn=tf.nn.relu,
        weights_regularizer=regularizer,
        weights_initializer=wgts_init,
        biases_initializer=bias_init
    )
    if (len(hSeq) > 1):
        hidden = slim.stack(hidden,
                            slim.fully_connected,
                            list(hSeq[1:]),
                            activation_fn=tf.nn.relu,
                            weights_regularizer=regularizer,
                            weights_initializer=wgts_init,
                            biases_initializer=bias_init
                            )

    # Reparametrize Bernoulli with logits instead of probabs; might address nan issues
    a_theta = slim.fully_connected(
        hidden, 1,
        activation_fn=None,
        weights_initializer=wgts_init,
        biases_initializer=bias_init
    )  # activation_fn=None
    return a_theta
