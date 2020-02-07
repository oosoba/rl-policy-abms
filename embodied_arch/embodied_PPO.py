'''
init: Sept-2019
Author: OO
Topic: Proximal Policy Optimization Implementation for Binary action-space env.
Issues: PPO is mostly only useful when there is a distributed worker architecture
with a central node issuing π_old evaluations to compare with π_t for the same {(a_t|s_t)}_t.
Goals: All restricted to Policy Gradient Methods...
    - `EmbodiedAgent` implements: (as base class)
        - tripartite algo split
        - embodiment concept
        - play() construct for generating experience tuples
    - `EmbodiedAgent_<Algo>` implements:
        - <Algo>'s training procedures
        - overrides work() fxn in base class
To Do:
    - Implement online training version
'''
import numpy as np
import tensorflow as tf
# from minoritygame.minority_env import MinorityGame1vN_env
from embodied import *
from embodied_misc import _zdim_, _sched_win_, cyc_learning_spd, policy_entropy
from embodied_misc import calc_V_TD_target, policy_entropy, bernoulli_H, bernoulli_LL

# from embodied_arch.embodied import *

tboard_path = "./log"
agent_name = "embodied_PPO"
__version__ = "0.0.1"
_DEBUG_ = False

# model: var defaults
_default_reports_ = ['Perf/Recent Reward',
                     'Losses/Policy LL', 'Losses/Entropy',
                     'Norms/Grad Norm', 'Norms/Var Norm']

_every_ = 100
_eps_ = 1.e-4
_clip_default_ = 0.2

_g_thresh_ = 20.0  # gradient clip threshold
_ent_decay_ = 5e-3
_zdim_ = 16
(lrp, lrv) = (1e-3, 1e-2)  # learning rates

# envs: var defaults
(na_, m_, s_, p_) = (33, 3, 4, 0.5)
_max_len_ = 200
_s_size_, _a_size_ = (4, 1)



class EmbodiedAgentPPO(EmbodiedAgent):
    def __init__(self, name=agent_name,
                 env_=gym.make('CartPole-v0'),
                 alpha_p=5e-2, alpha_v=1e-1, clipping=_clip_default_,
                 latentDim=_zdim_,
                 space_size=(_s_size_, _a_size_),
                 sensorium=SensoriumNetworkTemplate,
                 actorNN=ActionPolicyNetwork,
                 valueNN=ValueNetwork,
                 recover=None, _every_=_every_,
                 max_episode_length=_max_len_
                 ):
        super().__init__(name=name, env_=env_,
                         latentDim=latentDim, space_size=space_size,
                         sensorium=sensorium, actorNN=actorNN, valueNN=valueNN,
                         _every_=_every_, max_episode_length=max_episode_length
                         )

        self.optimizer_p = tf.train.AdamOptimizer(learning_rate=lrp)
        self.optimizer_v = tf.train.AdamOptimizer(learning_rate=lrv)
        self.clip_range = clipping

        self.report_labels = ['Perf/Recent Reward',
                              'Losses/Policy LL', 'Losses/Value Fxn', 'Losses/Entropy']
        self.last_good_model = recover

        with tf.variable_scope(self.name):
            # AC/PPO-specific placeholders: ln π, delta-Advantages, ratio
            self.old_lnPi_t = tf.placeholder(dtype=tf.float32, shape=[None, 1],
                                             name="old_ln_pi")

            # Intermediate variables
            # Evaluate ln π(a_t|s_t) for bernoulli action space output
            # Taking advantage of built-in cross_entropy fxn (w/ numeric guards). NB: need -1*...
            self.lnPi_t = - tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.a_logit,  ## a_t|s_t
                labels=self.actions_At,  ## use tf.one_hot for m-ary action spaces
                name="ln_pi"
            )
            self.policy_LL = tf.reduce_mean(self.lnPi_t)  # Report vars
            self.ratio = tf.exp(
                self.lnPi_t - self.old_lnPi_t,
                name="pi_ratio")  # holds: ratio bw new/old π
            self.Advs_t = self.returns_Gt - self.value

            self.clipped_ratio = tf.clip_by_value(
                self.ratio,
                1.0 - self.clip_range, 1.0 + self.clip_range,
                name="clipped_ratio"
            )
            self.policy_obj = tf.minimum(
                self.ratio * tf.stop_gradient(self.Advs_t),
                self.clipped_ratio * tf.stop_gradient(self.Advs_t),
                name="policy_objective"
            )  # replaces standard ∂_t * ln π_t PG-objective
            self.entropy = tf.clip_by_value(
                tf.reduce_mean(bernoulli_H(self.a_prob)),
                _eps_, 100.
            )
            # Losses
            self.ploss = tf.reduce_mean(self.policy_obj)
            self.vloss = 0.5 * tf.reduce_mean(
                tf.square(self.returns_Gt - self.value)
            )

            # Separate out: Training Vars
            self.policy_vars = [v for v in tf.trainable_variables()
                                if "action" in v.name or "sensorium" in v.name]
            self.value_vars = [v for v in tf.trainable_variables()
                               if "value_fxn" in v.name or "sensorium" in v.name]

            # Training Ops for p+v nets
            self.p_grads = self.optimizer_p.compute_gradients(
                loss=(-alpha_p) * (self.ploss + _ent_decay_ * self.entropy),
                var_list=self.policy_vars
            )
            self.p_train = self.optimizer_p.apply_gradients(self.p_grads)
            # self.optimizer_p.minimize(-alpha_p * self.ploss)
            self.v_train = self.optimizer_v.minimize(
                alpha_v * self.vloss,
                var_list=self.value_vars
            )
        return

    def act(self, state, sess):
        """Returns policy net sample action (in {0,1})"""
        probs = sess.run(self.a_prob, {self.states_St: np.expand_dims(state.flatten(), axis=0)})
        a_t = 1 * (np.random.rand() < probs)
        return a_t.squeeze()

    def train(self, sess, gamma=0.99, bootstrap_value=0.0, window=None):
        # parse buffer
        states = np.stack(self.episode_buffer['states'])
        actions = np.stack(self.episode_buffer['actions'])
        rewards = np.stack(self.episode_buffer['rewards'])
        next_states = np.stack(self.episode_buffer['next_states'])

        # generate state-value estimates; goes into delta_t
        rwds = rewards.ravel()
        vals = sess.run(self.value,
                        feed_dict={self.states_St: np.vstack(states)}).ravel()
        old_pis = sess.run(self.lnPi_t,
                           feed_dict={
                               self.actions_At: np.vstack(actions),
                               self.states_St: np.vstack(states)
                           }).ravel()  # cheating w/ old_pi = pi for now.
        Gt_TD = np.squeeze(calc_V_TD_target(rwds, vals,
                                            gamma=gamma, bootstrap_value=bootstrap_value))
        feed_dict = {
            self.states_St: np.vstack(states),
            self.actions_At: np.vstack(actions),
            self.returns_Gt: np.vstack(Gt_TD),
            self.old_lnPi_t: np.vstack(old_pis)
        }
        # Train
        sess.run([self.v_train, self.p_train], feed_dict=feed_dict)
        # Generate performance statistics to periodically save
        p_ll, v_l, p_ent = sess.run(
            [self.ploss, self.vloss, self.entropy],
            feed_dict=feed_dict
        )
        return p_ll, v_l, p_ent

    def pretrainV(self, sess, gamma=0.95):
        # parse buffer
        states = np.stack(self.episode_buffer['states'])
        rewards = np.stack(self.episode_buffer['rewards'])

        # generate state-value estimates; goes into delta_t
        rwds = rewards.ravel()
        vals = sess.run(self.value, feed_dict={self.states_St: np.vstack(states)}).ravel()
        Gt_TD = np.squeeze(calc_V_TD_target(rwds, vals, gamma=gamma))
        feed_dict = {
            self.states_St: np.vstack(states),
            self.returns_Gt: np.vstack(Gt_TD)
        }
        # Train & Record v_loss
        sess.run([self.v_train], feed_dict=feed_dict)
        v_l = sess.run(self.vloss, feed_dict=feed_dict)
        return v_l
