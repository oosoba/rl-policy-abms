'''
init: Feb2019
Author: OO
Goals: All restricted to Policy Gradient Methods...
    - `EmbodiedAgent` implements: (as base class)
        - tripartite algo split
        - embodiment concept
        - play() construct for generating experience tuples
    - `EmbodiedAgent_<Algo>` implements:
        - <Algo>'s training procedures
        - overrides work() fxn in base class
To Do:
    - Implement online training Baselined RF + AC...
        - Mingame is an iterative game with no end. Online training better fit.
'''

import sys

import gym
import numpy as np
import tensorflow as tf
from embodied_misc import *
from embodied_misc import _zdim_
from misc_helpers import discount

sys.path.append('.')

# class: var defaults
tboard_path = "./log"
agent_name = "embodied_agent"
__version__ = "0.0.3"
_DEBUG_ = False

# model: var defaults
_default_reports_ = ['Perf/Recent Reward',
                     'Losses/Policy LL', 'Losses/Entropy',
                     'Norms/Grad Norm', 'Norms/Var Norm']
_every_ = 500
_ent_decay_ = 5e-3
(lrp, lrv) = (1e-2, 5e-2)  # learning rates
_max_len_ = 400

# envs: var defaults
(na_, m_, s_, p_) = (33, 3, 4, 0.5)
_s_size_, _a_size_ = (4, 1)


class EmbodiedAgent(object):
    '''Base class for embodied agents. Implements:
        - connection to specified environment
        - base tripartite network (sensorium|action|value) for agent responses
        - play() construct for sampling rollouts
        - template for work() function
        - model init, report, & save functions
    Usage:
    *Subclass* this class with (overriding) implementations of:
        - derived tf variables & fxns needed for your training algo
        - train(): takes rollout and updates NN weights
        - work(): <may not need this override>
    '''

    def __init__(self, name=agent_name,
                 env_=gym.make('CartPole-v0'),
                 sensorium=SensoriumNetworkTemplate,
                 latentDim=_zdim_, space_size=(_s_size_, _a_size_)):
        self.name = name
        self.model_path = tboard_path + "/train_" + str(self.name)
        self.autosave_every = _every_
        self.max_episode_length = _max_len_
        self.last_good_model = None

        # monitors
        self.episode_buffer = list()
        self.last_total_return = None
        self.summary_writer = tf.summary.FileWriter(self.model_path)
        self.env = env_  # env setup
        if "state_space_size" in dir(self.env) and "action_space_size" in dir(self.env):
            self.s_size, self.a_size = (self.env.state_space_size, self.env.action_space_size)
            # s_size: state space dim = num_memory bits
            # ideally inferrable from the env spec
        else:
            self.s_size, self.a_size = space_size

        with tf.variable_scope(self.name):
            self.states_St = tf.placeholder(
                shape=[None, self.s_size], dtype=tf.float32, name='states')
            self.actions_At = tf.placeholder(
                shape=[None, self.a_size], dtype=tf.float32, name='actions-taken')
            self.returns_Gt = tf.placeholder(
                shape=[None, 1], dtype=tf.float32, name='returns_discounted')
            self.states_St_prime = tf.placeholder(
                shape=[None, self.s_size], dtype=tf.float32, name='states_prime')

            with tf.variable_scope('sensorium'):  # as sense_scope:
                self.sense_z = sensorium(self.states_St, out_dim=latentDim)
            with tf.variable_scope('action'):  # \pi(s,a)
                self.action_dist = ActionPolicyNetwork(self.sense_z)
            with tf.variable_scope('value_fxn'):
                self.value = ValueNetwork(self.sense_z)
        return

    def act(self, state, sess):
        """Returns policy net sample action (in {0,1})"""
        a_t = sess.run(
            self.action_dist.sample(),
            feed_dict={self.states_St: np.expand_dims(state.flatten(), axis=0)}
        )
        # feed_dict={self.states_St: state.reshape((-1, self.s_size))}
        # feed_dict = {self.states_St: np.expand_dims(state.flatten(), axis=0)}
        # feed_dict={self.states_St: np.vstack(state)}
        return np.array(a_t).squeeze()

    # def act_rand(self):
    #     """Returns completely random action (in {0,1})
    #     Only works for binary action spaces for now."""
    #     return np.squeeze(np.random.randint(low=0, high=2))

    def play(self, sess):
        self.episode_buffer = []  # reset experience buffer/empty rollout
        self.last_total_return = 0.
        d = False
        s = self.env.reset()
        # generate a full episode rollout (self.brain.episode_buffer)
        while (len(self.episode_buffer) < self.max_episode_length) and not d:
            act_pn = self.act(s, sess)  # print("Selected Action: ", act_pn)
            s1, r, d, *rest = self.env.step(act_pn)  # get next state, reward, & done flag
            self.episode_buffer.append([s, act_pn.squeeze(), float(r), s1])
            # (s_t, a_t, r_t, s_{t+1})
            self.last_total_return += float(r)
            s = s1
        return

    def init_graph(self, sess):
        # initialize variables for the attached session
        with sess.as_default(), sess.graph.as_default():
            sess.run(tf.global_variables_initializer())
            self.summary_writer.add_graph(sess.graph)
            self.summary_writer.flush()
        print("Tensorboard logs in: ", self.model_path)
        return

    def work(self, sess, saver, num_epochs, gamma):
        print("Starting agent " + str(self.name))
        with sess.as_default(), sess.graph.as_default():
            for tk in range(int(num_epochs)):
                print("\rEpoch no.: {}/{}".format(tk, num_epochs), end="")
                self.episode_buffer = []
                while len(self.episode_buffer) == 0:
                    self.play(sess)
                # save summary statistics for tensorboard
                stats = [self.last_total_return]
                self.model_summary(tk, stats_=stats)
        return

    def model_summary(self, tk, stats_, labels_=_default_reports_):
        summary = tf.Summary()
        for k in range(min(len(stats_), len(labels_))):
            summary.value.add(tag=labels_[k], simple_value=float(stats_[k]))
        self.summary_writer.add_summary(summary, tk)
        self.summary_writer.flush()
        return

    def model_save(self, sess, saver, tk, stats_, labels_=_default_reports_):
        print('\nStep {}: Stats({}): ( {} )'.format(tk, labels_, stats_))
        if not any(np.isnan(stats_)):
            chkpt_model = self.model_path + '/model-' + str(tk) + '.cptk'
            self.last_good_model = saver.save(sess, chkpt_model)
            print("Saved Model")
        else:
            print("Model problems... Not saved!")
        return


class EmbodiedAgentRF(EmbodiedAgent):
    def __init__(self, name=agent_name,
                 env_=gym.make('CartPole-v0'),
                 sensorium=SensoriumNetworkTemplate,
                 latentDim=_zdim_, alpha=5e-2,
                 space_size=(_s_size_, _a_size_),
                 recover=None):
        super().__init__(name, env_, sensorium, latentDim, space_size)
        self.report_labels = ['Perf/Recent Reward', 'Losses/Policy LL', 'Losses/Entropy']
        self.optimizer_p = tf.train.AdamOptimizer(learning_rate=lrp)
        self.last_good_model = recover

        with tf.variable_scope(self.name):  # slim stacks take care of regularization
            # Intermediate variables
            self.lnPi_t = self.action_dist.log_prob(value=self.actions_At)  # ln \pi(a_taken|s)
            self.GlnPi_t = self.returns_Gt * self.lnPi_t  # 'GlnP'
            # Losses and Gradients
            self.entropy = tf.clip_by_value(
                tf.reduce_mean(self.action_dist.entropy()),
                1.e-4, 100.)  # nan entropies fucking things up
            # Avg. over all S_t
            self.rfloss = - alpha * tf.reduce_mean(self.GlnPi_t)
            self.policy_LL = tf.reduce_mean(self.lnPi_t)
            self.rf_grads = self.optimizer_p.compute_gradients(loss=self.rfloss)

            # Separate out: Training Vars
            # self.policy_vars = [v for v in tf.trainable_variables()
            #                     if "action" in v.name or "sensorium" in v.name]
            # self.rf_grads = self.optimizer_p.compute_gradients(
            #     loss=(self.rfloss - alpha_p*_ent_decay_*self.entropy),
            #     var_list=self.policy_vars
            # )
            # to prevent saturation during training...
            # self.rf_grads = tf.clip_by_value(self.rf_grads, -_g_thresh_, _g_thresh_)
            # clipGrads(...), tf.clip_by_norm(.., clip_norm=_g_thresh_)

            # Create training operations
            self.rf_train = self.optimizer_p.apply_gradients(self.rf_grads)
        return

    def train(self, sess, gamma=0.95, bootstrap_value=0.0):
        buf = np.array(self.episode_buffer)
        buf = checkRollout(buf)
        # env.step returns
        # ... determines order of buffer entries...
        states = np.squeeze(buf[:, 0])
        actions = np.squeeze(buf[:, 1])
        rewards = np.squeeze(buf[:, 2])
        next_states = np.squeeze(buf[:, 3])

        # generate discounted returns; goes into G_t
        discounted_returns = discount(np.hstack([rewards, [bootstrap_value]]), gamma)
        discounted_returns = discounted_returns[:-1, None]
        if _DEBUG_:
            # print(self.episode_buffer)
            print(states)
            # print( list(map(lambda x: np.shape(x), [states, actions, discounted_rewards])) )
            input("About to start training call...")
        feed_dict = {
            self.states_St: np.vstack(states),
            self.actions_At: np.vstack(actions),
            self.returns_Gt: np.vstack(discounted_returns),
            self.states_St_prime: np.vstack(next_states)
        }
        # Generate network statistics to periodically save
        p_ll, p_ent, _ = sess.run(
            [self.policy_LL, self.entropy, self.rf_train],
            feed_dict=feed_dict
        )
        return p_ll, p_ent

    def work(self, sess, saver, num_epochs, gamma=0.95):
        print("Starting agent " + str(self.name))
        with sess.as_default(), sess.graph.as_default():
            for tk in range(int(num_epochs)):
                print("\rEpoch no.: {}/{}".format(tk, num_epochs), end="")
                self.episode_buffer = []
                while len(self.episode_buffer) == 0:
                    self.play(sess)
                # TRAIN: Update the network using episodes in the buffer.
                stats = list(self.train(sess, gamma))
                # save summary statistics for tensorboard
                stats.insert(0, self.last_total_return)
                if any(np.isnan(stats)):
                    saver.restore(sess, self.last_good_model)
                    print('Model issues @Step {}. Stats({}): ( {} )'.format(
                        tk, self.report_labels, stats))
                else:
                    super().model_summary(tk, stats, labels_=self.report_labels)
                # save model parameters periodically
                if tk % self.autosave_every == 0:
                    super().model_save(sess, saver, tk, stats, labels_=self.report_labels)
        return


class EmbodiedAgentRFBaselined(EmbodiedAgent):
    def __init__(self, name=agent_name,
                 env_=gym.make('CartPole-v0'),
                 sensorium=SensoriumNetworkTemplate,
                 latentDim=_zdim_, alpha_p=5e-3, alpha_v=1e-3,
                 space_size=(_s_size_, _a_size_),
                 recover=None
                 ):
        super().__init__(name, env_, sensorium, latentDim, space_size)

        self.optimizer_p = tf.train.AdamOptimizer(learning_rate=lrp)  # GradientDescentOptimizer
        self.optimizer_v = tf.train.AdamOptimizer(learning_rate=lrv)
        self.report_labels = ['Perf/Recent Reward',
                              'Losses/Policy LL', 'Losses/Value Fxn', 'Losses/Entropy']
        self.last_good_model = recover

        with tf.variable_scope(self.name):  # slim stacks take care of regularization
            # Intermediate variables; may need some tf.reshape(..., [-1])
            self.lnPi_t = self.action_dist.log_prob(value=self.actions_At)  # ln \pi(a_taken|s)
            self.Advs_t = self.returns_Gt - self.value
            self.AdvlnPi_t = tf.stop_gradient(self.Advs_t) * self.lnPi_t  # 'AdvlnP'
            self.policy_LL = tf.reduce_mean(self.lnPi_t)  # for reporting

            # Losses
            self.entropy = tf.clip_by_value(
                tf.reduce_mean(self.action_dist.entropy()),
                1.e-2, 100.)  # Avg. over all S_t
            self.ploss = - alpha_p * tf.reduce_mean(self.AdvlnPi_t)
            self.vloss = 0.5 * alpha_v * tf.reduce_mean(
                tf.square(
                    self.returns_Gt - tf.reshape(self.value, [-1])
                )
            )  # equiv to minimizing 0.5(targ-v)^2; adv = targ-v

            # Separate out: Training Vars
            self.policy_vars = [v for v in tf.trainable_variables()
                                if "action" in v.name or "sensorium" in v.name]
            self.value_vars = [v for v in tf.trainable_variables()
                               if "value_fxn" in v.name or "sensorium" in v.name]
            # Gradients
            self.p_grads = self.optimizer_p.compute_gradients(
                loss=(self.ploss - alpha_p * _ent_decay_ * self.entropy),
                var_list=self.policy_vars
            )
            # self.p_grads = clipGrads(self.p_grads)  # to prevent saturation during training...
            # Create training operations
            self.v_train = self.optimizer_v.minimize(
                self.vloss  # , var_list=self.value_vars
            )
            self.p_train = self.optimizer_p.apply_gradients(self.p_grads)
        return

    def work(self, sess, saver, num_epochs, gamma=0.99):
        print("Starting agent " + str(self.name))
        with sess.as_default(), sess.graph.as_default():
            for tk in range(int(num_epochs)):
                print("\rEpoch no.: {}/{}".format(tk, num_epochs), end="")
                self.episode_buffer = []
                while len(self.episode_buffer) == 0:
                    self.play(sess)
                # TRAIN: Update the network using episodes in the buffer.
                stats = list(self.train(sess, gamma))
                # save summary statistics for tensorboard
                stats.insert(0, self.last_total_return)
                if any(np.isnan(stats)):
                    print('Model issues @Step {}. Stats({}): ( {} )'.format(
                        tk, self.report_labels, stats))
                    saver.restore(sess, self.last_good_model)
                else:
                    self.model_summary(tk, stats, labels_=self.report_labels)
                # save model parameters periodically
                if tk % self.autosave_every == 0:
                    self.model_save(sess, saver, tk, stats, labels_=self.report_labels)
        return

    def train(self, sess, gamma=0.99, bootstrap_value=0.0):
        buf = np.array(self.episode_buffer)
        buf = checkRollout(buf)

        states = np.squeeze(buf[:, 0])
        actions = np.squeeze(buf[:, 1])
        rewards = np.squeeze(buf[:, 2])
        next_states = np.squeeze(buf[:, 3])

        # generate discounted returns; goes into G_t
        discounted_returns = discount(np.hstack([rewards, [bootstrap_value]]), gamma)
        discounted_returns = discounted_returns[:-1, None]
        if _DEBUG_:
            # print(self.episode_buffer)
            print(states)
            # print( list(map(lambda x: np.shape(x), [states, actions, discounted_rewards])) )
            input("About to start training call...")
        feed_dict = {
            self.states_St: np.vstack(states),
            self.actions_At: np.vstack(actions),
            self.returns_Gt: np.vstack(discounted_returns),
            self.states_St_prime: np.vstack(next_states)
        }
        # Train
        sess.run([self.v_train, self.p_train], feed_dict=feed_dict)
        # Generate performance statistics to periodically save
        p_ll, v_l, p_ent = sess.run([self.policy_LL, self.vloss, self.entropy], feed_dict=feed_dict)
        return p_ll, v_l, p_ent
