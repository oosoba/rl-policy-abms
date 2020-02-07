'''
init: Sep2019
Author: OO
Status: basic initialization; no implementation yet
Goals: RF+AC for collusive multi-agents
Design Logic:
    - All agents interact w/ a common env => single menv
    - The shared env is a property of the agent popn => menv is a param
    - RL learners single hive-mind RL policy net & learn to collude k-dim R-model updates
    - All RL agents colluding => single Actor, Sensoria, & Value NN
'''

from abc import abstractmethod

import numpy as np

from embodied_arch.embodied_misc import *
from embodied_arch.embodied_misc import _zdim_
from embodied_arch.misc_helpers import discount
from minoritygame.minority_multienv import MinorityGame_Multiagent_env

sys.path.append('.')

# class: var defaults
tboard_path = "./log"
agent_name = "embodied_agent_IRL"
__version__ = "0.0.1"
_DEBUG_ = False

# model: var defaults
_default_reports_ = ['Perf/Recent Reward',
                     'Losses/Policy LL', 'Losses/Entropy',
                     'Norms/Grad Norm', 'Norms/Var Norm']
_every_ = 100  # 500
_eps_ = 1.e-4
_ent_decay_ = 5e-5
(lrp, lrv) = (1e-2, 5e-2)  # learning rates
_max_len_ = 400

# envs: var defaults
(na_, m_, s_, p_) = (33, 3, 4, 0.5)
_s_size_, _a_size_ = (4, 1)

# bernoulli(p) log-likelihood function
bernoulli_LL = (lambda x, p: x * tf.log(p) + (tf.ones_like(x) - x) * tf.log(tf.ones_like(p) - p))
# entropy functional for bernoulli(p)
bernoulli_H = (lambda p: -p * tf.log(p) - (tf.ones_like(p) - p) * tf.log(tf.ones_like(p) - p))
default_sense_hSeq = (32,)


def summarize(vec):  # vec = tf.stack(list(vec_d.values()), axis=1)
    return tf.reduce_min(vec), tf.reduce_mean(vec), tf.reduce_max(vec)


def summarize_np(vec):
    return np.min(vec), np.mean(vec), np.max(vec)


class EmbodiedAgent_Collusive(object):
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
                 env_=MinorityGame_Multiagent_env,
                 latentDim=_zdim_,
                 space_size=(_s_size_, _a_size_),
                 sensorium=SensoriumNetworkTemplate,
                 actorNN=ActionPolicyNetwork,
                 valueNN=ValueNetwork,
                 _every_=_every_,
                 max_episode_length=_max_len_
                 ):
        self.name = name
        self.model_path = tboard_path + "/train_" + str(self.name)
        self.autosave_every = _every_
        self.max_episode_length = max_episode_length
        self.last_good_model = None

        # monitors
        self.episode_buffer = {
            'states': list(),
            'actions': list(),
            'rewards': list(),
            'next_states': list()
        }
        self.last_total_returns = None
        self.summary_writer = tf.summary.FileWriter(self.model_path)
        self.env = env_  # env setup
        # s_size: state space dim = num_memory bits
        # ideally inferrable from the env spec
        if "state_space_size" in dir(self.env) and "action_space_size" in dir(self.env):
            self.s_size, self.a_size = (self.env.state_space_size, self.env.action_space_size)
        else:
            self.s_size, self.a_size = space_size
        # multi-agent accounting
        # no need to specify which/how many agents are RL; inferred from env.
        self.actor_count = self.env.actor_count
        # infer vector action/state space spec for colluding bloc.
        self.a_size = int(self.a_size * self.actor_count)
        self.s_size = int(self.s_size * self.actor_count)

        with tf.variable_scope(self.name):
            self.states_St = tf.placeholder(
                shape=[None, self.s_size], dtype=tf.float32, name='states_' + self.name)
            self.states_St_prime = tf.placeholder(
                shape=[None, self.s_size], dtype=tf.float32, name='states_prime_' + self.name)
            self.actions_Ats = tf.placeholder(shape=[None, self.a_size],
                                              dtype=tf.float32, name='actions-taken_' + self.name)
            self.returns_Gts = tf.placeholder(shape=[None, 1],
                                              dtype=tf.float32, name='returns_discounted_' + self.name)
            # NNs
            with tf.variable_scope('sensorium'):
                self.sense_z = sensorium(self.states_St, hSeq=default_sense_hSeq, out_dim=latentDim)
            with tf.variable_scope('actor'):
                self.a_probs = actorNN(self.sense_z)
            with tf.variable_scope('critic'):
                self.values = valueNN(self.sense_z)
        return

    @abstractmethod
    def act(self, state, sess):
        """Default action assumes binary action space.
        Returns completely random action (in {0,1})."""
        return self.env.random_actions()

    @abstractmethod
    def train(self, sess, gamma=0.99, upd_list=None, bootstrap_value=0.0):
        pass

    def episode_length(self):
        return len(self.episode_buffer['states'])

    def reset_buffer(self):
        self.episode_buffer = {
            'states': list(),
            'actions': list(),
            'rewards': list(),
            'next_states': list()
        }
        return

    def play(self, sess):
        # add tqdm progressbar?
        self.reset_buffer()
        self.last_total_returns = np.zeros(self.actor_count)
        d = False
        s = self.env.reset()
        # generate a full episode rollout
        while (self.episode_length() < self.max_episode_length) and not d:
            acts_lst = self.act(s, sess)
            s1, r_lst, d, *rest = self.env.step(acts_lst)  # get next state, reward_vec, & done flag
            # (s_t, a_t, r_t, s_{t+1})
            self.episode_buffer['states'].append(s)
            self.episode_buffer['actions'].append(acts_lst.squeeze())
            self.episode_buffer['rewards'].append(r_lst.squeeze())
            self.episode_buffer['next_states'].append(s1)
            s = s1
        self.last_total_returns = np.sum(
            np.array(self.episode_buffer['rewards']),
            axis=0
        )
        return

    def init_graph(self, sess):
        # initialize variables for the attached session
        with sess.as_default(), sess.graph.as_default():
            sess.run(tf.global_variables_initializer())
            self.summary_writer.add_graph(sess.graph)
            self.summary_writer.flush()
        print("Tensorboard logs in: ", self.model_path)
        return

    def work(self, sess, num_epochs, saver, upd_list=None, gamma=0.99):
        print("Starting agent " + str(self.name))
        with sess.as_default(), sess.graph.as_default():
            for tk in range(int(num_epochs)):
                print("\rEpoch no.: {}/{}".format(tk, num_epochs), end="")
                self.reset_buffer()
                while self.episode_length() == 0:
                    self.play(sess)
                # TRAIN: Update the network using episodes in the buffer.
                stats = list(self.train(sess, gamma=gamma, upd_list=upd_list))
                # save summary statistics for tensorboard
                stats.insert(0, summarize_np(self.last_total_returns))
                if any(np.isnan(np.array(stats).ravel())):
                    print('Model issues @Step {}. Stats({}): ( {} )'.format(
                        tk, self.report_labels, stats))
                    if self.last_good_model is not None:
                        saver.restore(sess, self.last_good_model)
                else:
                    self.model_summary(sess, tk, stats, saver, labels_=self.report_labels)
        return

    def model_summary(self, sess, tk, stats_, saver, labels_=_default_reports_):
        '''Adapted summarizer for vector return stats'''
        summary = tf.Summary()
        for k in range(min(len(stats_), len(labels_))):
            summary.value.add(tag=labels_[k], simple_value=float(stats_[k][1]))
        self.summary_writer.add_summary(summary, tk)
        self.summary_writer.flush()

        # to save or not to save...
        if tk % self.autosave_every == 0:
            print('\n\n\tStats @Step {}: \t(Min, Mean, Max)'.format(tk))
            i = 0
            for st in stats_:
                print('{}: {}'.format(labels_[i], st))
                i += 1
            if not any(np.isnan(np.array(stats_).ravel())):
                chkpt_model = self.model_path + '/model-' + str(tk) + '.cptk'
                self.last_good_model = saver.save(sess, chkpt_model)
                print("Saved Model")
            else:
                print("Model problems... Not saved!")
        return


class EmbodiedAgent_IRF(EmbodiedAgent_Collusive):
    def __init__(self, name=agent_name,
                 env_=MinorityGame_Multiagent_env,
                 alpha=5.e-2,
                 latentDim=_zdim_,
                 space_size=(_s_size_, _a_size_),
                 sensorium=SensoriumNetworkTemplate,
                 actorNN=ActionPolicyNetwork,
                 recover=None,
                 _every_=_every_,
                 max_episode_length=_max_len_
                 ):  # hseq=None,
        super().__init__(name=name, env_=env_,
                         latentDim=latentDim, space_size=space_size,
                         sensorium=sensorium, actorNN=actorNN,
                         _every_=_every_, max_episode_length=max_episode_length
                         )
        self.report_labels = ['Perf/Recent Rewards',
                              'Losses/Policy LLs', 'Losses/Policy Entropies']

        self.last_good_model = recover
        self.lnPi_ts = {}
        self.entropies = {}
        self.GlnPi_ts = {}
        self.rflosses = {}
        self.policy_LLs = {}
        self.rf_trainer = {}
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lrp)
        # {name_i: tf.train.AdamOptimizer(learning_rate=lrp) for name_i in self.actor_names} # WoLF access pt?
        with tf.variable_scope(self.name):
            # Intermediate variables
            self.lnPi_ts = bernoulli_LL(self.actions_Ats, self.a_probs)
            self.entropies = tf.clip_by_value(bernoulli_H(self.a_probs), _eps_, 100.)
            self.GlnPi_ts = tf.multiply(self.returns_Gts, self.lnPi_ts)
            # Losses
            self.rflosses = tf.reduce_mean(- alpha * tf.reduce_sum(self.GlnPi_ts))
            # Training ops
            self.rf_trainer = self.optimizer.minimize(
                loss=self.rflosses,
                var_list=[v for v in tf.trainable_variables() if name in v.name]
            )  # probably need to separate A-vs-C gradients later...
            self.policy_LLs = tf.stack(list(self.lnPi_ts.values()), axis=1)
            self.entropy = tf.stack(list(self.entropies.values()), axis=1)
            self.summLLs = summarize(self.policy_LLs)
            self.summEntropy = summarize(self.entropy)
        return

    def act(self, state, sess):
        """Returns vector of p-net sample action (in {0,1})"""
        probs = sess.run(
            self.a_probs,
            {self.states_St: np.expand_dims(state.flatten(), axis=0)}
        ).squeeze()
        a_ts = 1 * (np.random.rand(self.a_size) < probs).squeeze()  # vec -> vec comparison
        return a_ts

    def train(self, sess, gamma=0.95, upd_list=None, bootstrap_value=0.0):
        assert all(np.diff([len(buf) for _, buf in self.episode_buffer.items()]) == 0), \
            "Rollout is not the correct shape"
        # self.episode_buffer.append([s, acts_lst.squeeze(), r_lst, s1])
        states = np.stack(self.episode_buffer['states'])
        actions = np.stack(self.episode_buffer['actions'])
        rewards = np.stack(self.episode_buffer['rewards'])
        next_states = np.stack(self.episode_buffer['next_states'])
        if _DEBUG_:
            print(states)
            input("About to start training call...") 
        discounted_returns = discount(np.hstack([rewards.ravel(), [bootstrap_value]]), gamma)
        discounted_returns = discounted_returns[:-1, None]
        feed_dict = {
            self.states_St: np.vstack(states.squeeze()),
            self.actions_Ats: np.vstack(actions.squeeze()),
            self.returns_Gts: np.vstack(discounted_returns) #, self.states_St_prime: np.vstack(next_states.squeeze())
        }
        sess.run(
            self.rf_trainer,
            feed_dict=feed_dict
        )

        act_dicts = {
            self.states_St: np.vstack(states[:, ...]),
            self.actions_Ats: np.vstack(actions[:, ...])
        }
        lls, ents = sess.run(
            [self.summLLs, self.summEntropy],
            feed_dict=act_dicts
        )
        return lls, ents
