'''
init: Dec2019
Author: OO
Goals: Base class for population of RL agents

Design Logic:
    - All agents interact w/ a common env => single menv
    - The shared env is a property of the agent popn => menv is a param
    - No training/optimizers; just fxn stubs
    - Needs flexibility to allow diff. learning algos
        - May need to inherit + add new TF structures e.g. for collusion/shared info
'''

from abc import abstractmethod

from embodied_arch.embodied_misc import *
from embodied_arch.embodied_misc import _zdim_
from minoritygame.minority_multienv import MinorityGame_Multiagent_env

# bernoulli_H, bernoulli_LL, summarize, summarize_np


sys.path.append('.')

# class: var defaults
tboard_path = "./log"
agent_name = "embodied_popn"
__version__ = "0.0.1"
_DEBUG_ = False

# model: var defaults
_default_reports_ = ['Perf/Recent Reward',
                     'Losses/Policy LL', 'Losses/Entropy',
                     'Norms/Grad Norm', 'Norms/Var Norm']
_every_ = 100  # 500
# _eps_ = 1.e-2  # 1.e-5
# _ent_decay_ = 5e-4
(lrp, lrv) = (1e-2, 5e-2)  # learning rates
_max_len_ = 400

# envs: var defaults
(na_, m_, s_, p_) = (33, 3, 4, 0.5)
_s_size_, _a_size_ = (4, 1)


class EmbodiedAgent_Population(object):
    '''Base class for population of embodied agents. Implements:
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
        self.total_epoch_count = 0  # num of training seasons agents have gone through
        self.summary_writer = tf.summary.FileWriter(self.model_path)
        self.env = env_  # env setup

        self.report_labels = ['Perf/Recent Rewards',
                              'Losses/Policy LLs',
                              'Losses/Policy Entropies']

        # s_size: state space dim = num_memory bits
        # ideally inferrable from the env spec
        if "state_space_size" in dir(self.env) and "action_space_size" in dir(self.env):
            self.s_size, self.a_size = (self.env.state_space_size,
                                        self.env.action_space_size)
        else:
            self.s_size, self.a_size = space_size
        # multi-agent accounting
        # no need to specify which/how many agents are RL; inferred from env.
        self.actor_count = self.env.actor_count
        self.actor_names = ['player_' + str(j) for j in range(self.actor_count)]

        with tf.variable_scope(self.name):
            # private A_t, G_t, S_t, S'_t for each agent
            self.states_St = {}
            self.states_St_prime = {}
            self.returns_Gts = {}
            self.actions_Ats = {}
            for name in self.actor_names:  # placeholders
                # no longer sharing S_t or S_t_prime across all agents.
                self.states_St[name] = tf.placeholder(
                    shape=[None, self.s_size], dtype=tf.float32, name='states_' + name)
                self.states_St_prime[name] = tf.placeholder(
                    shape=[None, self.s_size], dtype=tf.float32, name='states_prime_' + name)
                self.actions_Ats[name] = tf.placeholder(shape=[None, self.a_size],
                                                        dtype=tf.float32, name='actions-taken_' + name)
                self.returns_Gts[name] = tf.placeholder(shape=[None, 1],
                                                        dtype=tf.float32, name='returns_discounted_' + name)
            self.sense_z = {}
            self.a_probs = {}
            self.a_logits = {}
            self.values = {}
            for name in self.actor_names:  # NNs
                with tf.variable_scope(name):
                    with tf.variable_scope('sensorium'):
                        self.sense_z[name] = sensorium(self.states_St[name],
                                                       out_dim=latentDim)  # hSeq=default_sense_hSeq,
                    with tf.variable_scope('actor'):
                        self.a_logits[name] = actorNN(self.sense_z[name])
                        self.a_probs[name] = tf.nn.sigmoid(self.a_logits[name])
                    with tf.variable_scope('critic'):
                        self.values[name] = valueNN(self.sense_z[name])
            # tf-collations of (A_t, G_t, S'_t) tuples & NN outputs: tf.stack(list(dict.values()), axis=1)
            self.actions_At_tf = tf.stack(list(self.actions_Ats.values()), axis=1)
            self.returns_Gt_tf = tf.stack(list(self.returns_Gts.values()), axis=1)
            self.senses_tf = tf.stack(list(self.sense_z.values()), axis=1)
            self.a_logits_tf = tf.stack(list(self.a_logits.values()), axis=1)
            self.a_probs_tf = tf.stack(list(self.a_probs.values()), axis=1)
            self.values_tf = tf.stack(list(self.values.values()), axis=1)
        return

    @abstractmethod
    def act(self, state, sess):
        """Default action assumes binary action space.
        Returns completely random action (in {0,1})."""
        return self.env.random_actions()

    @abstractmethod
    def generate_summary(self, sess, act_dicts):
        pass

    @abstractmethod
    def train_single(self, sess, agent_index, rollout, Qsa_i,
                     gamma=0.95, bootstrap_value=0.0):
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

    def reset_training_clock(self):
        '''resets training counter.
        useful for reverting learning rate schedule back to high initial values.'''
        self.total_epoch_count = 0
        return

    def play(self, sess, terminal_reward=-20.0):
        # add tqdm progressbar?
        self.reset_buffer()
        self.last_total_returns = np.zeros(self.actor_count)
        d = False
        s = self.env.reset()
        # generate a full episode rollout
        while (self.episode_length() < self.max_episode_length) and not d:
            acts_lst = self.act(s, sess)
            s1, r_lst, d, *rest = self.env.step(acts_lst)  # get next state, reward_vec, & done flag
            if d:
                r_lst = terminal_reward * np.ones(self.actor_count)  # reinforcement for early termination, default: -ve
            # self.episode_buffer.append([s, acts_lst.squeeze(), r_lst.squeeze(), s1])
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
        totals = []
        with sess.as_default(), sess.graph.as_default():
            for tk in range(int(num_epochs)):
                print("\rEpoch no.: {}/{}".format(tk, num_epochs), end="")
                self.reset_buffer()
                self.total_epoch_count += 1
                while self.episode_length() == 0:
                    self.play(sess)
                # TRAIN: Update the network using episodes in the buffer.
                stats = list(self.train(sess, gamma=gamma, upd_list=upd_list))
                # save summary statistics for tensorboard
                stats.insert(0, summarize_np(self.last_total_returns))
                totals.append(stats[0])
                if any(np.isnan(np.array(stats).ravel())):
                    print('Model issues @Step {}. Stats({}): ( {} )'.format(
                        tk, self.report_labels, stats))
                    if self.last_good_model is not None:
                        saver.restore(sess, self.last_good_model)
                        print("\nRestored Last Good Model: ", self.last_good_model)
                else:
                    self.model_summary(sess, tk, stats, saver, labels_=self.report_labels)
        return np.array(totals)

    def train(self, sess, gamma=0.99, upd_list=None, bootstrap_value=0.0):
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
        learners = range(self.actor_count) if (upd_list is None) else upd_list
        for agent_idx in learners:
            self.train_single(sess, agent_index=agent_idx,
                              rollout=(states[:, agent_idx, ...],
                                       actions[:, agent_idx, ...],
                                       rewards[:, agent_idx, ...],
                                       next_states[:, agent_idx, ...]),
                              gamma=gamma, bootstrap_value=bootstrap_value)
        # Generate learning model statistics to periodically save
        act_dicts = {
            **{self.states_St[self.actor_names[idx]]: np.vstack(states[:, idx, ...])
               for idx in learners},
            **{self.actions_Ats[self.actor_names[idx]]: np.vstack(actions[:, idx, ...])
               for idx in learners}
        }
        reports = self.generate_summary(sess, act_dicts)
        return reports

    def model_summary(self, sess, tk, stats_,
                      saver, labels_=_default_reports_):
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
                print("\n\n\t\tModel problems... Not saved! Attempting Recovery...")
                if self.last_good_model is not None:
                    saver.restore(sess, self.last_good_model)
                    print("\nRestored Last Good Model: ", self.last_good_model)
        return

## Old/Init Base Class; for reference
# class EmbodiedAgent_Independent(object):
#     '''Base class for embodied agents. Implements:
#         - connection to specified environment
#         - base tripartite network (sensorium|action|value) for agent responses
#         - play() construct for sampling rollouts
#         - template for work() function
#         - model init, report, & save functions
#     Usage:
#     *Subclass* this class with (overriding) implementations of:
#         - derived tf variables & fxns needed for your training algo
#         - train(): takes rollout and updates NN weights
#         - work(): <may not need this override>
#     '''
#
#     def __init__(self, name=agent_name,
#                  env_=MinorityGame_Multiagent_env,
#                  latentDim=_zdim_,
#                  space_size=(_s_size_, _a_size_),
#                  sensorium=SensoriumNetworkTemplate,
#                  actorNN=ActionPolicyNetwork,
#                  valueNN=ValueNetwork,
#                  _every_=_every_,
#                  max_episode_length=_max_len_
#                  ):
#         self.name = name
#         self.model_path = tboard_path + "/train_" + str(self.name)
#         self.autosave_every = _every_
#         self.max_episode_length = max_episode_length
#         self.last_good_model = None
#
#         # monitors
#         self.episode_buffer = {
#             'states': list(),
#             'actions': list(),
#             'rewards': list(),
#             'next_states': list()
#         }
#         self.last_total_returns = None
#         self.summary_writer = tf.summary.FileWriter(self.model_path)
#         self.env = env_  # env setup
#         # s_size: state space dim = num_memory bits
#         # ideally inferrable from the env spec
#         if "state_space_size" in dir(self.env) and "action_space_size" in dir(self.env):
#             self.s_size, self.a_size = (self.env.state_space_size,
#                                         self.env.action_space_size)
#         else:
#             self.s_size, self.a_size = space_size
#         # multi-agent accounting
#         # no need to specify which/how many agents are RL; inferred from env.
#         self.actor_count = self.env.actor_count
#         self.actor_names = ['player_' + str(j) for j in range(self.actor_count)]
#
#         with tf.variable_scope(self.name):
#             # private A_t, G_t, S_t, S'_t for each agent
#             self.states_St = {}
#             self.states_St_prime = {}
#             self.returns_Gts = {}
#             self.actions_Ats = {}
#             for name in self.actor_names:  # placeholders
#                 # no longer sharing S_t or S_t_prime across all agents.
#                 self.states_St[name] = tf.placeholder(
#                     shape=[None, self.s_size], dtype=tf.float32, name='states_' + name)
#                 self.states_St_prime[name] = tf.placeholder(
#                     shape=[None, self.s_size], dtype=tf.float32, name='states_prime_' + name)
#                 self.actions_Ats[name] = tf.placeholder(shape=[None, self.a_size],
#                                                         dtype=tf.float32, name='actions-taken_' + name)
#                 self.returns_Gts[name] = tf.placeholder(shape=[None, 1],
#                                                         dtype=tf.float32, name='returns_discounted_' + name)
#             self.sense_z = {}
#             self.a_probs = {}
#             self.a_logits = {}
#             self.values = {}
#             for name in self.actor_names:  # NNs
#                 with tf.variable_scope(name):
#                     with tf.variable_scope('sensorium'):
#                         self.sense_z[name] = sensorium(self.states_St[name],
#                                                        out_dim=latentDim)  # hSeq=default_sense_hSeq,
#                     with tf.variable_scope('actor'):
#                         self.a_logits[name] = actorNN(self.sense_z[name])
#                         self.a_probs[name] = tf.nn.sigmoid(self.a_logits[name])
#                     with tf.variable_scope('critic'):
#                         self.values[name] = valueNN(self.sense_z[name])
#             # tf-collations of (A_t, G_t, S'_t) tuples & NN outputs: tf.stack(list(dict.values()), axis=1)
#             self.actions_At_tf = tf.stack(list(self.actions_Ats.values()), axis=1)
#             self.returns_Gt_tf = tf.stack(list(self.returns_Gts.values()), axis=1)
#             self.senses_tf = tf.stack(list(self.sense_z.values()), axis=1)
#             self.a_logits_tf = tf.stack(list(self.a_logits.values()), axis=1)
#             self.a_probs_tf = tf.stack(list(self.a_probs.values()), axis=1)
#             self.values_tf = tf.stack(list(self.values.values()), axis=1)
#         return
#
#     @abstractmethod
#     def act(self, state, sess):
#         """Default action assumes binary action space.
#         Returns completely random action (in {0,1})."""
#         return self.env.random_actions()
#
#     @abstractmethod
#     def generate_summary(self, sess, act_dicts):
#         pass
#
#     @abstractmethod
#     def train_single(self, sess, agent_index, rollout, gamma=0.95, bootstrap_value=0.0):
#         pass
#
#     def episode_length(self):
#         return len(self.episode_buffer['states'])
#
#     def reset_buffer(self):
#         self.episode_buffer = {
#             'states': list(),
#             'actions': list(),
#             'rewards': list(),
#             'next_states': list()
#         }
#         return
#
#     def play(self, sess, terminal_reward=-20.0):
#         # add tqdm progressbar?
#         self.reset_buffer()
#         self.last_total_returns = np.zeros(self.actor_count)
#         d = False
#         s = self.env.reset()
#         # generate a full episode rollout
#         while (self.episode_length() < self.max_episode_length) and not d:
#             acts_lst = self.act(s, sess)
#             s1, r_lst, d, *rest = self.env.step(acts_lst)  # get next state, reward_vec, & done flag
#             if d:
#                 r_lst = terminal_reward * np.ones(self.actor_count)  # reinforcement for early termination, default: -ve
#             # self.episode_buffer.append([s, acts_lst.squeeze(), r_lst.squeeze(), s1])
#             self.episode_buffer['states'].append(s)
#             self.episode_buffer['actions'].append(acts_lst.squeeze())
#             self.episode_buffer['rewards'].append(r_lst.squeeze())
#             self.episode_buffer['next_states'].append(s1)
#             s = s1
#         self.last_total_returns = np.sum(
#             np.array(self.episode_buffer['rewards']),
#             axis=0
#         )
#         return
#
#     def init_graph(self, sess):
#         # initialize variables for the attached session
#         with sess.as_default(), sess.graph.as_default():
#             sess.run(tf.global_variables_initializer())
#             self.summary_writer.add_graph(sess.graph)
#             self.summary_writer.flush()
#         print("Tensorboard logs in: ", self.model_path)
#         return
#
#     def work(self, sess, num_epochs, saver, upd_list=None, gamma=0.99):
#         print("Starting agent " + str(self.name))
#         with sess.as_default(), sess.graph.as_default():
#             for tk in range(int(num_epochs)):
#                 print("\rEpoch no.: {}/{}".format(tk, num_epochs), end="")
#                 self.reset_buffer()
#                 while self.episode_length() == 0:
#                     self.play(sess)
#                 # TRAIN: Update the network using episodes in the buffer.
#                 stats = list(self.train(sess, gamma=gamma, upd_list=upd_list))
#                 # save summary statistics for tensorboard
#                 stats.insert(0, summarize_np(self.last_total_returns))
#                 if any(np.isnan(np.array(stats).ravel())):
#                     print('Model issues @Step {}. Stats({}): ( {} )'.format(
#                         tk, self.report_labels, stats))
#                     if self.last_good_model is not None:
#                         saver.restore(sess, self.last_good_model)
#                 else:
#                     self.model_summary(sess, tk, stats, saver, labels_=self.report_labels)
#         return
#
#     def train(self, sess, gamma=0.99, upd_list=None, bootstrap_value=0.0):
#         assert all(np.diff([len(buf) for _, buf in self.episode_buffer.items()]) == 0), \
#             "Rollout is not the correct shape"
#         # self.episode_buffer.append([s, acts_lst.squeeze(), r_lst, s1])
#         states = np.stack(self.episode_buffer['states'])
#         actions = np.stack(self.episode_buffer['actions'])
#         rewards = np.stack(self.episode_buffer['rewards'])
#         next_states = np.stack(self.episode_buffer['next_states'])
#         if _DEBUG_:
#             print(states)
#             input("About to start training call...")
#         learners = range(self.actor_count) if (upd_list is None) else upd_list
#         for agent_idx in learners:
#             self.train_single(sess, agent_index=agent_idx,
#                               rollout=(states[:, agent_idx, ...],
#                                        actions[:, agent_idx, ...],
#                                        rewards[:, agent_idx, ...],
#                                        next_states[:, agent_idx, ...]),
#                               gamma=gamma, bootstrap_value=bootstrap_value)
#         # Generate learning model statistics to periodically save
#         act_dicts = {
#             **{self.states_St[self.actor_names[idx]]: np.vstack(states[:, idx, ...])
#                for idx in learners},
#             **{self.actions_Ats[self.actor_names[idx]]: np.vstack(actions[:, idx, ...])
#                for idx in learners}
#         }
#         reports = self.generate_summary(sess, act_dicts)
#         return reports
#
#     def model_summary(self, sess, tk, stats_, saver, labels_=_default_reports_):
#         '''Adapted summarizer for vector return stats'''
#         summary = tf.Summary()
#         for k in range(min(len(stats_), len(labels_))):
#             summary.value.add(tag=labels_[k], simple_value=float(stats_[k][1]))
#         self.summary_writer.add_summary(summary, tk)
#         self.summary_writer.flush()
#
#         # to save or not to save...
#         if tk % self.autosave_every == 0:
#             print('\n\n\tStats @Step {}: \t(Min, Mean, Max)'.format(tk))
#             i = 0
#             for st in stats_:
#                 print('{}: {}'.format(labels_[i], st))
#                 i += 1
#             if not any(np.isnan(np.array(stats_).ravel())):
#                 chkpt_model = self.model_path + '/model-' + str(tk) + '.cptk'
#                 self.last_good_model = saver.save(sess, chkpt_model)
#                 print("Saved Model")
#             else:
#                 print("Model problems... Not saved!")
#         return