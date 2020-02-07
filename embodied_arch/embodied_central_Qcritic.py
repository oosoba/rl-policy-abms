'''
init: Nov2019
Author: OO
Goals: AC with centralized training-time critic for multiple agents
(See: Lowe-et-al.'s "Multi-agent actor-critic for mixed cooperative-competitive environments." )
Still in development;
Design Logic:
    - All agents interact w/ a common env => single menv
    - The shared env is a property of the agent popn => menv is a param
    - Independent policy nets for each agent
        - All RL agents act independently => separate indeps Actor & Value NNs
    - Shared critic for all agents
        - using state-action value critic fxn, Q (instead of state value fxn, V)
        - training is collusive (via Q), evaluation/running is indep.
    - No explicit collusion or shared learning => separate sensoria NNs
'''

import tensorflow.contrib.slim as slim

from embodied_arch.EmbodiedPopulation import EmbodiedAgent_Population
from embodied_arch.embodied_misc import *
from embodied_arch.embodied_misc import _sched_win_, cyc_learning_spd
from embodied_arch.embodied_misc import _zdim_, calc_advantages, calc_Q_TD_target, checkRolloutDict
from embodied_arch.embodied_misc import bernoulli_H, summarize
from minoritygame.minority_multienv import MinorityGame_Multiagent_env

sys.path.append('.')

# class: var defaults
tboard_path = "./log"
agent_name = "embodied_agent_MAC"
__version__ = "0.0.2"
_DEBUG_ = False

# model: var defaults
_default_reports_ = ['Perf/Recent Reward',
                     'Losses/Policy LL', 'Losses/Entropy',
                     'Norms/Grad Norm', 'Norms/Var Norm']
_every_ = 1000  # 500
_eps_ = 1.e-1  # 1.e-5
_ent_decay_ = 5e-2  # 5e-3
(lrp, lrv) = (1e-2, 5e-2)  # learning rates
_max_len_ = 400
# default_sense_hSeq = (32,)

# envs: var defaults
(na_, m_, s_, p_) = (33, 3, 4, 0.5)
_s_size_, _a_size_ = (4, 1)


def QsaNetwork(senses_all, actions_all, hSeq=None, gamma_reg=1.):
    '''Global State-Action Value Network'''
    if hSeq is None:
        hSeq = (128, 128)

    ## prep popn inputs
    # collate = list(zip(
    #     [sv for _, sv in senses_all.items()],
    #     [av for _, av in actions_all.items()]
    # ))  # {(s_i, a_i)}_i
    # full_inp = tf.reshape(
    #     tf.concat(collate, axis=1),
    #     [-1]
    # )
    nagents = len(senses_all.keys())

    ## setup Q_global NN
    regularizer = slim.l2_regularizer(gamma_reg)
    full_inp = slim.flatten(tf.concat([
        tf.concat([sv for _, sv in senses_all.items()], axis=1),
        tf.concat([sv for _, sv in actions_all.items()], axis=1)
    ], axis=1))
    hidden = slim.fully_connected(
        slim.flatten(full_inp),
        num_outputs=hSeq[0],
        activation_fn=tf.nn.relu,
        weights_regularizer=regularizer,
        weights_initializer=tf.truncated_normal_initializer(stddev=.25),
        biases_initializer=tf.truncated_normal_initializer(stddev=.25)
    )
    if (len(hSeq) > 1):
        hidden = slim.stack(hidden,
                            slim.fully_connected,
                            list(hSeq[1:]),
                            activation_fn=tf.nn.relu,
                            weights_regularizer=regularizer,
                            weights_initializer=tf.truncated_normal_initializer(stddev=.25),
                            biases_initializer=tf.truncated_normal_initializer(stddev=.25)
                            )
    val = slim.fully_connected(
        hidden,
        num_outputs=nagents,  # num_outputs=1,
        activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=.25),
        biases_initializer=tf.truncated_normal_initializer(stddev=.25)
    )  # this is a regression network...
    return val


class EmbodiedAgent_MAC(EmbodiedAgent_Population):
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
                 QNN=QsaNetwork,
                 alpha_p=5e-2, alpha_v=1e-1, alpha_q=1e-1,
                 _every_=_every_, recover=None,
                 max_episode_length=_max_len_,
                 CyclicSchedule=None
                 ):
        super().__init__(name=name, env_=env_,
                         latentDim=latentDim, space_size=space_size,
                         sensorium=sensorium, actorNN=actorNN,
                         valueNN=valueNN,
                         _every_=_every_, max_episode_length=max_episode_length
                         )

        self.optimizers_p = {name_i: tf.train.AdamOptimizer(
            learning_rate=lrp) for name_i in self.actor_names}
        self.optimizers_v = {
            name_i: tf.train.AdamOptimizer(learning_rate=lrv)
            for name_i in self.actor_names
        }
        self.optimizer_q = tf.train.AdamOptimizer(learning_rate=lrv)

        ## Setup Cyclic Learning Rate Schedule
        self.init_spd = alpha_p
        if CyclicSchedule is None:
            self.sched_type, self.sched_halflife = "constant", 1
        else:
            self.sched_type, self.sched_halflife = CyclicSchedule
        self.alpha_p_schedule = cyc_learning_spd(self.sched_type, self.init_spd, self.sched_halflife)

        self.report_labels = ['Perf/Recent Rewards',
                              'Losses/Policy LLs', 'Losses/Policy Entropies',
                              'Values/Critic Scores', 'Values/Mean Q Scores']
        self.last_good_model = recover

        self.lnPi_ts = {}
        self.AdvlnPi_ts = {}
        self.delta_Advs_t = {}

        self.plosses = {}
        self.vlosses = {}
        self.p_trainers = {}
        self.v_trainers = {}

        self.entropies = {}
        self.policy_LLs = {}

        with tf.variable_scope(self.name):
            self.alpha_p_sched_t = tf.placeholder(
                shape=None, dtype=tf.float32,
                name='cyc_alpha_rate_p'
            )
            with tf.variable_scope('Qglobal'):
                self.Q_central = QNN(  # Qsa-function
                    self.sense_z,  # all_observations
                    self.actions_Ats  # all_actions
                )  # Q(obs_ts, a_ts)
                self.Q_target = tf.placeholder(
                    shape=[None, self.actor_count],
                    dtype=tf.float32, name='Q_targets'
                )  # Q-estimates for future states
            self.Q_loss = 0.5 * tf.reduce_sum(tf.square(self.Q_central - self.Q_target))
            self.Q_trainer = self.optimizer_q.minimize(
                loss=alpha_q * (
                        self.Q_loss +
                        tf.random_normal(shape=tf.shape(self.Q_loss), mean=0., stddev=_ent_decay_)
                ),
                var_list=[v for v in tf.trainable_variables()
                          if ((name in v.name) and ("Qglobal" in v.name)) or "sensorium" in v.name]
            )
            for name in self.actor_names:  # names per agent; need better vectorization...
                # AC-specific placeholder: holds value-NN-informed Advantages
                self.delta_Advs_t[name] = tf.placeholder(
                    shape=[None, 1], dtype=tf.float32,
                    name='delta_Advantages'
                )  # holds: TD_error aka. delta_t
                # Intermediate variables
                self.lnPi_ts[name] = -tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.a_logits[name],  # a_t|s_t: bernoulli_LL
                    labels=self.actions_Ats[name]
                )
                self.entropies[name] = tf.clip_by_value(
                    bernoulli_H(self.a_probs[name]),
                    _eps_ / self.env.nagents, 100.
                )
                # Advantage-weighted LLs
                self.AdvlnPi_ts[name] = tf.multiply(
                    self.delta_Advs_t[name],
                    self.lnPi_ts[name]
                )
                # Agent-specific Losses
                self.plosses[name] = tf.reduce_sum(self.AdvlnPi_ts[name])
                self.vlosses[name] = tf.reduce_sum(
                    self.delta_Advs_t[name] * self.values[name]
                )  # using semi-value loss version

                # Training ops
                self.p_trainers[name] = self.optimizers_p[name].minimize(
                    loss=(- self.alpha_p_sched_t) * (self.plosses[name] +
                                                     _ent_decay_ * self.entropies[name]),
                    var_list=[v for v in tf.trainable_variables()
                              if ((name in v.name) and ("actor" in v.name)) or "sensorium" in v.name]
                )
                self.v_trainers[name] = self.optimizers_v[name].minimize(
                    loss=(-alpha_v) * self.vlosses[name],
                    var_list=[v for v in tf.trainable_variables()
                              if ((name in v.name) and ("critic" in v.name)) or "sensorium" in v.name]
                )
        # Aggregate Learning Stats
        self.policy_LLs = tf.stack(list(self.lnPi_ts.values()), axis=1)
        self.crits = tf.stack(list(self.values.values()), axis=1)
        self.entropy = tf.stack(list(self.entropies.values()), axis=1)
        self.Qmeans = tf.reduce_mean(self.Q_central, axis=0)  # Q-stats per agent
        self.summLLs = summarize(self.policy_LLs)
        self.summValues = summarize(self.crits)
        self.summEntropy = summarize(self.entropy)
        self.summQs = summarize(self.Qmeans)
        return

    def act(self, state, sess):
        """Returns vector of p-net sample action (in {0,1})"""
        assert self.actor_count in state.shape
        ind = 0
        probs = {}
        a_ts = {}
        for name in self.actor_names:
            st = state[ind]
            ind += 1
            probs[name] = sess.run(
                self.a_probs[name],
                {self.states_St[name]: np.expand_dims(st.flatten(), axis=0)}
            ).squeeze()
            a_ts[name] = 1 * (np.random.rand() < probs[name])  # scalar -> vector comparison
        return np.array(list(a_ts.values())).squeeze()

    def generate_summary(self, sess, act_dicts):
        # 'Perf/Recent Rewards',
        # 'Losses/Policy LLs', 'Losses/Policy Entropies'
        # 'Values/Critic Scores', 'Values/Mean Q Scores'
        return sess.run(
            [self.summLLs, self.summEntropy,
             self.summValues, self.summQs],
            feed_dict=act_dicts
        )

    def train(self, sess, gamma=0.99, bootstrap_value=0.0, upd_list=None):
        '''Order of Operations:
            1. - Check and Parse Rollout Buffer
            2. - Prep Population Index
            3. - Collate Full Popn (s,a)-info dict
            4. - Policy Eval + Update: Q_central
                    - get Q_i_t
                    - calc TD target
                    - exec Q_trainer
            5. - Policy Updates per agent: (π_i, v_i)
            6. - Reporting Functions...
        '''
        eplen = checkRolloutDict(self.episode_buffer)
        states = np.stack(self.episode_buffer['states'])
        actions = np.stack(self.episode_buffer['actions'])
        rewards = np.stack(self.episode_buffer['rewards'])
        next_states = np.stack(self.episode_buffer['next_states'])
        if _DEBUG_:
            print(eplen, " epochs in current episode")
            print(states)
            input("About to start training call...")
        learners = range(self.actor_count)  # if (upd_list is None) else upd_list
        Q_targs = self.train_eval_QC(sess, gamma=gamma, bootstrap_value=bootstrap_value)

        for agent_idx in learners:
            Qi_ts = Q_targs[:, agent_idx, ...]
            self.train_single(sess, agent_index=agent_idx,
                              rollout=(states[:, agent_idx, ...],
                                       actions[:, agent_idx, ...],
                                       rewards[:, agent_idx, ...],
                                       next_states[:, agent_idx, ...]),
                              Qsa_i=Qi_ts,
                              gamma=gamma, bootstrap_value=bootstrap_value)
        # Generate learning model statistics to periodically save
        s_a_dict = {
            **{self.states_St[self.actor_names[idx]]: np.vstack(states[:, idx, ...])
               for idx in learners},
            **{self.actions_Ats[self.actor_names[idx]]: np.vstack(actions[:, idx, ...])
               for idx in learners}
        }
        reports = self.generate_summary(sess, s_a_dict)
        return reports

    def train_single(self, sess, agent_index, rollout, Qsa_i,
                     gamma=0.95, bootstrap_value=0.0):
        # Rollout Structure: [S0, A, R, S1]
        states = np.vstack(rollout[0].squeeze())
        returns = rollout[2].ravel()

        # generate state-value estimates; goes into delta_t
        # TD(1) target for v-net train: G_t + gamma*v(s_{t+1})
        advantages = np.squeeze(calc_advantages(returns, vals=Qsa_i,
                                                gamma=gamma, bootstrap_value=bootstrap_value))
        feed_dict = {
            self.states_St[self.actor_names[agent_index]]: states,
            self.actions_Ats[self.actor_names[agent_index]]: np.vstack(rollout[1].squeeze()),
            self.returns_Gts[self.actor_names[agent_index]]: np.vstack(returns),
            self.states_St_prime[self.actor_names[agent_index]]: np.vstack(rollout[3].squeeze()),
            self.delta_Advs_t[self.actor_names[agent_index]]: np.vstack(advantages),
            self.alpha_p_sched_t: self.alpha_p_schedule[self.total_epoch_count % _sched_win_]
        }
        sess.run([
            self.p_trainers[self.actor_names[agent_index]],
            self.v_trainers[self.actor_names[agent_index]]
        ],
            feed_dict=feed_dict
        )
        return

    def train_eval_QC(self, sess, gamma=0.95, bootstrap_value=0.0):
        eplen = checkRolloutDict(self.episode_buffer)
        states = np.stack(self.episode_buffer['states'])
        actions = np.stack(self.episode_buffer['actions'])
        rewards = np.stack(self.episode_buffer['rewards'])
        if _DEBUG_:
            print(eplen, " epochs in current episode")
            input("About to start training call...")
        learners = range(self.actor_count)

        # Need *all* (s,a) popn info for global/central Q-critic eval
        act_dicts = {}
        act_dicts.update(
            {self.states_St[self.actor_names[idx]]: np.vstack(states[:, idx, ...])
             for idx in learners})
        act_dicts.update(
            {self.actions_Ats[self.actor_names[idx]]: np.vstack(actions[:, idx, ...])
             for idx in learners})

        # Q estimates for all agents, using all popn info; check data structure...
        Q_ts = sess.run(self.Q_central, feed_dict=act_dicts)
        Q_targs = np.squeeze(calc_Q_TD_target(
            rewards, Q_ts, gamma=gamma, bootstrap_value=bootstrap_value))
        act_dicts.update({self.Q_target: Q_targs})
        # Update & Eval central Q-function
        sess.run(self.Q_trainer, feed_dict=act_dicts)
        Q_ts = sess.run(self.Q_central, feed_dict=act_dicts)

        ## Train individual value functions too
        for agent_idx in learners:
            # generate state-value estimates; goes into delta_t
            returns = (rewards[:, agent_idx, ...]).ravel()
            vals = sess.run(
                self.values[self.actor_names[agent_idx]],
                feed_dict={
                    self.states_St[self.actor_names[agent_idx]]: np.vstack(states[:, agent_idx, ...])
                }
            ).ravel()
            advantages = np.squeeze(calc_advantages(np.vstack(returns), vals, gamma=0.95))
            feed_dict = {
                self.states_St[self.actor_names[agent_idx]]: np.vstack(states[:, agent_idx, ...]),
                self.returns_Gts[self.actor_names[agent_idx]]: np.vstack(returns),
                self.delta_Advs_t[self.actor_names[agent_idx]]: np.vstack(advantages)
            }
            # Train & Record v_loss
            sess.run(self.v_trainers[self.actor_names[agent_idx]], feed_dict=feed_dict)
            _ = sess.run(self.v_trainers[self.actor_names[agent_idx]], feed_dict=feed_dict)
        return Q_ts  # np.squeeze(Q_ts)
