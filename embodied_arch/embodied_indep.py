'''
init: May2019
Author: OO
Goals: RF+AC for indep multi-agents
Still in development:
    - base class now encapsulated off in EmbodiedPopulation.py
    - Next: CycLR + WoLF
Design Logic:
    - All agents interact w/ a common env => single menv
    - The shared env is a property of the agent popn => menv is a param
    - Independent state reports for each agent now
        - formerly: ~~All agents see the same full state info. => S_t is shared~~
    - All RL agents act independently => separate indeps Actor & Value NNs
    - No explicit collusion or shared learning => separate sensoria NNs
'''

from embodied_arch.EmbodiedPopulation import EmbodiedAgent_Population
from embodied_arch.embodied_misc import *
from embodied_arch.embodied_misc import _zdim_, bernoulli_H, summarize, _sched_win_
from embodied_arch.misc_helpers import discount
from minoritygame.minority_multienv import MinorityGame_Multiagent_env

sys.path.append('.')

# class: var defaults
tboard_path = "./log"
agent_name = "embodied_agent_IRL"
__version__ = "0.0.3"
_DEBUG_ = False

# model: var defaults
_default_reports_ = ['Perf/Recent Reward',
                     'Losses/Policy LL', 'Losses/Entropy',
                     'Norms/Grad Norm', 'Norms/Var Norm']
_every_ = 100  # 500
_eps_ = 1.e-2  # 1.e-5
_ent_decay_ = 5e-2
(lrp, lrv) = (1e-2, 5e-2)  # learning rates
_max_len_ = 400
# default_sense_hSeq = (32,)

# envs: var defaults
(na_, m_, s_, p_) = (33, 3, 4, 0.5)
_s_size_, _a_size_ = (4, 1)


class EmbodiedAgent_IRF(EmbodiedAgent_Population):  # instead of EmbodiedAgent_Independent
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
        self.rf_trainers = {}
        self.optimizers = {name_i: tf.train.AdamOptimizer(learning_rate=lrp) for name_i in self.actor_names}
        with tf.variable_scope(self.name):
            for name in self.actor_names:  # need to do better vectorization at some point...
                # Intermediate variables
                self.lnPi_ts[name] = -tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.a_logits[name],  ## a_t|s_t
                    labels=self.actions_Ats[name]  ## use tf.one_hot for m-ary action spaces
                )  # bernoulli_LL(self.actions_Ats[name], self.a_probs[name])
                self.entropies[name] = tf.clip_by_value(
                    bernoulli_H(self.a_probs[name]),
                    _eps_ / self.env.nagents, 100.
                )
                self.GlnPi_ts[name] = tf.multiply(self.returns_Gts[name], self.lnPi_ts[name])

                # Losses
                self.rflosses[name] = tf.reduce_mean(tf.reduce_sum(self.GlnPi_ts[name]))
                # Training ops
                self.rf_trainers[name] = self.optimizers[name].minimize(
                    loss=(- alpha) * (self.rflosses[name] + _ent_decay_ * self.entropies[name]),
                    var_list=[v for v in tf.trainable_variables() if name in v.name]
                )  # probably need to separate A-vs-C gradients later...
            self.policy_LLs = tf.stack(list(self.lnPi_ts.values()), axis=1)
            self.entropy = tf.stack(list(self.entropies.values()), axis=1)
            self.summLLs = summarize(self.policy_LLs)
            self.summEntropy = summarize(self.entropy)
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
        return sess.run(
            [self.summLLs, self.summEntropy],
            feed_dict=act_dicts
        )

    def train_single(self, sess, agent_index, rollout, Qsa_i=None, gamma=0.95, bootstrap_value=0.0):
        # self.episode_buffer.append([s, acts_lst.squeeze(), r_lst, s1])
        discounted_returns = discount(np.hstack([rollout[2].ravel(), [bootstrap_value]]), gamma)
        discounted_returns = discounted_returns[:-1, None]
        feed_dict = {
            self.states_St[self.actor_names[agent_index]]: np.vstack(rollout[0].squeeze()),
            self.actions_Ats[self.actor_names[agent_index]]: np.vstack(rollout[1].squeeze()),
            self.returns_Gts[self.actor_names[agent_index]]: np.vstack(discounted_returns),
            self.states_St_prime[self.actor_names[agent_index]]: np.vstack(rollout[3].squeeze())
        }
        sess.run(
            self.rf_trainers[self.actor_names[agent_index]],
            feed_dict=feed_dict
        )
        return


class EmbodiedAgent_IRFB(EmbodiedAgent_Population):  # instead of EmbodiedAgent_Independent
    def __init__(self, name=agent_name,
                 env_=MinorityGame_Multiagent_env,
                 alpha_p=5.e-2, alpha_v=1e-1,
                 latentDim=_zdim_,
                 space_size=(_s_size_, _a_size_),
                 sensorium=SensoriumNetworkTemplate,
                 actorNN=ActionPolicyNetwork,
                 valueNN=ValueNetwork,
                 recover=None,
                 _every_=_every_,
                 max_episode_length=_max_len_
                 ):
        super().__init__(name=name, env_=env_,
                         latentDim=latentDim, space_size=space_size,
                         sensorium=sensorium, actorNN=actorNN, valueNN=valueNN,
                         _every_=_every_, max_episode_length=max_episode_length
                         )
        self.report_labels = ['Perf/Recent Rewards', 'Losses/Policy LLs',
                              'Losses/Critic Scores', 'Losses/Policy Entropies']
        self.last_good_model = recover

        self.lnPi_ts = {}
        self.GlnPi_ts = {}
        self.Advs_ts = {}
        self.AdvlnPi_ts = {}

        self.plosses = {}
        self.vlosses = {}
        self.p_trainers = {}
        self.v_trainers = {}

        self.entropies = {}
        self.policy_LLs = {}

        self.optimizers_p = {name_i: tf.train.AdamOptimizer(
            learning_rate=lrp) for name_i in self.actor_names}
        self.optimizers_v = {name_i: tf.train.AdamOptimizer(
            learning_rate=lrv) for name_i in self.actor_names}
        with tf.variable_scope(self.name):
            for name in self.actor_names:  # need to do better vectorization at some point...
                # Intermediate variables
                self.lnPi_ts[name] = -tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.a_logits[name],  ## a_t|s_t
                    labels=self.actions_Ats[name]  ## use tf.one_hot for m-ary action spaces
                )  # bernoulli_LL(self.actions_Ats[name], self.a_probs[name])
                self.entropies[name] = tf.clip_by_value(
                    bernoulli_H(self.a_probs[name]),
                    _eps_, 100.
                )
                # using value-baselined returns instead of raw G_t
                self.Advs_ts[name] = self.returns_Gts[name] - self.values[name]
                self.AdvlnPi_ts[name] = tf.multiply(
                    tf.stop_gradient(self.Advs_ts[name]),
                    self.lnPi_ts[name]
                )  # AdvlnP
                # Losses
                self.plosses[name] = tf.reduce_sum(self.AdvlnPi_ts[name])
                self.vlosses[name] = 0.5 * tf.reduce_sum(
                    tf.square(self.returns_Gts[name] - tf.reshape(self.values[name], [-1]))
                )
                # Training ops
                self.p_trainers[name] = self.optimizers_p[name].minimize(
                    loss=(- alpha_p) * (self.plosses[name] + _ent_decay_ * self.entropies[name]),
                    var_list=[v for v in tf.trainable_variables()
                              if ((name in v.name) and ("actor" in v.name)) or "sensorium" in v.name]
                )
                self.v_trainers[name] = self.optimizers_v[name].minimize(
                    loss=alpha_v * self.vlosses[name],
                    var_list=[v for v in tf.trainable_variables()
                              if ((name in v.name) and ("critic" in v.name)) or "sensorium" in v.name]
                )
            # Aggregate Learning Stats
            self.policy_LLs = tf.stack(list(self.lnPi_ts.values()), axis=1)
            self.crits = tf.stack(list(self.values.values()), axis=1)
            self.entropy = tf.stack(list(self.entropies.values()), axis=1)
            self.summLLs = summarize(self.policy_LLs)
            self.summValues = summarize(self.crits)
            self.summEntropy = summarize(self.entropy)
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
        # 'Perf/Recent Rewards', 'Losses/Policy LLs','Losses/Mean Value Fxn', 'Losses/Policy Entropies']
        return sess.run(
            [self.summLLs, self.summValues, self.summEntropy],
            feed_dict=act_dicts
        )

    def train_single(self, sess, agent_index, rollout, Qsa_i=None, gamma=0.95, bootstrap_value=0.0):
        # self.episode_buffer.append([s, acts_lst.squeeze(), r_lst, s1])
        discounted_returns = discount(np.hstack([rollout[2].ravel(), [bootstrap_value]]), gamma)
        discounted_returns = discounted_returns[:-1, None]
        feed_dict = {
            self.states_St[self.actor_names[agent_index]]: np.vstack(rollout[0].squeeze()),
            self.actions_Ats[self.actor_names[agent_index]]: np.vstack(rollout[1].squeeze()),
            self.returns_Gts[self.actor_names[agent_index]]: np.vstack(discounted_returns),
            self.states_St_prime[self.actor_names[agent_index]]: np.vstack(rollout[3].squeeze())
        }
        sess.run([
            self.p_trainers[self.actor_names[agent_index]],
            self.v_trainers[self.actor_names[agent_index]]
        ],
            feed_dict=feed_dict
        )
        return


class EmbodiedAgent_IAC(EmbodiedAgent_Population):
    def __init__(self, name=agent_name,
                 env_=MinorityGame_Multiagent_env,
                 alpha_p=5.e-2, alpha_v=1e-1,
                 latentDim=_zdim_,
                 space_size=(_s_size_, _a_size_),
                 sensorium=SensoriumNetworkTemplate,
                 actorNN=ActionPolicyNetwork,
                 valueNN=ValueNetwork,
                 recover=None,
                 _every_=_every_,
                 max_episode_length=_max_len_,
                 CyclicSchedule=None
                 ):
        super().__init__(name=name, env_=env_,
                         latentDim=latentDim, space_size=space_size,
                         sensorium=sensorium, actorNN=actorNN, valueNN=valueNN,
                         _every_=_every_, max_episode_length=max_episode_length
                         )
        self.report_labels = ['Perf/Recent Rewards', 'Losses/Policy LLs',
                              'Losses/Critic Scores', 'Losses/Policy Entropies']
        self.last_good_model = recover

        self.lnPi_ts = {}
        self.GlnPi_ts = {}
        self.Advs_ts = {}
        self.AdvlnPi_ts = {}
        # self.delta_Advs_t = {}

        self.plosses = {}
        self.vlosses = {}
        self.p_trainers = {}
        self.v_trainers = {}

        self.entropies = {}
        self.policy_LLs = {}

        self.optimizers_p = {name_i: tf.train.AdamOptimizer(
            learning_rate=lrp) for name_i in self.actor_names}
        self.optimizers_v = {name_i: tf.train.AdamOptimizer(
            learning_rate=lrv) for name_i in self.actor_names}

        ## Setup Cyclic Learning Rate Schedule
        self.init_spd = alpha_p
        if CyclicSchedule is None:
            self.sched_type, self.sched_halflife = "constant", 1
        else:
            self.sched_type, self.sched_halflife = CyclicSchedule
        self.alpha_p_schedule = cyc_learning_spd(self.sched_type, self.init_spd, self.sched_halflife)

        with tf.variable_scope(self.name):
            self.alpha_p_sched_t = tf.placeholder(
                shape=None, dtype=tf.float32,
                name='cyc_alpha_rate_p'
            )
            for name in self.actor_names:  # need to do better vectorization at some point...
                # Intermediate variables
                self.lnPi_ts[name] = -tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.a_logits[name],  ## a_t|s_t
                    labels=self.actions_Ats[name]  ## use tf.one_hot for m-ary action spaces
                )  # bernoulli_LL(self.actions_Ats[name], self.a_probs[name])
                self.entropies[name] = tf.clip_by_value(
                    bernoulli_H(self.a_probs[name]),
                    _eps_, 10.
                )
                # using value-baselined returns instead of raw G_t
                self.Advs_ts[name] = self.returns_Gts[name] - self.values[name]
                self.AdvlnPi_ts[name] = tf.multiply(
                    tf.stop_gradient(self.Advs_ts[name]),
                    self.lnPi_ts[name]
                )  # AdvlnP
                # Losses
                self.plosses[name] = tf.reduce_mean(self.AdvlnPi_ts[name])
                self.vlosses[name] = 0.5 * tf.reduce_mean(
                    tf.square(self.returns_Gts[name] - self.values[name])
                )  # squared TD error loss version
                # tf.reduce_sum(self.delta_Advs_t[name] * self.values[name])  # semi-value loss version
                # Training ops
                self.p_trainers[name] = self.optimizers_p[name].minimize(
                    loss=(- self.alpha_p_sched_t) * (self.plosses[name] + _ent_decay_ * self.entropies[name]),
                    var_list=[v for v in tf.trainable_variables()
                              if ((name in v.name) and ("actor" in v.name)) or "sensorium" in v.name]
                )
                self.v_trainers[name] = self.optimizers_v[name].minimize(
                    loss=alpha_v * self.vlosses[name],
                    var_list=[v for v in tf.trainable_variables()
                              if ((name in v.name) and ("critic" in v.name)) or "sensorium" in v.name]
                )
            # Aggregate Learning Stats
            self.policy_LLs = tf.stack(list(self.lnPi_ts.values()), axis=1)
            self.crits = tf.stack(list(self.values.values()), axis=1)
            self.entropy = tf.stack(list(self.entropies.values()), axis=1)
            self.summLLs = summarize(self.policy_LLs)
            self.summValues = summarize(self.crits)
            self.summEntropy = summarize(self.entropy)
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
        # 'Perf/Recent Rewards', 'Losses/Policy LLs','Losses/Mean Value Fxn', 'Losses/Policy Entropies']
        return sess.run(
            [self.summLLs, self.summValues, self.summEntropy],
            feed_dict=act_dicts
        )

    def train_single(self, sess, agent_index, rollout, Qsa_i=None, gamma=0.95, bootstrap_value=0.0):
        # Rollout Structure: [S0, A, R, S1]
        states = np.vstack(rollout[0].squeeze())
        rewards = rollout[2].ravel()
        vals = sess.run(
            self.values[self.actor_names[agent_index]],
            feed_dict={
                self.states_St[self.actor_names[agent_index]]: states
            }
        ).ravel()  # v(s)

        # generate TD(1) target of state value: G_t + gamma*v(s_{t+1})
        Gt_TD = np.squeeze(
            calc_V_TD_target(rewards, vals=vals,
                             gamma=gamma, bootstrap_value=bootstrap_value)
        )  # discounted total returns after t based on current v-net

        feed_dict = {
            self.states_St[self.actor_names[agent_index]]: states,
            self.actions_Ats[self.actor_names[agent_index]]: np.vstack(rollout[1].squeeze()),
            self.returns_Gts[self.actor_names[agent_index]]: np.vstack(Gt_TD),
            self.alpha_p_sched_t: self.alpha_p_schedule[self.total_epoch_count % _sched_win_]
        }
        sess.run([
            self.p_trainers[self.actor_names[agent_index]],
            self.v_trainers[self.actor_names[agent_index]]
        ],
            feed_dict=feed_dict
        )
        return

    def pretrainCritics(self, sess):
        assert all(np.diff([len(buf) for _, buf in self.episode_buffer.items()]) == 0), \
            "Rollout is not the correct shape"
        states = np.stack(self.episode_buffer['states'])
        rewards = np.stack(self.episode_buffer['rewards'])

        learners = range(self.actor_count)
        vls = np.zeros_like(learners, dtype=float)
        for agent_idx in learners:
            rwds = rewards[:, agent_idx, ...].ravel()
            vals = sess.run(
                self.values[self.actor_names[agent_idx]],
                feed_dict={
                    self.states_St[self.actor_names[agent_idx]]: (states[:, agent_idx, ...])
                }).ravel()  # v(s)
            # generate TD(1) target of state value: G_t + gamma*v(s_{t+1})
            Gt_TD = np.squeeze(calc_V_TD_target(rwds, vals=vals))
            feed_dict = {
                self.states_St[self.actor_names[agent_idx]]: (states[:, agent_idx, ...]),
                self.returns_Gts[self.actor_names[agent_idx]]: np.vstack(Gt_TD)
            }
            sess.run(self.v_trainers[self.actor_names[agent_idx]], feed_dict=feed_dict)
            vls[agent_idx] = sess.run(self.vlosses[self.actor_names[agent_idx]], feed_dict=feed_dict)
        return vls
