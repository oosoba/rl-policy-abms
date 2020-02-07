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
    - Add placeholder var for learning rate; feed lr var with cycling values
Notes:
    - Modeling action net with tf_probab distribution is valuable for entropy calc in future non-Bernoulli actors
'''

from abc import abstractmethod

import gym

from embodied_arch.embodied_misc import *
from embodied_arch.embodied_misc import _zdim_, _sched_win_, cyc_learning_spd
from embodied_arch.embodied_misc import bernoulli_H
from embodied_arch.misc_helpers import discount

# from embodied_arch.embodied_misc import *
# from embodied_arch.embodied_misc import _zdim_
# from embodied_arch.misc_helpers import discount

sys.path.append('.')

# class: var defaults
__version__ = "0.0.5"
_DEBUG_ = False
_salted_suffix_ = '_' + ('{}'.format(np.random.rand()))[2:6]
agent_name = "embodied_agent"  # gets salted in init
tboard_path = "./log"


# model: var defaults
_default_reports_ = ['Perf/Recent Reward',
                     'Losses/Policy LL', 'Losses/Entropy',
                     'Norms/Grad Norm', 'Norms/Var Norm']

_every_ = 50
_ent_decay_ = 5e-3
(lrp, lrv) = (1e-2, 5e-2)  # learning rates
_max_len_ = 400

# envs: var defaults
(na_, m_, s_, p_) = (33, 3, 4, 0.5)
_s_size_, _a_size_ = (4, 1)
_eps_ = .0001


class EmbodiedAgent(object):
    """Base class for embodied agents. Implements:
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
    Caveat:
    This is a base class that needs to keep the same member function signature for every env.
    Do not change the signature of base member functions.
    If you need modified functions, subclass and override as needed.
    """

    def __init__(self, name=agent_name,
                 env_=gym.make('CartPole-v0'),
                 latentDim=_zdim_,
                 space_size=(_s_size_, _a_size_),
                 sensorium=SensoriumNetworkTemplate,
                 actorNN=ActionPolicyNetwork,
                 valueNN=ValueNetwork,
                 _every_=_every_,
                 max_episode_length=_max_len_
                 ):
        self.name = name + _salted_suffix_
        self.model_path = tboard_path + "/train_" + str(self.name)
        self.autosave_every = _every_
        self.max_episode_length = max_episode_length
        self.last_good_model = None
        self.report_labels = ['Perf/Recent Reward', 'Losses/Policy LL', 'Losses/Entropy']

        # monitors
        self.episode_buffer = list()
        self.episode_buffer = {
            'states': list(),
            'actions': list(),
            'rewards': list(),
            'next_states': list()
        }
        self.last_total_return = None
        self.total_epoch_count = 0  # num of training seasons agents have gone through
        self.agent_age = 0  # num of play seasons agents have gone through
        self.summary_writer = tf.summary.FileWriter(self.model_path)

        ## Collect Characteristics of the Env
        self.env = env_  # env setup
        if "state_space_size" in dir(self.env) and "action_space_size" in dir(self.env):
            self.s_size, self.a_size = (self.env.state_space_size, self.env.action_space_size)
            # s_size: state space dim = num_memory bits
            # ideally inferrable from the env spec
        else:
            self.s_size, self.a_size = space_size
        self.action_space_option_count = 2  ## use tf.one_hot encoding for (m>2)-ary action spaces

        with tf.variable_scope(self.name):
            # Setup: env. signals
            self.states_St = tf.placeholder(
                shape=[None, self.s_size], dtype=tf.float32, name='states')
            self.actions_At = tf.placeholder(
                shape=[None, self.a_size], dtype=tf.float32, name='actions-taken')
            self.returns_Gt = tf.placeholder(
                shape=[None, 1], dtype=tf.float32, name='returns_discounted')
            self.states_St_prime = tf.placeholder(
                shape=[None, self.s_size], dtype=tf.float32, name='states_prime')

            # need to modularize what follows to allow for general distribution type (i.e. non-Bernoulli)
            with tf.variable_scope('sensorium'):  # as sense_scope:
                self.sense_z = sensorium(self.states_St, out_dim=latentDim)
            with tf.variable_scope('action'):  # \pi(s,a)
                self.a_logit = actorNN(self.sense_z)
                self.a_prob = tf.nn.sigmoid(self.a_logit)

            with tf.variable_scope('value_fxn'):
                self.value = valueNN(self.sense_z)
        return

    @abstractmethod
    def act(self, state, sess):
        """Default action assumes binary action space.
        Returns completely random action (in {0,1})."""
        return np.squeeze(np.random.randint(low=0, high=2))

    @abstractmethod
    def train(self, sess, gamma=0.95, bootstrap_value=0.0, window=None):
        pass

    def reset_buffer(self):
        """Clears episode buffer."""
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

    def episode_length(self):
        return len(self.episode_buffer['states'])

    def play(self, sess, terminal_reward=-10.0):
        '''Generates and populates a complete buffer of experience tuples from the attached environment.
        Format is generally: (s_t, a_t, r_t, s_{t+1})

        Caveat: This is a base function that needs to keep the same signature for every env.
        Do not change the signature of play & reset functions.
        If you need modified functions, subclass and override as needed.
        '''
        self.reset_buffer()
        self.last_total_return = 0.
        self.agent_age += 1
        d = False
        s = self.env.reset()  # doozy of a bug: this used to scrub and resample all brains/strats
        # generate a full episode rollout (self.brain.episode_buffer)
        while (self.episode_length() < self.max_episode_length) and not d:
            act_pn = self.act(s, sess)
            s1, r, d, *rest = self.env.step(act_pn)  # get next state, reward, & done flag
            if d:
                r = terminal_reward  # reinforcement for early termination, default: -ve
            # self.episode_buffer.append([s, act_pn.squeeze(), float(r), s1])
            self.episode_buffer['states'].append(s)
            self.episode_buffer['actions'].append(act_pn.squeeze())
            self.episode_buffer['rewards'].append(float(r))
            self.episode_buffer['next_states'].append(s1)
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

    def work(self, sess, saver, num_epochs, gamma=0.95, window=None):
        totals = []
        print("Starting agent " + str(self.name))
        with sess.as_default(), sess.graph.as_default():
            for tk in range(int(num_epochs)):
                print("\rEpoch no.: {}/{}".format(tk, num_epochs), end="")
                self.reset_buffer()
                self.total_epoch_count += 1
                while self.episode_length() == 0:
                    self.play(sess)
                # TRAIN: Update the network using episodes in the buffer.
                stats = list(self.train(sess, gamma, window=window))
                # save summary statistics for tensorboard
                stats.insert(0, self.last_total_return)
                totals.append(stats[0])
                if any(np.isnan(np.array(stats).ravel())):
                    print('Model issues @Step {}. Stats({}): ( {} )'.format(
                        tk, self.report_labels, stats))
                    if self.last_good_model is not None:
                        print("Attempting Model Restore...")
                        saver.restore(sess, self.last_good_model)
                        print("\nRestored Last Good Model: ", self.last_good_model)
                else:  # save model parameters periodically
                    self.model_summary(sess, tk, stats, saver, labels_=self.report_labels)
        return totals

    def model_summary(self, sess, tk, stats_,
                      saver, labels_=None):
        if labels_ is None:
            labels_ = _default_reports_
        summary = tf.Summary()
        for k in range(min(len(stats_), len(labels_))):
            summary.value.add(tag=labels_[k], simple_value=float(stats_[k]))
        self.summary_writer.add_summary(summary, tk)
        self.summary_writer.flush()

        # to save or not to save...
        if tk % self.autosave_every == 0: 
            print('\nStep {}: Stats({}): ( {} )'.format(tk, labels_, stats_))
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


class EmbodiedAgentRF(EmbodiedAgent):
    def __init__(self, name=agent_name,
                 env_=gym.make('CartPole-v0'),
                 alpha=1e-1, lrp=lrp,
                 latentDim=_zdim_,
                 space_size=(_s_size_, _a_size_),
                 sensorium=SensoriumNetworkTemplate,
                 actorNN=ActionPolicyNetwork,
                 recover=None,
                 _every_=_every_,
                 max_episode_length=_max_len_,
                 CyclicSchedule=None
                 ):
        super().__init__(name=name, env_=env_,
                         latentDim=latentDim, space_size=space_size,
                         sensorium=sensorium, actorNN=actorNN,
                         _every_=_every_, max_episode_length=max_episode_length
                         )
        self.report_labels = ['Perf/Recent Reward', 'Losses/Policy LL', 'Losses/Entropy']
        self.last_good_model = recover

        ## Learning Rate mechanisms
        ### Basic optimizer rate set:
        self.learning_rate_p = lrp
        self.optimizer_p = tf.train.AdamOptimizer(learning_rate=self.learning_rate_p, beta1=.2, beta2=.3, name="adam")
        # self.optimizer_p = tf.train.AdagradOptimizer(learning_rate=self.learning_rate_p, name="adagrad")
        # self.optimizer_p = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_p, name="sgd")

        ### Mechanism for variable learning speeds via alpha:
        self.init_spd = alpha
        if CyclicSchedule is None:
            self.sched_type, self.sched_halflife = "constant", 1
        else:
            self.sched_type, self.sched_halflife = CyclicSchedule
        self.alpha_p_schedule = cyc_learning_spd(self.sched_type, self.init_spd, self.sched_halflife)

        with tf.variable_scope(self.name):  # slim stacks take care of regularization
            # 1st: setup: variable learning rate
            self.alpha_p_sched_t = tf.placeholder(
                shape=None, dtype=tf.float32, name='cyc_alpha_rate_p')
            # 2nd: RF/MC-PG Intermediate variables
            # Evaluate ln π(a_t|s_t) for bernoulli action space output
            # Taking advantage of built-in cross_entropy fxn (w/ numeric guards).
            self.lnPi_t = -tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.a_logit,  ## a_t|s_t
                labels=self.actions_At  ## use tf.one_hot for m-ary action spaces
            )  # bernoulli_LL(self.actions_At, self.a_prob)
            self.entropy = tf.clip_by_value(
                tf.reduce_mean(bernoulli_H(self.a_prob)),
                _eps_, 100.
            )
            self.GlnPi_t = self.returns_Gt * self.lnPi_t  # 'GlnP'
            self.policy_LL = tf.reduce_mean(self.lnPi_t)

            # Losses and Gradients
            self.rfobj = tf.reduce_mean(self.GlnPi_t)
            self.policy_vars = [v for v in tf.trainable_variables()
                                if "action" in v.name or "sensorium" in v.name]
            self.rf_grads = self.optimizer_p.compute_gradients(
                loss=(-self.alpha_p_sched_t * self.rfobj),
                var_list=self.policy_vars
            )
            # Create training operations
            self.rf_train = self.optimizer_p.apply_gradients(self.rf_grads)
            # self.optimizer_p.minimize(-alpha * self.rfobj, var_list=self.policy_vars)
        return

    def act(self, state, sess):
        """Returns policy net sample action (in {0,1})"""
        probs = sess.run(self.a_prob, {self.states_St: np.expand_dims(state.flatten(), axis=0)})
        a_t = 1 * (np.random.rand() < probs)
        return a_t.squeeze()

    def train(self, sess, gamma=0.95, bootstrap_value=0.0, window=None):
        # parse buffer
        states = np.stack(self.episode_buffer['states'])
        actions = np.stack(self.episode_buffer['actions'])
        rewards = np.stack(self.episode_buffer['rewards'])
        next_states = np.stack(self.episode_buffer['next_states'])

        # generate discounted returns; goes into G_t
        discounted_returns = discount(np.hstack([rewards, [bootstrap_value]]), gamma=gamma, window=window)
        discounted_returns = discounted_returns[:-1, None]
        if _DEBUG_:
            # print(self.episode_buffer)
            print(states)
            # print( list(map(lambda x: np.shape(x), [states, actions, discounted_rewards])) )
            input("About to start training call...")
        feed_dict = {
            self.alpha_p_sched_t: self.alpha_p_schedule[self.total_epoch_count % _sched_win_],
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
        return p_ll, p_ent  # return


class EmbodiedAgentRFBaselined(EmbodiedAgent):
    def __init__(self, name=agent_name,
                 env_=gym.make('CartPole-v0'),
                 alpha_p=5e-3, alpha_v=1e-3,
                 lrp=lrp, lrv=lrv,
                 latentDim=_zdim_,
                 space_size=(_s_size_, _a_size_),
                 sensorium=SensoriumNetworkTemplate,
                 actorNN=ActionPolicyNetwork,
                 valueNN=ValueNetwork,
                 recover=None,
                 max_episode_length=_max_len_,
                 CyclicSchedule=None
                 ):
        super().__init__(name=name, env_=env_,
                         latentDim=latentDim, space_size=space_size,
                         sensorium=sensorium, actorNN=actorNN, valueNN=valueNN,
                         max_episode_length=max_episode_length
                         )

        ## Learning Rate mechanisms
        ### Basic optimizer rate set:
        self.learning_rate_p, self.learning_rate_v = (lrp, lrv)
        self.optimizer_p = tf.train.AdamOptimizer(learning_rate=self.learning_rate_p, name="adam_p")
        self.optimizer_v = tf.train.AdamOptimizer(learning_rate=self.learning_rate_v, name="adam_v")
        # GradientDescentOptimizer/RMSPropOptimizer

        ### Mechanism for variable learning speeds via alpha: (focusing on just π-learner for now)
        self.init_spd = alpha_p
        if CyclicSchedule is None:
            self.sched_type, self.sched_halflife = "constant", 1
        else:
            self.sched_type, self.sched_halflife = CyclicSchedule
        self.alpha_p_schedule = cyc_learning_spd(self.sched_type, self.init_spd, self.sched_halflife)
        with tf.variable_scope(self.name):
            self.alpha_p_sched_t = tf.placeholder(shape=None, dtype=tf.float32, name='cyc_alpha_rate_p')

        self.report_labels = ['Perf/Recent Reward',
                              'Losses/Policy LL', 'Losses/Value Fxn', 'Losses/Entropy']
        self.last_good_model = recover

        with tf.variable_scope(self.name):  # slim stacks take care of regularization
            self.learning_rate = tf.placeholder(
                shape=1, dtype=tf.float32, name='learning_rate')

            # Intermediate variables
            # Evaluate ln π(a_t|s_t) for bernoulli action space output
            # Taking advantage of built-in cross_entropy fxn (w/ numeric guards). NB: need -1*...
            self.lnPi_t = - tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.a_logit,  ## a_t|s_t
                labels=self.actions_At  ## use tf.one_hot for m-ary action spaces
            )  # bernoulli_LL(self.actions_At, self.a_prob)
            self.entropy = tf.clip_by_value(
                tf.reduce_mean(bernoulli_H(self.a_prob)),
                _eps_, 100.
            )  # policy_entropy(self.a_logit)
            self.policy_LL = tf.reduce_mean(self.lnPi_t)  # for reporting

            self.Advs_t = self.returns_Gt - self.value  # using value-baselined returns instead of raw G_t
            self.AdvlnPi_t = tf.stop_gradient(self.Advs_t) * self.lnPi_t  # 'AdvlnP'

            # Losses and Gradients
            self.ploss = tf.reduce_mean(self.AdvlnPi_t)
            self.vloss = 0.5 * tf.reduce_mean(
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
                loss=(-self.alpha_p_sched_t) * (self.ploss + _ent_decay_ * self.entropy),
                var_list=self.policy_vars
            )
            # Create training operations
            self.v_train = self.optimizer_v.minimize(
                alpha_v * self.vloss, var_list=self.value_vars
            )
            self.p_train = self.optimizer_p.apply_gradients(self.p_grads)
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

        # generate discounted returns; goes into G_t
        discounted_returns = discount(np.hstack([rewards, [bootstrap_value]]), gamma, window=window)
        discounted_returns = discounted_returns[:-1, None]
        if _DEBUG_:
            # print(self.episode_buffer)
            print(states)
            # print( list(map(lambda x: np.shape(x), [states, actions, discounted_rewards])) )
            input("About to start training call...")
        feed_dict = {
            self.alpha_p_sched_t: self.alpha_p_schedule[self.total_epoch_count % _sched_win_],
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
