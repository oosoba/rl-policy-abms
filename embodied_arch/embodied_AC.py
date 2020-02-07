'''
init: Feb2019
Author: OO
Topic: Actor-Critic Implementation for Binary action-space env.
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
# from minoritygame.minority_env import MinorityGame1vN_env
from embodied_arch.embodied import *
from embodied_arch.embodied_misc import bernoulli_H
from embodied_arch.embodied_misc import calc_V_TD_target, _sched_win_, cyc_learning_spd

tboard_path = "./log"
agent_name = "embodied_AC"
__version__ = "0.0.3"
_DEBUG_ = False

# model: var defaults
_default_reports_ = ['Perf/Recent Reward',
                     'Losses/Policy LL', 'Losses/Entropy',
                     'Norms/Grad Norm', 'Norms/Var Norm']

_every_ = 300
_eps_ = 1.e-4

_g_thresh_ = 20.0  # gradient clip threshold
_ent_decay_ = 5e-3
_zdim_ = 16
(lrp, lrv) = (1e-3, 1e-2)  # learning rates

# envs: var defaults
(na_, m_, s_, p_) = (33, 3, 4, 0.5)
_max_len_ = 200
_s_size_, _a_size_ = (4, 1)


class EmbodiedAgentAC(EmbodiedAgent):
    def __init__(self, name=agent_name,
                 env_=gym.make('CartPole-v0'),
                 alpha_p=5e-2, alpha_v=1e-1,
                 latentDim=_zdim_,
                 space_size=(_s_size_, _a_size_),
                 sensorium=SensoriumNetworkTemplate,
                 actorNN=ActionPolicyNetwork,
                 valueNN=ValueNetwork,
                 recover=None, _every_=_every_,
                 max_episode_length=_max_len_,
                 lrp=lrp, lrv=lrv,
                 CyclicSchedule=None
                 ):
        super().__init__(name=name, env_=env_,
                         latentDim=latentDim, space_size=space_size,
                         sensorium=sensorium, actorNN=actorNN, valueNN=valueNN,
                         _every_=_every_, max_episode_length=max_episode_length
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

        self.report_labels = ['Perf/Recent Reward',
                              'Losses/Policy LL', 'Losses/Value Fxn', 'Losses/Entropy']
        self.last_good_model = recover

        with tf.variable_scope(self.name):
            self.alpha_p_sched_t = tf.placeholder(
                shape=None, dtype=tf.float32,
                name='cyc_alpha_rate_p'
            )
            # Intermediate variables
            # Evaluate ln π(a_t|s_t) for bernoulli action space output
            # Taking advantage of built-in cross_entropy fxn (w/ numeric guards). NB: need -1*...
            self.lnPi_t = - tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.a_logit,  ## a_t|s_t
                labels=self.actions_At  ## use tf.one_hot for m-ary action spaces
            )  # bernoulli_LL(self.actions_At, self.a_prob)
            self.entropy = tf.clip_by_value(
                tf.reduce_mean(bernoulli_H(self.a_prob)),
                _eps_, 10.
            )  # policy_entropy(self.a_logit)
            self.policy_LL = tf.reduce_mean(self.lnPi_t)  # Report vars
            self.Advs_t = self.returns_Gt - self.value
            self.AdvlnPi_t = tf.stop_gradient(self.Advs_t) * self.lnPi_t

            # Losses
            self.ploss = tf.reduce_mean(self.AdvlnPi_t)
            self.vloss = 0.5 * tf.reduce_mean(
                tf.square(self.returns_Gt - self.value)
            )
            # self.semi_value_loss = tf.reduce_mean(self.delta_Advs_t * self.value)
            # obj: to max the dot prod Adv.V. Loss: (- alpha_v)*self.semi_value_loss

            # Separate out: Training Vars
            self.policy_vars = [v for v in tf.trainable_variables()
                                if "action" in v.name or "sensorium" in v.name]
            self.value_vars = [v for v in tf.trainable_variables()
                               if "value_fxn" in v.name or "sensorium" in v.name]

            # Training Ops for p+v nets
            self.p_grads = self.optimizer_p.compute_gradients(
                loss=(-self.alpha_p_sched_t) * (self.ploss + _ent_decay_ * self.entropy),
                var_list=self.policy_vars
            )
            self.p_train = self.optimizer_p.apply_gradients(self.p_grads)
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

        # generate state-value estimates; goes into delta_t
        rwds = rewards.ravel()
        vals = sess.run(self.value,
                        feed_dict={self.states_St: np.vstack(states)}
                        ).ravel()  # v(s)
        Gt_TD = np.squeeze(
            calc_V_TD_target(rwds, vals,
                             gamma=gamma, bootstrap_value=bootstrap_value)
        )
        # TD(1) target for v-net train: G_t + gamma*v(s_{t+1})
        feed_dict = {
            self.alpha_p_sched_t: self.alpha_p_schedule[self.total_epoch_count % _sched_win_],
            self.states_St: np.vstack(states),
            self.actions_At: np.vstack(actions),
            self.returns_Gt: np.vstack(Gt_TD)
        }
        # Train
        sess.run(
            [self.v_train, self.p_train],
            feed_dict=feed_dict
        )
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
