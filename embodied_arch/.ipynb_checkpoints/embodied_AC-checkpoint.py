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
# from minoritygame.minority_env import MinorityGame1vN_env
from embodied_arch.embodied import *

tboard_path = "./log"
agent_name = "embodied_AC"
__version__ = "0.0.1"
_DEBUG_ = False

# model: var defaults
_default_reports_ = ['Perf/Recent Reward',
                     'Losses/Policy LL', 'Losses/Entropy',
                     'Norms/Grad Norm', 'Norms/Var Norm']
_every_ = 500
_eps_ = 1.e-4
_g_thresh_ = 20.0  # gradient clip threshold
_ent_decay_ = 5e-2
_zdim_ = 16
(lrp, lrv) = (1e-1, 1e-1)  # learning rates

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
                 recover=None
                 ):
        super().__init__(name=name, env_=env_,
                         latentDim=latentDim, space_size=space_size,
                         sensorium=sensorium, actorNN=actorNN, valueNN=valueNN)

        self.optimizer_p = tf.train.GradientDescentOptimizer(learning_rate=lrp)
        self.optimizer_v = tf.train.GradientDescentOptimizer(learning_rate=lrv)
        self.report_labels = ['Perf/Recent Reward',
                              'Losses/Policy LL', 'Losses/Value Fxn', 'Losses/Entropy']
        self.last_good_model = recover

        with tf.variable_scope(self.name):  # slim stacks take care of regularization
            # # Specify vs_prime evaluator?
            # with tf.variable_scope('value_fxn', reuse=True):
            #     self.vs_prime = ValueNetwork(
            #         sensorium(self.states_St_prime, out_dim=latentDim)
            #     )  # nah... just spec td_target for v-net training instead
            # with tf.variable_scope(dis_scope, reuse=True): # reusing same scope as D(x)
            #     self.disc_fake = discriminatorTemplateFxn(self.class_input, self.gen_sample,
            #                                               hSeq=discSpec)  # D(G(z))

            # Extra placeholder for TD-target in value/critic training
            self.v_TD_target = tf.placeholder(
                shape=[None, 1], dtype=tf.float32, name='Vnet-TD-target')

            # Intermediate variables; may need some tf.reshape(..., [-1])
            # self.lnPi_t = self.action_dist.log_prob(value=self.actions_At)  # ln \pi(a_taken|s)
            # self.entropy = tf.clip_by_value(
            #     tf.reduce_mean(self.action_dist.entropy()),
            #     1.e-2, 100.)  # Avg. over all S_t
            self.lnPi_t = (self.actions_At - _eps_) * tf.log(self.a_prob) + \
                          (1. - self.actions_At) * tf.log(1. - self.a_prob)  # ln \pi(a_taken|s)
            self.entropy = tf.clip_by_value(
                -tf.reduce_mean(tf.log(self.a_prob) * self.a_prob +
                                tf.log(1 - self.a_prob) * (1 - self.a_prob)),
                _eps_, 100.)  # nan entropies fucking things up
            self.Advs_t = self.returns_Gt - self.value
            self.AdvlnPi_t = tf.stop_gradient(self.Advs_t) * self.lnPi_t  # 'AdvlnP'

            # Report vars
            self.policy_LL = tf.reduce_mean(self.lnPi_t)

            # Losses
            self.ploss = - alpha_p * tf.reduce_sum(self.AdvlnPi_t)
            self.vloss = 0.5 * alpha_v * tf.reduce_sum(
                tf.square(
                    self.v_TD_target - tf.reshape(self.value, [-1])
                )
            )  # equiv to minimizing 0.5(td_targ-v)^2

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
            # Training Ops for p+v nets
            self.v_train = self.optimizer_v.minimize(
                self.vloss  # , var_list=self.value_vars
            )
            self.p_train = self.optimizer_p.apply_gradients(self.p_grads)
        return

    def act(self, state, sess):
        """Returns policy net sample action (in {0,1})"""
        # a_t = sess.run(
        #   self.action_dist.sample(),
        #   feed_dict={self.states_St: np.expand_dims(state.flatten(), axis=0)}
        # )
        # return np.array(a_t).squeeze()
        probs = sess.run(self.a_prob, {self.states_St: np.expand_dims(state.flatten(), axis=0)})
        a_t = 1. * (np.random.rand() < probs)
        return a_t.squeeze()

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
        discounted_returns = discounted_returns.ravel()

        # TD(1) target for v-net train: G_t + gamma*v(s_{t+1})
        v_s_prime_td = np.ravel(
            sess.run(self.value,
                     feed_dict={self.states_St: np.vstack(next_states)})
        )
        v_s_prime_td = discounted_returns + gamma * v_s_prime_td

        if _DEBUG_:
            # print(self.episode_buffer)
            print(states)
            # print( list(map(lambda x: np.shape(x), [states, actions, discounted_rewards])) )
            input("About to start training call...")
        feed_dict = {
            self.states_St: np.vstack(states),
            self.actions_At: np.vstack(actions),
            self.returns_Gt: np.vstack(discounted_returns),
            self.v_TD_target: np.vstack(v_s_prime_td),
            self.states_St_prime: np.vstack(next_states)
        }
        # Train
        sess.run([self.v_train, self.p_train], feed_dict=feed_dict)
        # Generate performance statistics to periodically save
        p_ll, v_l, p_ent = sess.run([self.policy_LL, self.vloss, self.entropy], feed_dict=feed_dict)
        return p_ll, v_l, p_ent

    # def work(self, sess, saver, num_epochs, gamma=0.99):
    #     print("Starting agent " + str(self.name))
    #     with sess.as_default(), sess.graph.as_default():
    #         for tk in range(int(num_epochs)):
    #             print("\rEpoch no.: {}/{}".format(tk, num_epochs), end="")
    #             self.episode_buffer = []
    #             while len(self.episode_buffer) == 0:
    #                 self.play(sess)
    #             # TRAIN: Update the network using episodes in the buffer.
    #             stats = list(self.train(sess, gamma))
    #             # save summary statistics for tensorboard
    #             stats.insert(0, self.last_total_return)
    #             if any(np.isnan(stats)):
    #                 saver.restore(sess, self.last_good_model)
    #                 print('Model issues @Step {}. Stats({}): ( {} )'.format(tk, self.report_labels, stats))
    #             else:
    #                 super().model_summary(tk, stats, labels_=self.report_labels)
    #             # save model parameters periodically
    #             if tk % self.autosave_every == 0:
    #                 super().model_save(sess, saver, tk, stats, labels_=self.report_labels)
    #     return
