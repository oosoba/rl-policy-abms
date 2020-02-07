'''
init: Jan2019
Author: LAZ
Goals:
    - Test OO's implementation of REINFORCE
    His claim of testing his own algorithm is a LIE! tsk tsk
        - Hey! depends on how you define "testing"...
        - Test of running performance passe; this was true of the original code
            (see mingame-explore-v2.ipynb)
        - Test of learning performance actually fails... the RF trained net performs worse after training
            - Performance metric: how long pole stays up (episode length)
            - (see test_RF.ipynb)
'''

# from tensorflow.python import debug as tf_debug # debug
import importlib  # debug
import itertools

import gym
import minority_agent
import numpy as np
import tensorflow as tf

importlib.reload(minority_agent)  # debug
tf.reset_default_graph()
env = gym.make('CartPole-v0')

learning_rate = 0.01
num_episodes = 500

rollout = [[] for i in range(4)]
# rollout is [states | actions | rewards | next_states]
episode_rewards = np.zeros(num_episodes)
episode_lengths = np.zeros(num_episodes)

# sess = tf.Session()
REINFORCE = minority_agent.REINFORCE_MG(name='Tester',
                                        s_size=env.observation_space.shape[0],
                                        a_size=1,
                                        trainer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
                                        )
sess = tf.Session(graph=REINFORCE.graph)
# sess = tf_debug.LocalCLIDebugWrapperSession(sess) # debug

REINFORCE.init_graph(sess)


# sess.run(tf.global_variables_initializer())
# sess.run(REINFORCE.init_var())
def process_state(state):
    # helperfunction to make state the correct dims for tensorflow
    # (4,) -> (1, 4)
    return np.expand_dims(state, 0)


for i_episode in range(num_episodes):
    state = env.reset()
    episode = []

    # One step in the environment
    for t in itertools.count():
        state = process_state(state)
        # Take a step
        # tensor_states = tf.get_variable('Tester/states:0')
        # tensor_actions = tf.get_variable('Tester/output_action:0')
        # action = sess.run(tensor_actions, feed_dict={tensor_states: state})
        # action = sess.run(REINFORCE.a, feed_dict={REINFORCE.states: state})
        action = REINFORCE.generate_action(sess, state)
        next_state, reward, done, _ = env.step(np.squeeze(action))
        # Keep track of the transition
        rollout[0].append(state)
        rollout[1].append(action)
        rollout[2].append(reward)
        rollout[3].append(next_state)

        # Update statistics
        episode_rewards[i_episode] += reward
        episode_lengths[i_episode] = t

        # Print out which step we're on, useful for debugging.
        print("\rStep {} @ Episode {}/{} ({})".format(
            t, i_episode + 1, num_episodes, episode_rewards[i_episode - 1]), end="")
        # sys.stdout.flush()

        if done:
            break

        state = next_state

    # Go through the episode and make policy updates
    REINFORCE.train(rollout, sess, 1.0)

# Now test it!

print('Testing...')
state = env.reset()
done = False
rewards = []
while done is False:
    # env.render() # cannot use this on OSX because OpenAI GYM causes segfaults
    state = process_state(state)
    action = REINFORCE.generate_action(sess, state)
    next_state, reward, done, _ = env.step(np.squeeze(action))
    rewards.append(reward)

    state = next_state

assert sum(rewards) == len(rewards), "Test Failed!"
print("Test Succeeded!")
