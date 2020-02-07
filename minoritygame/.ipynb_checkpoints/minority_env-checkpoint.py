import numpy as np
import itertools
from matplotlib import pyplot as plt
from minority import Agent

__version__ = "0.0.1"

'''Needs:
    - m-vs-(N-m) env module'''


class MinorityGame1vN_env(object):
    """
    A class that creates a multi-agent environment for a minority game.

    Parameters
    ----------

    nagents : int (odd)
        N number of agents
        N^{th} agent is a shell for external actor

    m : int
        The memory of each agent

    s : int
        The number of strategies per agent

    p : float \in [0,1] (default:0.5)
        Minority proportion. i.e.  Agent i wins if the proportion of
        agents taking the same action of agent i is less than p.
        (reward evaluation is a global/popn-level function)
    """
    def __init__(self, nagents, m, s, p=.5):
        self.p = p
        self.nagents = nagents
        self.agents = [Agent(m,s) for x in range(nagents)]
        self.h = ''.join(np.random.choice(['0', '1'], size=m)) # Initial History
        self.state = np.array([float(hi) for hi in (self.h)])
        self.state_space_size = m
        self.action_space_size = 1


        return

    def stepAll(self):
        """
        Steps the game 1 step forward in time for all players (including the N^{th} player).
        All players evolve according to default rules (fixed strategy-book policy update).
        Returns:
            s_{t+1}: next env state (limited common record of past winning)
            A_wonQ: did side A win?
        """
        actions = [ a.get_action(self.h) for a in self.agents ]
        # may want to update this to get agent-N to evolve separately here
        A_wonQ = int(np.sum(actions) / float(self.nagents) < self.p)
        [a.update_virtual_points(self.h, A_wonQ) for a in self.agents]
        self.h = self.h[1:] +str(A_wonQ)
        self.state = np.array([float(hi) for hi in (self.h)])
        return self.state, A_wonQ

    def step(self, act_N):
        """
        Steps the game 1 step forward in time, taking the N^{th} player's action
        into account.
        (N-1) players evolve according to default rules (fixed strategy-book policy update).
        Agent-N rules exposed for separate adaptation.

        Returns:
            s_{t+1}: next env state (limited common record of past winning)
            r_t: reward for N^{th} agents actions i.e. was agent-N on winning side?
            done: flag stating whether current episode is over (always false)
        """
        actions = [a.get_action(self.h) for a in self.agents]
        act_N_clean = min( max(0, int(act_N)), 1)
        actions[-1] = act_N_clean
        A_wonQ = int(np.sum(actions) / float(self.nagents) < self.p)
        [a.update_virtual_points(self.h, A_wonQ) for a in self.agents]
        self.h = self.h[1:] + str(A_wonQ)
        self.state = np.array([float(hi) for hi in (self.h)])
        reward = int( A_wonQ == act_N_clean )
        return self.state, float(reward), False, {}

    def get_state(self): return np.array([float(hi) for hi in (self.h)])

    def reset(self):
        _m, _s, _n = (self.agents[0].m, self.agents[0].s, len(self.agents))
        self.agents = [Agent(_m,_s) for x in range(_n)]
        self.h = ''.join( np.random.choice(['0', '1'], size=_m) )
        self.state = np.array([float(hi) for hi in (self.h)])
        return self.state
