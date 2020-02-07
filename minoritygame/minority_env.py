import copy
from collections import Counter

import numpy as np

from .minority_agent import Agent

__version__ = "0.0.3"

'''see *_multienv code for multi-agent m-vs-(N-m) version of this'''

_burn_in_default_ = 500
_newstrats_default_ = False


def clip_rwds(x): return np.clip(x, a_min=0., a_max=1.)


class MinorityGame1vN_env(object):
    """
    A class that creates a multi-agent environment for a minority game.
    (N-1) non-RL agents interacting with 1 RL agent (the N^{th}).
    Sides: A|B <-> 0|1

    Constructor Parameters:
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

    def __init__(self, nagents,
                 m, s, p=.5, mrl=None,
                 burn_in=_burn_in_default_):
        if mrl is None:
            self.mrl = m
        else:
            self.mrl = mrl
        self.m = m
        self.p = p
        self.nagents = nagents
        self.agents = [Agent(m, s) for _ in range(nagents)]  # [Agent(m, s) for _ in range(nagents - 1)]
        self.burn_in = burn_in
        # self.agents.append(Agent(mrl, s))  # I don't think we need an RL agent as an actual agent instance
        if self.mrl >= self.m:
            self.hrl = ''.join(np.random.choice(['0', '1'], size=self.mrl))  # Initial History
            self.h = copy.copy(self.hrl[self.mrl - self.m:])
        else:
            self.h = ''.join(np.random.choice(['0', '1'], size=self.m))  # Initial History
            self.hrl = copy.copy(self.h[self.m - self.mrl:])
        self.state = np.array([float(hi) for hi in (self.hrl)])
        self.state_space_size = len(self.state)  # mrl
        self.action_space_size = 1
        return

    def stepAll(self):
        """
        Steps the game 1 step forward in time for all players (including the N^{th} player).
        All players evolve according to default rules (fixed strategy-book policy update).
        Returns:
            s_{t+1}: next env state (limited common record of past winning)
            a_t: actions taken in this step
            winner: which side won minority vote? (0|1)
                (may not be winning side if p<0.5)
        """
        actions = [a.get_action(self.h) for a in self.agents]
        # may want to update this to get agent-N to evolve separately here
        winner = int(Counter(actions).most_common()[-1][0])
        # i.e. frac of agents take action 0|A < p?
        [a.update_virtual_points(self.h, winner) for a in self.agents]
        self.h = self.h[1:] + str(winner)
        self.hrl = self.hrl[1:] + str(winner)
        self.state = np.array([int(hi) for hi in (self.hrl)])
        return self.state, actions, winner

    def step(self, act_N):
        """
        Steps the game 1 step forward in time, taking the N^{th} player's action
        into account (exogenously specified).
        (N-1) players evolve according to default rules (fixed strategy-book policy update).
        Agent-N reward exposed for separate adaptation.

        Returns:
            s_{t+1}: next env state (limited common record of past winning)
            r_t: reward for N^{th} agents actions i.e. was agent-N on winning side?
                shaped to give some rwd when on the minority side but not below p threshold.
            done: flag stating whether current episode is over (always false)
        """
        # sample default/non-rl action for all agents
        actions = [a.get_action(self.h) for a in self.agents]

        # parse/input/override default actions for RL agents
        act_N_clean = np.clip(int(act_N), a_min=0, a_max=1)
        actions[-1] = act_N_clean

        # judge the winning side
        most_common = Counter(actions).most_common()
        if len(most_common)<2: print(most_common)
        if most_common[0][1] == most_common[1][1]:
            winner = act_N_clean
            tally = most_common[-1][1]
        else:
            winner, tally = Counter(actions).most_common()[-1]
            winner = int(winner)  # winner: label of least common choice
        tally = tally / float(self.nagents)
        # update memory/state info
        [a.update_virtual_points(self.h, winner) for a in self.agents]
        self.h = self.h[1:] + str(winner)
        self.hrl = self.hrl[1:] + str(winner)
        self.state = np.array([int(hi) for hi in self.hrl])

        # Calc scaled rwd;
        #   accounts for cases where p=/=0.5
        scaled_rwd = 1. if (self.p == 0.5) \
            else float((0.5 - tally) / (0.5 - self.p))
        reward = clip_rwds(scaled_rwd) if (winner == act_N_clean) else 0.
        # reward = 1. if act_N_clean==1 else 0 #Just testing to make sure he learns to always play 1.  Change back
        return self.state, float(reward), False, {}

    def get_state(self):
        return self.state

    def reset(self):
        '''custom reset procedure for minority game env. Edit _newstrats_default_ for different behavior'''
        st = self.resetMinGame(newstrats=_newstrats_default_)
        return st

    def resetMinGame(self, newstrats=_newstrats_default_):
        '''This needs to just reset the virtual points for the same strategies
        without resampling entirely new strategies!!!
        Retaining the old histories/memories for now...
        '''
        self.state = np.array([float(hi) for hi in (self.hrl)])
        # now burn strategies in for a few iterations to get semi-stationary behavior
        # print("Resetting and burn-in strategies...\n")
        for _ in range(self.burn_in):
            _, _, _ = self.stepAll()
        # might want to reset virtual points on reset too...
        # for ag in self.agents:
        #     ag.reset_virtual_points()

        if newstrats:
            # reinitilizing is a bad idea; resamples completely new strats
            _m, _s, _n = (self.agents[0].m, self.agents[0].s, len(self.agents))
            self.agents = [Agent(_m, _s) for x in range(_n)]
            if self.mrl >= self.m:
                self.hrl = ''.join(np.random.choice(['0', '1'], size=self.mrl))  # Initial History
                self.h = copy.copy(self.hrl[self.mrl - self.m:])
            else:
                self.h = ''.join(np.random.choice(['0', '1'], size=self.m))  # Initial History
                self.hrl = copy.copy(self.h[self.m - self.mrl:])
        return self.state
