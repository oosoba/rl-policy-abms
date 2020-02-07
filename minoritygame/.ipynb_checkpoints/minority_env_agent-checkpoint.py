import numpy as np
import itertools
from matplotlib import pyplot as plt
import minority


class MinorityGame1vN_env(object):
    """
    A class that creates a multi-agent environment for a minority game.

    Parameters
    ----------

    nagents : int (even)
        (N-1) number of agents

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
        self.nagents = nagents
        self.agents = [Agent(m,s) for x in range(nagents)]
        self.h = ''.join(np.random.choice(['0', '1'], size=m)) # Initial History
        self.p = p

    def stepAll(self):
        """ 
        Steps the game 1 step forward in time for all players (including the N^{th} player). 
        All players evolve according to default rules (fixed strategy-book policy update)
        """
        actions = [a.get_action(self.h) for a in self.agents]
        wonQ = int(np.sum(actions) / float(self.nagents) < self.p)
        [a.update_virtual_points(self.h, wonQ) for a in self.agents]
        self.h = self.h[1:] +str(wonQ)
        return actions, wonQ # This is like returning the next state?

    def step(self, actN):
        """ 
        Steps the game 1 step forward in time taking the N^{th} player's action
        into account. 
        (N-1) players evolve according to default rules (fixed strategy-book policy update)
        """
        actions = [a.get_action(self.h) for a in self.agents]
        wonQ = int(np.sum(actions) / float(self.nagents) < self.p)
        [a.update_virtual_points(self.h, wonQ) for a in self.agents]
        self.h = self.h[1:] +str(wonQ)
        return actions, wonQ # This is like returning the next state?

