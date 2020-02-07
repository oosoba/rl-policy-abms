import numpy as np
import itertools


class Agent(object):
    """
    An Agent Object for the Minority Game. The agent is based off of:

    Challet and Zhang 'Emergence of Cooperation and Organization in an
    Evolutionary Game.' Physica A: Statistical Mechanics and its
    Applications. 1997.  These objects should only be created through
    the MinorityGame class.

    Parameters
    ----------
    m : int
        Memory Parameter
    s : int
        Stratgy space parameter.  Of the 2^(2^M) strategies, how many
        strategies does the agent consider.  There may be duplicates.
        Can always put in a check to ensure no duplicates.
    """

    def __init__(self, m, s):
        self.m = m
        self.s = s
        self.strategies = self._draw_strategies()
        self.vpoints = np.zeros(self.s)

    def _draw_strategies(self):  # this needs optimization... no str conversion
        history = list(itertools.product(['0', '1'], repeat=self.m))
        history = [''.join(x) for x in history]
        actions = np.random.randint(0, 2, size=(self.s, 2**self.m))
        # Can check here to make sure no two rows the same if want to
        # eliminate duplicates.  However, as m->infty, the probability
        # of a duplicate goes to 0.
        strats = [dict(zip(history, actions[i, :])) for i in range(self.s)]
        return strats

    def get_action(self, h):
        strat = self.strategies[np.argmax(self.vpoints)]
        return strat[h]

    def update_virtual_points(self, h, winner):
        for ix, s in enumerate(self.strategies):
            if s[h] == winner:
                self.vpoints[ix] += 1
        return

    def reset_virtual_points(self):
        self.vpoints = np.zeros(self.s)
        return
