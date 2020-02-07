import numpy as np
import itertools
from matplotlib import pyplot as plt

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
        
    def _draw_strategies(self):
        history = list(itertools.product(['0','1'], repeat=self.m))
        history = [''.join(x) for x in history]
        actions = np.random.randint(0,2, size=(self.s, 2**self.m))
        # Can check here to make sure no two rows the same if want to
        # eliminate duplicates.  However, as m->infty, the probability
        # of a duplicate goes to 0.
        strats = [dict(zip(history, actions[i,:])) for i in range(self.s)]
        return strats

    def get_action(self, h):
        strat = self.strategies[np.argmax(self.vpoints)]
        return strat[h]
    
    def update_virtual_points(self, h, winner):
      for ix, s in enumerate(self.strategies):
          if s[h] == winner:
              self.vpoints[ix] += 1
    


class MinorityGame(object):
    """
    A class that creates agents and runs the game.

    Parameters
    ----------

    nagents : int
        The number of agents

    m : int
        The memory of each agent

    s : int
        The number of strategies per agent

    p : float \in [0,1]
        Minority proportion. i.e.  Agent i wins if the proporation of
        agents taking the same action of agent i is less than p.  

    """
    def __init__(self, nagents, m, s, p=.5):
        self.nagents = nagents
        self.agents = [Agent(m,s) for x in range(nagents)]
        self.h = ''.join(np.random.choice(['0', '1'], size=m)) # Initial History
        self.p = p

    def step(self):
        actions = [a.get_action(self.h) for a in self.agents]
        winner = int(np.sum(actions) / float(self.nagents) < self.p)
        [a.update_virtual_points(self.h, winner) for a in self.agents]
        self.h = self.h[1:] +str(winner)
        return actions, winner # This is like returning the next state?


def repro_fig_1():
    """
    This function (roughly) reproduces figure 1 in 
    Challet and Zhang 'Emergence of Cooperation and Organization in an
    Evolutionary Game.' Physica A: Statistical Mechanics and its
    Applications. 1997.
    """
    game1 = MinorityGame(1001,6,3)
    g1y  = [np.sum(game1.step()[0]) for x in range(1000)]
    game2 = MinorityGame(1001,8,3)
    g2y  = [np.sum(game2.step()[0]) for x in range(1000)]
    game3 = MinorityGame(1001,10,3)
    g3y  = [np.sum(game3.step()[0]) for x in range(1000)]
    fig, axes = plt.subplots(3)
    ys = [g1y, g2y, g3y]
    [axes[i].plot(ys[i]) for i in range(3)]
    [ax.set_ylim(0,1000) for ax in axes]
    [ax.set_xlim(0,1000) for ax in axes]
    [ax.set_xlabel('Iteration') for ax in axes]
    [ax.set_ylabel('Bar Attendees') for ax in axes]
    fig.text(.5, .03,
        'Minority game with s=3 and M=6,8,10 from top to bottom',
             ha='center', fontsize=18)    
    return fig, axes
