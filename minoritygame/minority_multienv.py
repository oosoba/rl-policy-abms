'''
init: May2019
Author: OO
<<in development>>
Goals: env spec for multiple indep agents
    - All agents interact w/ a common env => single menv
    - The shared env is a property of the agent popn => menv is a param
    - All agents see the same full state info. => S_t is shared
    - All RL agents act independently => separate indeps Actor & Value NNs
    - No explicit collusion or shared learning => separate sensoria NNs
'''

from itertools import compress

from minoritygame.minority_env import *
from minoritygame.minority_env import MinorityGame1vN_env as MinorityGame_SingleAgent_env

__version__ = "0.0.1"

unit_clip = lambda x: np.clip(x, a_min=0., a_max=1.)
_n_agents_ = 33


class MinorityGame_Multiagent_env(MinorityGame_SingleAgent_env):
    """
    A class that creates a multi-agent environment for a minority game.
    (N-k) non-RL agents interacting with k RL agents.

    Constructor Parameters:
    exo_actorsQ : list (odd in length)
        Boolean index list of length N=# of agents (of which k act and evolve via RL policies).
        Specify 'True' at positions for actors with exogenous RL-based behavior.
        Replaces/deprecates: nagents : int (odd)
    m : int
        The memory of each agent
    s : int
        The number of strategies per agent
    p : float \in [0,1] (default:0.5)
        Minority proportion. i.e.  Agent i wins if the proportion of
        agents taking the same action of agent i is less than p.
        (reward evaluation is a global/popn-level function)

    Inherits: reset() & stepAll() & get_state()
    Overrides: step()
    """

    def __init__(self, m, s, exo_actorsQ=None, p=.5, mrl=None):  # nagents deprecated
        self.actor_exoQ = exo_actorsQ if \
            exo_actorsQ is not None \
            else np.ones(_n_agents_, dtype=bool)  # bool flag vec, defaults to all actor being exogenous
        super().__init__(
            nagents=len(self.actor_exoQ),
            m=m, s=s, p=p, mrl=mrl)
        self.actor_exoQ = self.actor_exoQ[:min(len(self.actor_exoQ), self.nagents)]
        self.actor_count = sum(self.actor_exoQ)
        self.actor_index = list(compress(range(self.nagents), self.actor_exoQ))
        self.action_space_size = 1  # always == action-dim per actor
        # each self.actor_count emits 1-d actions
        return

    def random_actions(self):
        return 1 * (np.random.random(self.actor_count) > 0.5)

    def reset(self):
        '''custom reset procedure for multi-agent minority game env. Edit _newstrats_default_ for different behavior'''
        st = super().reset()
        sts = np.tile(st, (self.actor_count, 1))  # repl same state to all RL agents
        return sts

    def step(self, action_list):
        """Steps the game dt=1 step forward in time.
        m Agents with exogenously-specified actions.
        (N-m) players evolve according to default rules (fixed strategy-book policy update).
        m-length reward vector exposed/returned for separate adaptation.

        Parameter:
        a_t: exogenous action specification for the m RL players. (iterable: list|np.array)

        Returns:
            s_{t+1}: next env state (limited common record of past winning)
            r_t: reward for N^{th} agents actions i.e. was agent-N on winning side? (1.|0.)
            done: flag stating whether current episode is over (always false)
        """
        # sample default/non-rl action for all agents
        actions = [a.get_action(self.h) for a in self.agents]

        # parse/input/override default actions for RL agents
        if self.actor_count == 1:
            action_list = [action_list]
        assert len(action_list) == self.actor_count, "wrong number of actions specified"
        action_list_clean = [int(unit_clip(a)) for a in action_list]
        for k in range(self.actor_count):
            actions[self.actor_index[k]] = action_list_clean[k]

        # judge the winning side
        winner, tally = Counter(actions).most_common()[-1]
        winner = int(winner)
        tally = tally / float(self.nagents)

        # update memory/state vars: shared across agent sub-populations
        [a.update_virtual_points(self.h, winner) for a in self.agents]
        self.h = self.h[1:] + str(winner)
        self.hrl = self.hrl[1:] + str(winner)
        self.state = np.tile(
            np.array([int(hi) for hi in self.hrl]),
            (self.actor_count, 1)
        )  # repl same state to all RL agents

        # Calc scaled rwd: same value shared across sub-population of *winning* RL agents
        #   accounts for cases where p=/=0.5
        scaled_rwd = 1. if (self.p == 0.5) \
            else float((0.5 - tally) / (0.5 - self.p))
        scaled_rwd = unit_clip(scaled_rwd)
        rewards = scaled_rwd * (np.array(action_list_clean) == winner)
        return self.state, rewards, False, {}
