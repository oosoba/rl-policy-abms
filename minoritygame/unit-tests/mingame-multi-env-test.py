# coding: utf-8
# ## k-vs-(N-k) Mingame Env

import importlib
import sys
# from collections import Counter

import numpy as np

import minoritygame.minority_multienv as MGME

sys.path.append('./..')
importlib.reload(MGME)

## Init
n_ag = 51
exos = (np.random.sample(n_ag) < 0.3)
n_rl = np.sum(exos)
multmingame = MGME.MinorityGame_Multiagent_env(
    nagents=n_ag, m=3, s=4, exo_actorsQ=exos
)
print(multmingame.actor_count, multmingame.actor_index)
# print(list(zip(range(multmingame.nagents), multmingame.actor_exoQ)))
print(multmingame.h, multmingame.state, multmingame.get_state())

# Test sub-functions
for _ in range(5):
    tmp = multmingame.stepAll()
    print(multmingame.h, tmp[-1])

for _ in range(5):
    acts = 1*(np.random.random(multmingame.actor_count)>0.5)
    tmp2 = multmingame.step(action_list=acts)
    print(multmingame.h, acts)
    print(multmingame.h, tmp2[1])
