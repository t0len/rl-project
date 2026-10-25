"""Random, battery-blind, fuel-aware, and value-iteration policies."""
import numpy as np

import env as E


def _target(s):
    return E.DEST if s[3] == 1 else E.PICKUP


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step_toward(pos, goal):
    if pos[0] != goal[0]:
        return 1 if goal[0] > pos[0] else 0
    return 3 if goal[1] > pos[1] else 2


def random_policy(rng):
    return lambda s: int(rng.integers(E.N_ACTIONS))


def blind_policy(rng=None):
    return lambda s: _step_toward(s[:2], _target(s))


def fuel_aware_policy(rng=None):
    def pol(s):
        pos, b = s[:2], s[2]
        tgt = _target(s)
        if b < _dist(pos, tgt):
            if pos in E.STATIONS:
                return 4
            st = min(E.STATIONS, key=lambda x: _dist(pos, x))
            return _step_toward(pos, st)
        return _step_toward(pos, tgt)
    return pol


def value_iteration(gamma=0.99, tol=1e-8):
    states = E.all_states()
    V = {s: 0.0 for s in states}
    while True:
        delta = 0.0
        for s in states:
            best = max(sum(p * (r + (0.0 if d else gamma * V[s2]))
                           for p, s2, r, d in E.transitions(s, a))
                       for a in range(E.N_ACTIONS))
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < tol:
            break
    pi = {}
    for s in states:
        q = [sum(p * (r + (0.0 if d else gamma * V[s2])) for p, s2, r, d in E.transitions(s, a))
             for a in range(E.N_ACTIONS)]
        pi[s] = int(np.argmax(q))
    return lambda s: pi[s]
