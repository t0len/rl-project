"""Drone delivery MDP: 5x5 grid, battery, charging stations, windy zone.

State s = (row, col, battery, has_package).
transitions(s, a) yields (prob, next_state, reward, terminated).
"""
from functools import lru_cache

import numpy as np

N = 5
B_MAX = 8
START = (0, 0)
PICKUP = (0, 4)
DEST = (4, 0)
STATIONS = frozenset({(0, 2), (2, 2)})
WIND_ZONE = frozenset((r, c) for r in range(1, 4) for c in range(1, 4))
WIND_P = 0.3
CHARGE_AMT = 2
MAX_STEPS = 60

R_STEP = -1.0
R_DELIVER = 30.0
R_CRASH = -30.0

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["up", "down", "left", "right", "charge"]
N_ACTIONS = 5
N_STATES = N * N * (B_MAX + 1) * 2


def encode(s):
    r, c, b, p = s
    return ((r * N + c) * (B_MAX + 1) + b) * 2 + p


def all_states():
    return [(r, c, b, p) for r in range(N) for c in range(N)
            for b in range(B_MAX + 1) for p in (0, 1)]


@lru_cache(maxsize=None)
def transitions(s, a):
    r, c, b, p = s
    if a == 4:
        nb = min(B_MAX, b + CHARGE_AMT) if (r, c) in STATIONS else b
        return ((1.0, (r, c, nb, p), R_STEP, False),)

    if (r, c) in WIND_ZONE:
        side = (2, 3) if a in (0, 1) else (0, 1)  # perpendicular to the chosen move
        outcomes = [(1 - WIND_P, a), (WIND_P / 2, side[0]), (WIND_P / 2, side[1])]
    else:
        outcomes = [(1.0, a)]

    result = []
    for prob, act in outcomes:
        dr, dc = MOVES[act]
        nr = min(max(r + dr, 0), N - 1)
        nc = min(max(c + dc, 0), N - 1)
        nb = b - 1
        np_ = 1 if (nr, nc) == PICKUP else p
        reward, done = R_STEP, False
        if p == 1 and (nr, nc) == DEST:
            reward, done = R_STEP + R_DELIVER, True
        elif nb < 0 or (nb == 0 and (nr, nc) not in STATIONS):  # empty battery off a station
            reward, done = R_STEP + R_CRASH, True
        result.append((prob, (nr, nc, max(nb, 0), np_), reward, done))
    return tuple(result)


class DroneEnv:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.s, self.t = None, 0
        self.last_crashed = False

    def reset(self):
        self.s, self.t = (*START, B_MAX, 0), 0
        self.last_crashed = False
        return self.s

    def step(self, a):
        outs = transitions(self.s, a)
        u, acc = self.rng.random(), 0.0
        for prob, s2, rew, done in outs:
            acc += prob
            if u <= acc:
                break
        self.s = s2
        self.t += 1
        self.last_crashed = done and rew < 0
        truncated = (not done) and self.t >= MAX_STEPS
        return s2, rew, done, truncated
