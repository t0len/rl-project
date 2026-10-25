"""Tabular TD control with epsilon-greedy exploration (Q-learning or SARSA)."""
import numpy as np

import env as E


class TabularAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, method="qlearning", seed=0):
        self.alpha, self.gamma, self.epsilon, self.method = alpha, gamma, epsilon, method
        self.rng = np.random.default_rng(seed)
        # Returns are negative at first, so zeros look better than tried actions.
        self.Q = np.zeros((E.N_STATES, E.N_ACTIONS))

    def _greedy(self, i, random_ties=True):
        row = self.Q[i]
        if random_ties:
            return int(self.rng.choice(np.flatnonzero(row == row.max())))
        return int(np.argmax(row))

    def act(self, s):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(E.N_ACTIONS))
        return self._greedy(E.encode(s))

    def greedy_policy(self):
        return lambda s: self._greedy(E.encode(s), random_ties=False)

    def update(self, s, a, r, s2, a2, terminated):
        i, j = E.encode(s), E.encode(s2)
        if terminated:
            target = r
        elif self.method == "qlearning":
            target = r + self.gamma * self.Q[j].max()
        else:
            target = r + self.gamma * self.Q[j, a2]
        self.Q[i, a] += self.alpha * (target - self.Q[i, a])


def train(agent, env, episodes):
    """Per-episode return and success while exploring."""
    returns, successes = [], []
    for _ in range(episodes):
        s = env.reset()
        a = agent.act(s)
        G, done, trunc, ok = 0.0, False, False, 0
        while not (done or trunc):
            s2, r, done, trunc = env.step(a)
            a2 = agent.act(s2)
            agent.update(s, a, r, s2, a2, done)
            G += r
            if done and r > 0:
                ok = 1
            s, a = s2, a2
        returns.append(G)
        successes.append(ok)
    return np.array(returns), np.array(successes)


def evaluate(policy, env, episodes):
    """Mean return, success rate, and crash rate of a fixed policy."""
    G_all, ok, crash = [], 0, 0
    for _ in range(episodes):
        s, G, done, trunc = env.reset(), 0.0, False, False
        while not (done or trunc):
            s, r, done, trunc = env.step(policy(s))
            G += r
        G_all.append(G)
        ok += int(done and r > 0)
        crash += int(env.last_crashed)
    return float(np.mean(G_all)), ok / episodes, crash / episodes
