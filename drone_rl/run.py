"""Train the agent, compare baselines, and write tables and figures."""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import baselines as B
import env as E
from agent import TabularAgent, evaluate, train

ALPHA, GAMMA = 0.1, 0.99
EVAL_EPISODES = 200
OUT = "results"


def smooth(x, w=100):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def run_agent(seed, epsilon, episodes, method):
    agent = TabularAgent(ALPHA, GAMMA, epsilon, method, seed=seed)
    ret, suc = train(agent, E.DroneEnv(seed=seed), episodes)
    ev = evaluate(agent.greedy_policy(), E.DroneEnv(seed=10_000 + seed), EVAL_EPISODES)
    return ret, suc, ev, agent


def ms(x):
    return f"{np.mean(x):.2f} ± {np.std(x):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--method", default="qlearning", choices=["qlearning", "sarsa"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.episodes, args.seeds = 500, 3
    seeds = list(range(args.seeds))
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    print(f"method={args.method} episodes={args.episodes} seeds={seeds}")

    eps_values = [0.01, 0.1, 0.3]
    runs = {}
    for eps in eps_values:
        runs[eps] = [run_agent(s, eps, args.episodes, args.method) for s in seeds]
        print(f"  eps={eps}: done ({time.time() - t0:.0f}s)")

    vi_policy = B.value_iteration(GAMMA)
    base = {"Random": [], "Battery-blind rule": [], "Fuel-aware rule": [], "Value iteration (model known)": []}
    for s in seeds:
        mk = lambda: E.DroneEnv(seed=10_000 + s)
        base["Random"].append(evaluate(B.random_policy(np.random.default_rng(s)), mk(), EVAL_EPISODES))
        base["Battery-blind rule"].append(evaluate(B.blind_policy(), mk(), EVAL_EPISODES))
        base["Fuel-aware rule"].append(evaluate(B.fuel_aware_policy(), mk(), EVAL_EPISODES))
        base["Value iteration (model known)"].append(evaluate(vi_policy, mk(), EVAL_EPISODES))

    label = "Q-learning" if args.method == "qlearning" else "SARSA"
    table = {f"{label} (eps=0.1)": [r[2] for r in runs[0.1]], **base}
    lines = ["| Method | Mean return | Success rate | Crash rate |", "|---|---|---|---|"]
    for name, evs in table.items():
        a = np.array(evs)
        lines.append(f"| {name} | {ms(a[:, 0])} | {ms(a[:, 1])} | {ms(a[:, 2])} |")
    t1 = "\n".join(lines)

    lines = ["| epsilon | Eval return | Success rate | Crash rate | Training return (last 500 ep.) |",
             "|---|---|---|---|---|"]
    for eps in eps_values:
        a = np.array([r[2] for r in runs[eps]])
        tr = [r[0][-500:].mean() for r in runs[eps]]
        lines.append(f"| {eps} | {ms(a[:, 0])} | {ms(a[:, 1])} | {ms(a[:, 2])} | {ms(tr)} |")
    t2 = "\n".join(lines)

    with open(f"{OUT}/tables_{args.method}.md", "w") as f:
        f.write(f"## Table 1. Final comparison (greedy eval, {EVAL_EPISODES} episodes x {len(seeds)} seeds; mean ± std over seeds)\n\n{t1}\n\n")
        f.write(f"## Table 2. Epsilon study\n\n{t2}\n")
    print("\nTable 1\n" + t1 + "\n\nTable 2\n" + t2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    curves = np.array([smooth(r[0]) for r in runs[0.1]])
    succ = np.array([smooth(r[1]) for r in runs[0.1]])
    x = np.arange(curves.shape[1]) + 100
    for a_, data, yl in [(ax[0], curves, "Return per episode"), (ax[1], succ, "Delivery success rate")]:
        m, sd = data.mean(0), data.std(0)
        a_.plot(x, m, label=f"{label} (eps=0.1), mean of {len(seeds)} seeds")
        a_.fill_between(x, m - sd, m + sd, alpha=0.25, label="± 1 std")
        a_.set_xlabel("Training episode"); a_.set_ylabel(yl); a_.grid(alpha=.3)
    vi_mean = np.mean([e[0] for e in base["Value iteration (model known)"]])
    rnd_mean = np.mean([e[0] for e in base["Random"]])
    ax[0].axhline(vi_mean, color="g", ls="--", label="Value iteration (optimal)")
    ax[0].axhline(rnd_mean, color="r", ls=":", label="Random")
    ax[0].legend(fontsize=7)
    fig.suptitle("Figure 1. Training performance (moving average, window 100; exploration on)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_training_{args.method}.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    names = list(table.keys())
    for a_, k, yl in [(ax[0], 0, "Eval return"), (ax[1], 1, "Success rate")]:
        vals = [np.array(table[n])[:, k] for n in names]
        a_.bar(range(len(names)), [v.mean() for v in vals], yerr=[v.std() for v in vals], capsize=4,
               color=["tab:blue", "tab:red", "tab:orange", "tab:olive", "tab:green"])
        a_.set_xticks(range(len(names)))
        a_.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
        a_.set_ylabel(yl); a_.grid(axis="y", alpha=.3)
    fig.suptitle("Figure 2. Final greedy evaluation (mean ± std over seeds)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_comparison_{args.method}.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for eps in eps_values:
        c = np.array([smooth(r[0]) for r in runs[eps]])
        m, sd = c.mean(0), c.std(0)
        ax.plot(x, m, label=f"eps={eps}")
        ax.fill_between(x, m - sd, m + sd, alpha=0.2)
    ax.set_xlabel("Training episode"); ax.set_ylabel("Return per episode (training)")
    ax.set_title("Figure 3. Effect of epsilon"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_epsilon_{args.method}.png", dpi=150); plt.close(fig)

    agent = runs[0.1][0][3]
    e = E.DroneEnv(seed=123)
    s, pol, steps = e.reset(), agent.greedy_policy(), []
    done = trunc = False
    while not (done or trunc):
        a = pol(s)
        s2, r, done, trunc = e.step(a)
        steps.append(f"{s} --{E.ACTION_NAMES[a]}--> r={r:+.0f} {s2}")
        s = s2
    with open(f"{OUT}/example_trajectory_{args.method}.txt", "w") as f:
        f.write("\n".join(steps))
    print(f"\nTotal time: {time.time() - t0:.0f}s. Results in ./{OUT}/")


if __name__ == "__main__":
    main()
