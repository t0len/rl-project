# Drone delivery with battery and wind: tabular Q-learning (CSCI3235 Midterm 1)

## Setup
Python 3.10+ (tested on 3.12).
    pip install -r requirements.txt

## Run (one command)
    python run.py                 # Q-learning, 5000 episodes, 10 seeds, ~35 s
    python run.py --method sarsa  # optional: SARSA for comparison
    python run.py --quick         # smoke test, a few seconds

## Seeds
Training seeds 0..9 (seed s seeds the environment and the agent).
Evaluation uses separate environment seeds 10000+s (200 greedy episodes, no learning, epsilon=0).

## Output (folder results/)
- fig1_training_*.png    training curve (mean ± std over seeds)
- fig2_comparison_*.png  final comparison: agent vs baselines vs value iteration
- fig3_epsilon_*.png     epsilon study (0.01 / 0.1 / 0.3)
- tables_*.md            Table 1 (comparison), Table 2 (epsilon study)
- example_trajectory_*.txt  one greedy episode (s, a, r, s') for the MDP step example

## Files
- env.py        MDP: states, actions, rewards, explicit transition model transitions(s, a)
- baselines.py  random, battery-blind rule, fuel-aware rule, value iteration (non-RL reference)
- agent.py      tabular agent (Q-learning / SARSA), train(), evaluate()
- run.py        all experiments, tables and figures

## References
Sutton & Barto, Reinforcement Learning: An Introduction, 2nd ed. (Q-learning, Sec. 6.5).
No external environment code was used; the environment is written from scratch.
