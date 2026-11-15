# Learning Agent Goal

The agent's goal is to learn how to travel from the bottom-left starting cell
to the top-right goal cell while avoiding hazard cells. It is not given a
predefined route: it discovers useful moves by trying actions and learning
from their rewards.

Reaching the goal earns `+1`, entering a hazard earns `-1`, and each ordinary
move costs `-0.04`. These rewards encourage the agent to find a short, safe
route. The Q-table records how valuable each action appears to be from each
cell. Over repeated episodes, the agent uses those values to choose better
moves while still occasionally exploring alternatives.

The demo succeeds when the agent learns a policy that reaches the goal more
reliably and avoids unnecessary steps and hazards. Its purpose is to make the
reinforcement learning process visible and understandable; it is a small
teaching example, not a general-purpose navigation system.
