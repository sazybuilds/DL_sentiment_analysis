# Q-Learning Grid World walkthrough

This folder contains a small, runnable reinforcement learning demo. A tabular
Q-learning agent learns to move from the lower-left corner of a grid to the
goal, avoiding hazard cells along the way. Start the local web server, then use
the browser controls to watch the agent explore and improve.

The learning logic uses only Python's standard library. The web service uses
FastAPI and Uvicorn, which are already listed in the repository's
`requirements.txt`. No GPU, pretrained model, dataset download, or external
service is needed.

## Run it

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn RL_model.src.backend.main:app --reload
```

If you already use the repository's `venv` folder, activate it with
`.\venv\Scripts\Activate.ps1` instead. Open
[http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser.

Use **Start learning** to let the agent run continuously, **Pause learning** to
stop it, **Step once** to advance exactly one action, and **Reset** to clear its
learned Q-table and start a fresh run. The speed slider changes the animation
delay. The API is documented at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## The reinforcement learning loop

The agent is a learner, not a pre-programmed route. It repeatedly observes its
position, selects an action, receives a reward from the environment, and updates
its estimate of how useful that action is.

### Environment and actions

`src/backend/services/agent.py` defines a 6-by-6 grid. The agent starts at the
bottom-left. It can move **up**, **right**, **down**, or **left**. The goal gives
`+1`, a hazard gives `-1`, and an ordinary move costs `-0.04`. Bumping into a
wall leaves the agent in place and still costs `-0.04`. Reaching either the
goal or a hazard ends the episode; the next action starts a new one.

The small cost for each move teaches the agent to prefer a short route instead
of wandering indefinitely.

### Q-table and Q-learning update

The Q-table is a Python dictionary with one entry per grid position. Each entry
contains four values, one for each action. A Q-value estimates the long-term
return expected from taking that action in that position.

After each move, the agent applies this update:

```text
Q(state, action) += learning_rate *
    (reward + discount_factor * best_next_Q - Q(state, action))
```

`learning_rate` (`0.2`) controls how much a new experience changes the old
estimate. `discount_factor` (`0.95`) controls how much future rewards matter.
For terminal transitions, `best_next_Q` is zero because there is no next state
in the episode. This is the core of **Q-learning**: update the value of the
action that was just taken toward the reward plus the best estimated future
value.

### Exploration and exploitation

Action selection uses epsilon-greedy behavior. With probability `epsilon`, the
agent explores by choosing a random action. Otherwise, it exploits by choosing
the action with the highest current Q-value. Random tie-breaking gives
previously unseen actions a chance instead of always favoring the first one.

Epsilon starts at `0.30` and decays by `0.97` after each episode, down to a
minimum of `0.03`. Early episodes therefore explore more; later episodes use
what has been learned more often. Exploration never completely stops.

## Files and what each part does

| File | Purpose |
| --- | --- |
| `src/backend/services/agent.py` | Defines the grid, rewards, Q-table, epsilon-greedy action selection, learning update, episode handling, and state returned to the UI. |
| `src/backend/api/routes.py` | Exposes `GET /api/state`, `POST /api/step`, and `POST /api/reset` for the browser and API clients. |
| `src/backend/main.py` | Creates the FastAPI application, serves the HTML page, mounts static assets, and registers the API routes. |
| `src/frontend/index.html` | Defines the dashboard: grid, learning controls, Q-values, training statistics, and reward chart. |
| `src/frontend/styles.css` | Styles the dark, responsive dashboard and its grid cells, controls, cards, and chart. |
| `src/frontend/app.js` | Calls the API, redraws the current policy and agent position, manages the run/pause/step/reset controls, and plots episode rewards. |

### How the UI shows learning

The grid's arrows show the best action currently known for each safe cell. The
agent marker shows its current position; goal and hazard cells have distinct
colors. The Q-value panel shows all four action values for the current cell.
Episode and step counters update as it moves. The reward chart records the
most recent completed episodes, making successes and hazard penalties visible.

The browser only displays state sent by the API. It does not calculate the
policy or train a second copy of the agent.

## API quick reference

| Method and path | Behavior |
| --- | --- |
| `GET /api/state` | Read the current position, policy, Q-values, counters, and reward history without changing the model. |
| `POST /api/step` | Take one action and update the Q-table. |
| `POST /api/reset` | Reset the environment and clear all learned Q-values. |

For example, from PowerShell:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/step -Method Post
```

## Limitations

This intentionally small example uses a deterministic grid and a Q-table rather
than a neural network. It is designed to make the real RL feedback loop visible
and understandable, not to solve large or continuous environments. The Q-table
exists in server memory and is cleared by **Reset** or when the server restarts.
