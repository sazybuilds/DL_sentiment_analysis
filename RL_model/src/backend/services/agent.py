import random

Position = tuple[int, int]
Action = int

ROWS = 6
COLS = 6
START: Position = (5, 0)
GOAL: Position = (0, 5)
HAZARDS: set[Position] = {(1, 1), (2, 3), (4, 2), (4, 4)}
ACTIONS: tuple[tuple[str, int, int], ...] = (
    ("up", -1, 0),
    ("right", 0, 1),
    ("down", 1, 0),
    ("left", 0, -1),
)

LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.95
INITIAL_EPSILON = 0.3
MIN_EPSILON = 0.03
EPSILON_DECAY = 0.97


class QLearningAgent:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> dict:
        self.q_table: dict[Position, list[float]] = {
            (row, col): [0.0] * len(ACTIONS)
            for row in range(ROWS)
            for col in range(COLS)
        }
        self.position = START
        self.episode = 1
        self.steps = 0
        self.episode_reward = 0.0
        self.epsilon = INITIAL_EPSILON
        self.reward_history: list[float] = []
        self.total_steps = 0
        return self.state()

    def step(self) -> dict:
        if self.position in (GOAL,) or self.position in HAZARDS:
            self.position = START
            self.steps = 0
            self.episode_reward = 0.0

        old_position = self.position
        action = self._choose_action(old_position)
        _, row_delta, col_delta = ACTIONS[action]
        next_position = (
            min(ROWS - 1, max(0, old_position[0] + row_delta)),
            min(COLS - 1, max(0, old_position[1] + col_delta)),
        )

        if next_position == GOAL:
            reward = 1.0
            terminal = True
        elif next_position in HAZARDS:
            reward = -1.0
            terminal = True
        else:
            reward = -0.04
            terminal = False

        next_best = 0.0 if terminal else max(self.q_table[next_position])
        old_value = self.q_table[old_position][action]
        self.q_table[old_position][action] = old_value + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * next_best - old_value
        )

        self.position = next_position
        self.steps += 1
        self.total_steps += 1
        self.episode_reward += reward

        if terminal:
            self.reward_history.append(round(self.episode_reward, 2))
            self.reward_history = self.reward_history[-60:]
            self.episode += 1
            self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

        return self.state(reward=reward, action=ACTIONS[action][0], terminal=terminal)

    def state(
        self,
        reward: float | None = None,
        action: str | None = None,
        terminal: bool = False,
    ) -> dict:
        policy = []
        for row in range(ROWS):
            policy_row = []
            for col in range(COLS):
                position = (row, col)
                values = self.q_table[position]
                best_action = max(range(len(values)), key=values.__getitem__)
                policy_row.append(
                    None if position == GOAL or position in HAZARDS else ACTIONS[best_action][0]
                )
            policy.append(policy_row)

        return {
            "rows": ROWS,
            "cols": COLS,
            "start": START,
            "goal": GOAL,
            "hazards": sorted(HAZARDS),
            "position": self.position,
            "episode": self.episode,
            "steps": self.steps,
            "total_steps": self.total_steps,
            "epsilon": round(self.epsilon, 3),
            "episode_reward": round(self.episode_reward, 2),
            "reward": reward,
            "action": action,
            "terminal": terminal,
            "q_values": [round(value, 3) for value in self.q_table[self.position]],
            "policy": policy,
            "reward_history": self.reward_history,
        }

    def _choose_action(self, position: Position) -> Action:
        if random.random() < self.epsilon:
            return random.randrange(len(ACTIONS))

        values = self.q_table[position]
        highest_value = max(values)
        best_actions = [index for index, value in enumerate(values) if value == highest_value]
        return random.choice(best_actions)


agent = QLearningAgent()
