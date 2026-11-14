const grid = document.querySelector("#grid");
const runButton = document.querySelector("#run-button");
const runIcon = document.querySelector("#run-icon");
const runLabel = document.querySelector("#run-label");
const stepButton = document.querySelector("#step-button");
const resetButton = document.querySelector("#reset-button");
const speed = document.querySelector("#speed");
const speedLabel = document.querySelector("#speed-label");
const statusMessage = document.querySelector("#status-message");

const arrows = { up: "↑", right: "→", down: "↓", left: "←" };
let running = false;
let requestPending = false;

async function request(path) {
  const response = await fetch(`/api/${path}`, {
    method: path === "state" ? "GET" : "POST",
  });
  if (!response.ok) {
    throw new Error(`The server returned ${response.status}.`);
  }
  return response.json();
}

function render(state) {
  const [agentRow, agentCol] = state.position;
  const [goalRow, goalCol] = state.goal;
  const start = state.start.join(",");
  const hazards = new Set(state.hazards.map((cell) => cell.join(",")));
  grid.replaceChildren();

  state.policy.forEach((policyRow, row) => {
    policyRow.forEach((action, col) => {
      const coordinate = `${row},${col}`;
      const cell = document.createElement("div");
      cell.className = "cell";
      if (coordinate === start) cell.classList.add("start");
      if (row === goalRow && col === goalCol) {
        cell.classList.add("goal");
        cell.textContent = "★";
        cell.setAttribute("aria-label", "Goal");
      } else if (hazards.has(coordinate)) {
        cell.classList.add("hazard");
        cell.textContent = "×";
        cell.setAttribute("aria-label", "Hazard");
      } else if (row === agentRow && col === agentCol) {
        cell.classList.add("agent");
        cell.textContent = "●";
        cell.setAttribute("aria-label", "Agent");
      } else {
        cell.textContent = arrows[action] ?? "·";
        cell.setAttribute("aria-label", `Best known action: ${action ?? "none"}`);
      }
      grid.append(cell);
    });
  });

  document.querySelector("#state-coordinate").textContent =
    `Row ${agentRow + 1} · Col ${agentCol + 1}`;
  document.querySelector("#episode").textContent = state.episode;
  document.querySelector("#steps").textContent = state.total_steps;
  document.querySelector("#epsilon").textContent = `${Math.round(state.epsilon * 100)}%`;

  const reward = document.querySelector("#reward");
  if (state.reward === null) {
    reward.textContent = "—";
    reward.className = "";
  } else {
    reward.textContent = state.reward > 0 ? `+${state.reward.toFixed(2)}` : state.reward.toFixed(2);
    reward.className = state.reward > 0 ? "positive" : "negative";
  }

  const qValues = document.querySelector("#q-values");
  qValues.replaceChildren();
  state.q_values.forEach((value, index) => {
    const item = document.createElement("span");
    item.className = "q-value";
    item.title = ["Up", "Right", "Down", "Left"][index];
    item.textContent = `${["↑", "→", "↓", "←"][index]} ${value.toFixed(2)}`;
    qValues.append(item);
  });

  const rewards = state.reward_history;
  const points = rewards.map((value, index) => {
    const x = rewards.length === 1 ? 180 : 4 + index * (352 / (rewards.length - 1));
    const y = 56 - Math.max(-3, Math.min(3, value)) * 16;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  document.querySelector("#reward-line").setAttribute("points", points.join(" "));
  document.querySelector("#episode-count").textContent =
    rewards.length ? `${rewards.length} recent` : "No episodes yet";

  if (state.terminal) {
    statusMessage.textContent = state.reward > 0
      ? "Goal reached! The agent received a positive reward."
      : "Hazard hit. The agent received a penalty and will try again.";
  } else if (state.action) {
    statusMessage.textContent =
      `Moved ${state.action}. ${state.reward < 0 ? "Small step cost." : ""}`;
  }
}

function setRunning(value) {
  running = value;
  runButton.classList.toggle("is-running", value);
  runIcon.textContent = value ? "Ⅱ" : "▶";
  runLabel.textContent = value ? "Pause learning" : "Start learning";
  statusMessage.textContent = value ? "Agent is exploring and updating its Q-table…" : "Learning paused.";
}

async function oneStep() {
  if (requestPending) return;
  requestPending = true;
  stepButton.disabled = true;
  try {
    render(await request("step"));
  } catch (error) {
    setRunning(false);
    statusMessage.textContent = `Could not take a step: ${error.message}`;
  } finally {
    requestPending = false;
    stepButton.disabled = false;
  }
}

async function runLoop() {
  while (running) {
    await oneStep();
    await new Promise((resolve) => setTimeout(resolve, Number(speed.value)));
  }
}

runButton.addEventListener("click", () => {
  setRunning(!running);
  if (running) void runLoop();
});
stepButton.addEventListener("click", oneStep);

resetButton.addEventListener("click", async () => {
  setRunning(false);
  try {
    render(await request("reset"));
    statusMessage.textContent = "World and Q-table reset. Ready to learn.";
  } catch (error) {
    statusMessage.textContent = `Could not reset: ${error.message}`;
  }
});

speed.addEventListener("input", () => {
  const delay = Number(speed.value);
  speedLabel.textContent = delay < 180 ? "Fast" : delay > 480 ? "Slow" : "Normal";
});

request("state")
  .then(render)
  .catch((error) => {
    statusMessage.textContent = `Could not connect to the learning service: ${error.message}`;
  });
