const form = document.querySelector("#review-form");
const reviewInput = document.querySelector("#review-text");
const count = document.querySelector("#character-count");
const analyzeButton = document.querySelector("#analyze-button");
const clearButton = document.querySelector("#clear-button");
const errorMessage = document.querySelector("#error-message");
const resultCard = document.querySelector("#result-card");
const modelStatus = document.querySelector("#model-status");
const statusLabel = document.querySelector("#status-label");
const MAX_CHARACTERS = 10000;

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
  resultCard.hidden = true;
}

function clearError() {
  errorMessage.textContent = "";
  errorMessage.hidden = true;
}

function setModelStatus(state, message) {
  modelStatus.className = `model-status ${state}`;
  statusLabel.textContent = message;
}

async function checkModelStatus() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error(`The local API returned ${response.status}.`);
    }

    const health = await response.json();
    if (health.model_loaded) {
      setModelStatus("ready", "Model ready");
    } else if (health.checkpoint_available) {
      setModelStatus("missing", "Checkpoint found; loading on first review");
    } else {
      setModelStatus("missing", "Train the model to create its checkpoint");
    }
  } catch {
    setModelStatus("offline", "Local API is offline");
  }
}

function updateCharacterCount() {
  count.textContent = `${reviewInput.value.length.toLocaleString()} / ${MAX_CHARACTERS.toLocaleString()}`;
}

for (const button of document.querySelectorAll("[data-review]")) {
  button.addEventListener("click", () => {
    reviewInput.value = button.dataset.review;
    updateCharacterCount();
    reviewInput.focus();
    clearError();
  });
}

reviewInput.addEventListener("input", updateCharacterCount);

clearButton.addEventListener("click", () => {
  reviewInput.value = "";
  updateCharacterCount();
  clearError();
  resultCard.hidden = true;
  reviewInput.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();

  const text = reviewInput.value.trim();
  if (!text) {
    showError("Write or paste a review before asking the model to analyze it.");
    reviewInput.focus();
    return;
  }

  analyzeButton.disabled = true;
  analyzeButton.querySelector("span").textContent = "Reading review...";
  resultCard.hidden = true;

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || `The local API returned ${response.status}.`);
    }

    const positive = data.class_id === 1;
    const confidence = Math.round(data.probability * 1000) / 10;

    resultCard.classList.toggle("negative", !positive);
    document.querySelector("#result-label").textContent = data.prediction;
    document.querySelector("#result-description").textContent = positive
      ? "The review leans positive."
      : "The review leans negative.";
    document.querySelector("#result-icon").textContent = positive ? "+" : "-";
    document.querySelector("#confidence-value").textContent = `${confidence.toFixed(1)}%`;
    document.querySelector("#confidence-bar").style.width = `${confidence}%`;

    const confidenceTrack = document.querySelector("#confidence-track");
    confidenceTrack.setAttribute("aria-valuenow", confidence.toFixed(1));
    resultCard.hidden = false;
    await checkModelStatus();
  } catch (error) {
    showError(error instanceof Error ? error.message : "The prediction could not be completed.");
    await checkModelStatus();
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.querySelector("span").textContent = "Analyze review";
  }
});

updateCharacterCount();
checkModelStatus();
