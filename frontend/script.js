// Change this if your Flask API runs on a different host/port.
const API_URL = "http://localhost:5000/predict";

document.getElementById("apiUrlDisplay").textContent = API_URL;

const articleInput = document.getElementById("articleInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusText = document.getElementById("statusText");

const resultCard = document.getElementById("resultCard");
const resultLabel = document.getElementById("resultLabel");
const resultConfidence = document.getElementById("resultConfidence");
const barReal = document.getElementById("barReal");
const barFake = document.getElementById("barFake");

const historySection = document.getElementById("historySection");
const historyList = document.getElementById("historyList");

let history = [];

analyzeBtn.addEventListener("click", async () => {
  const text = articleInput.value.trim();
  if (!text) {
    statusText.textContent = "Please paste some text first.";
    return;
  }

  analyzeBtn.disabled = true;
  statusText.textContent = "Analyzing...";
  resultCard.classList.add("hidden");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();

    if (!response.ok) {
      statusText.textContent = data.error || "Something went wrong.";
      return;
    }

    renderResult(data, text);
    statusText.textContent = "";
  } catch (err) {
    statusText.textContent =
      "Could not reach the API. Is the Flask server running on " + API_URL + "?";
    console.error(err);
  } finally {
    analyzeBtn.disabled = false;
  }
});

function renderResult(data, originalText) {
  const isReal = data.label.toUpperCase() === "REAL";

  resultLabel.textContent = isReal ? "✅ Likely Real" : "⚠️ Likely Fake";
  resultLabel.className = "result-label " + (isReal ? "real" : "fake");
  resultConfidence.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;

  const realProb = (data.probabilities.REAL ?? 0) * 100;
  const fakeProb = (data.probabilities.FAKE ?? 0) * 100;
  barReal.style.width = `${realProb}%`;
  barFake.style.width = `${fakeProb}%`;

  resultCard.classList.remove("hidden");

  addToHistory(originalText, data.label, data.confidence);
}

function addToHistory(text, label, confidence) {
  history.unshift({ text, label, confidence });
  history = history.slice(0, 5); // keep last 5

  historyList.innerHTML = "";
  history.forEach((item) => {
    const li = document.createElement("li");
    const snippet = item.text.length > 60 ? item.text.slice(0, 60) + "..." : item.text;
    li.innerHTML = `
      <span class="snippet">${escapeHtml(snippet)}</span>
      <strong style="color:${item.label.toUpperCase() === "REAL" ? "#1f9d55" : "#d9364b"}">
        ${item.label.toUpperCase()} (${(item.confidence * 100).toFixed(0)}%)
      </strong>
    `;
    historyList.appendChild(li);
  });

  historySection.classList.remove("hidden");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
