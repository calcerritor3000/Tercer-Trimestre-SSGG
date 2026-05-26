const modelSelect = document.getElementById("model-select");
const statusEl = document.getElementById("ollama-status");
const loadingEl = document.getElementById("loading");

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`panel-${btn.dataset.tab}`).classList.add("active");
  });
});

async function checkHealth() {
  const model = modelSelect.value;
  try {
    const res = await fetch(`/api/health?model=${encodeURIComponent(model)}`);
    const data = await res.json();
    if (data.ok) {
      statusEl.textContent = `Ollama conectado · modelo: ${model}`;
      statusEl.className = "status ok";
    } else {
      statusEl.textContent = "Ollama no disponible. Abre la app Ollama.";
      statusEl.className = "status err";
    }
  } catch {
    statusEl.textContent = "No se pudo conectar al servidor web.";
    statusEl.className = "status err";
  }
}

function showLoading(show) {
  loadingEl.classList.toggle("hidden", !show);
  document.querySelectorAll(".run-btn").forEach((b) => (b.disabled = show));
}

function block(title, body, cls = "") {
  return `<div class="card-block ${cls}"><strong>${title}</strong>\n${body}</div>`;
}

async function postApi(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, model: modelSelect.value }),
  });
  const data = await res.json();
  if (!res.ok || !data.ok) {
    throw new Error(data.error || "Error en la petición");
  }
  return data;
}

document.querySelectorAll(".run-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const action = btn.dataset.action;
    showLoading(true);
    try {
      if (action === "model") {
        const data = await postApi("/api/model", {
          prompt: document.getElementById("model-prompt").value,
        });
        document.getElementById("out-model").innerHTML =
          block("Modelo", data.model) +
          block("Prompt", data.prompt) +
          block("Respuesta", data.answer, "green") +
          block("Justificación técnica", data.justification, "amber");
      }

      if (action === "training") {
        const data = await postApi("/api/training", {
          question: document.getElementById("training-question").value,
        });
        document.getElementById("out-training").innerHTML =
          block("Pregunta", data.question) +
          block(
            "Conocimiento recuperado del JSON",
            `Score: ${data.retrieval_score}\n${data.retrieved_answer}`
          ) +
          block("ANTES del entrenamiento", data.before) +
          block("DESPUÉS del entrenamiento", data.after, "green") +
          block(
            "Comparación DSO",
            `Antes menciona DSO: ${data.before_has_dso}\nDespués menciona DSO: ${data.after_has_dso}\n\n${data.summary}`,
            "amber"
          );
      }

      if (action === "mcp") {
        const data = await postApi("/api/mcp", {
          question: document.getElementById("mcp-question").value,
        });
        document.getElementById("out-mcp").innerHTML =
          block("1. Pregunta humana", data.human_question) +
          block(
            "2. Petición JSON (IA)",
            JSON.stringify(data.machine_request, null, 2)
          ) +
          block(
            "3. Respuesta MiniSaaS",
            JSON.stringify(data.saas_response, null, 2)
          ) +
          block("4. Interpretación IA", data.interpretation, "green");
      }

      if (action === "agentic") {
        const data = await postApi("/api/agentic", {
          mission: document.getElementById("agentic-mission").value,
        });
        let stepsHtml = data.steps
          .map(
            (s) =>
              `Paso ${s.step}: ${s.action}` +
              (s.report ? `\n\n${s.report}` : "") +
              (s.result ? `\n${JSON.stringify(s.result)}` : "")
          )
          .join("\n\n---\n\n");
        document.getElementById("out-agentic").innerHTML =
          block("Misión", data.mission) +
          block("Pasos del agente", stepsHtml) +
          (data.report
            ? block("Informe final", data.report, "green")
            : block("Estado", "No se generó informe", "amber"));
      }
    } catch (err) {
      const out = document.getElementById(`out-${action}`);
      out.textContent = `Error: ${err.message}`;
    } finally {
      showLoading(false);
    }
  });
});

checkHealth();
modelSelect.addEventListener("change", checkHealth);
