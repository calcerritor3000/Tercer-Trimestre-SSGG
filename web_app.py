from flask import Flask, jsonify, render_template, request

from app.agentic_mission import run_agentic_flow
from app.minisaas_mcp import run_mcp_flow
from app.ollama_client import OllamaClient
from app.training_demo import run_training_flow

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    model = request.args.get("model", "llama3.1:8b")
    base_url = request.args.get("base_url", "http://localhost:11434")
    client = OllamaClient(model=model, base_url=base_url)
    ok = client.healthcheck()
    return jsonify({"ok": ok, "model": model, "base_url": base_url})


@app.route("/api/model", methods=["POST"])
def api_model():
    data = request.get_json(force=True) or {}
    client = OllamaClient(
        model=data.get("model", "llama3.1:8b"),
        base_url=data.get("base_url", "http://localhost:11434"),
    )
    if not client.healthcheck():
        return jsonify({"ok": False, "error": "Ollama no disponible en localhost:11434"}), 503

    prompt = data.get(
        "prompt",
        "Explica en 4 lineas qué es un ERP para una pyme.",
    )
    answer = client.generate(prompt, temperature=0.2)
    return jsonify(
        {
            "ok": True,
            "model": client.model,
            "prompt": prompt,
            "answer": answer,
            "justification": (
                "Modelo llama3.1:8b: buen equilibrio entre calidad, velocidad y consumo de RAM "
                "para demos locales en gestión empresarial."
            ),
        }
    )


@app.route("/api/training", methods=["POST"])
def api_training():
    data = request.get_json(force=True) or {}
    client = OllamaClient(
        model=data.get("model", "llama3.1:8b"),
        base_url=data.get("base_url", "http://localhost:11434"),
    )
    if not client.healthcheck():
        return jsonify({"ok": False, "error": "Ollama no disponible"}), 503

    result = run_training_flow(
        client,
        data.get("dataset", "data/train_qa.json"),
        data.get(
            "question",
            "¿Qué indicador se usa para saber cuántos días tarda una empresa en cobrar?",
        ),
    )
    return jsonify({"ok": True, **result})


@app.route("/api/mcp", methods=["POST"])
def api_mcp():
    data = request.get_json(force=True) or {}
    client = OllamaClient(
        model=data.get("model", "llama3.1:8b"),
        base_url=data.get("base_url", "http://localhost:11434"),
    )
    if not client.healthcheck():
        return jsonify({"ok": False, "error": "Ollama no disponible"}), 503

    question = data.get(
        "question",
        "¿Qué productos tienen poco stock por debajo de 10 unidades?",
    )
    result = run_mcp_flow(client, question)
    return jsonify({"ok": True, **result})


@app.route("/api/agentic", methods=["POST"])
def api_agentic():
    data = request.get_json(force=True) or {}
    client = OllamaClient(
        model=data.get("model", "llama3.1:8b"),
        base_url=data.get("base_url", "http://localhost:11434"),
    )
    if not client.healthcheck():
        return jsonify({"ok": False, "error": "Ollama no disponible"}), 503

    mission = data.get(
        "mission",
        "Analiza stock y ventas y redacta un informe con 3 recomendaciones prioritarias.",
    )
    result = run_agentic_flow(client, mission)
    return jsonify({"ok": True, **result})


if __name__ == "__main__":
    print("Abre en el navegador: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
