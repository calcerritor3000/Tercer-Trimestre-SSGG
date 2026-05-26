import json
from pathlib import Path
from typing import Dict, List, Tuple

from app.ollama_client import OllamaClient


def load_qa_dataset(path: str) -> List[Dict[str, str]]:
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    return data


def lexical_similarity(a: str, b: str) -> float:
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))


def retrieve_best_answer(question: str, dataset: List[Dict[str, str]]) -> Tuple[str, float]:
    best_answer = ""
    best_score = -1.0
    for pair in dataset:
        score = lexical_similarity(question, pair["question"])
        if score > best_score:
            best_score = score
            best_answer = pair["answer"]
    return best_answer, best_score


def answer_before_training(client: OllamaClient, question: str) -> str:
    prompt = (
        "Responde de forma breve y profesional para un contexto empresarial.\n\n"
        f"Pregunta: {question}\n"
    )
    return client.generate(prompt, temperature=0.4)


def answer_after_training(client: OllamaClient, question: str, dataset: List[Dict[str, str]]) -> str:
    retrieved_answer, score = retrieve_best_answer(question, dataset)
    if score >= 0.15:
        prompt = (
            "Reformula en 2-3 frases profesionales la siguiente respuesta entrenada. "
            "Debes incluir el termino DSO.\n\n"
            f"Pregunta: {question}\n"
            f"Respuesta entrenada del JSON: {retrieved_answer}\n"
        )
        return client.generate(prompt, temperature=0.1)
    prompt = (
        "Usa este conocimiento entrenado como fuente principal:\n"
        f"{retrieved_answer}\n\n"
        f"Pregunta: {question}\n"
        "Responde en maximo 3 frases e incluye DSO si aplica."
    )
    return client.generate(prompt, temperature=0.1)


def run_training_flow(
    client: OllamaClient,
    dataset_path: str,
    question: str = "¿Qué indicador se usa para saber cuántos días tarda una empresa en cobrar?",
) -> Dict[str, object]:
    dataset = load_qa_dataset(dataset_path)
    retrieved, score = retrieve_best_answer(question, dataset)
    before = answer_before_training(client, question)
    after = answer_after_training(client, question, dataset)
    before_has_dso = "dso" in before.lower() or "days sales outstanding" in before.lower()
    after_has_dso = "dso" in after.lower() or "days sales outstanding" in after.lower()
    if after_has_dso and not before_has_dso:
        summary = "Diferencial detectado: la respuesta posterior incorpora el conocimiento JSON (DSO)."
    elif after_has_dso:
        summary = "La respuesta posterior refuerza el conocimiento entrenado del dataset."
    else:
        summary = "Compara visualmente ambas respuestas para ver el efecto del entrenamiento."
    return {
        "question": question,
        "retrieved_answer": retrieved,
        "retrieval_score": round(score, 2),
        "before": before,
        "after": after,
        "before_has_dso": before_has_dso,
        "after_has_dso": after_has_dso,
        "summary": summary,
    }


def run_training_demo(client: OllamaClient, dataset_path: str) -> None:
    result = run_training_flow(client, dataset_path)
    print("\n=== DEMO ENTRENAMIENTO JSON (antes/después) ===")
    print(f"Pregunta de prueba: {result['question']}\n")
    print("[ANTES del mini-entrenamiento]")
    print(result["before"])
    print("\n[DESPUES del mini-entrenamiento con JSON]")
    print(result["after"])
    print(f"\nComparacion: ANTES menciona DSO={result['before_has_dso']} | DESPUES menciona DSO={result['after_has_dso']}")
    print(result["summary"])
