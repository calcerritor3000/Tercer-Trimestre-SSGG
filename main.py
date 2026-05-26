import argparse

from app.agentic_mission import run_agentic_mission
from app.minisaas_mcp import run_mcp_demo
from app.ollama_client import OllamaClient
from app.training_demo import run_training_demo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Proyecto examen SSGG: Ollama + entrenamiento JSON + MCP + IA agéntica"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["model", "training", "mcp", "agentic"],
        help="Demo a ejecutar",
    )
    parser.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Modelo en Ollama (ej: llama3.1:8b, mistral:7b, phi3:mini)",
    )
    parser.add_argument(
        "--question",
        default="Dime que productos tienen poco stock por debajo de 10 unidades",
        help="Pregunta humana para demo MCP",
    )
    parser.add_argument(
        "--mission",
        default="Analiza stock y ventas y redacta un informe con 3 recomendaciones prioritarias.",
        help="Mision para la IA agentica",
    )
    parser.add_argument(
        "--dataset",
        default="data/train_qa.json",
        help="Ruta al JSON de pares pregunta/respuesta",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="URL base de Ollama",
    )
    return parser.parse_args()


def run_model_justification_demo(client: OllamaClient):
    print("\n=== DEMO MODELO OLLAMA ===")
    print(f"Modelo configurado: {client.model}")
    print(f"Servidor: {client.base_url}")
    print("Cargando modelo en memoria (la 1ª vez puede tardar 1-3 min)...")
    prompt = "Explica en 4 lineas qué es un ERP para una pyme."
    answer = client.generate(prompt, temperature=0.2)
    print(f"Prompt: {prompt}")
    print(f"Respuesta del modelo:\n{answer}")


def main():
    args = parse_args()
    client = OllamaClient(model=args.model, base_url=args.base_url)

    if not client.healthcheck():
        print("No se pudo conectar con Ollama. Verifica que esté levantado en el puerto 11434.")
        print("Ejemplo: ollama serve")
        return

    if args.mode == "model":
        run_model_justification_demo(client)
    elif args.mode == "training":
        run_training_demo(client, args.dataset)
    elif args.mode == "mcp":
        run_mcp_demo(client, args.question)
    elif args.mode == "agentic":
        run_agentic_mission(client, args.mission)


if __name__ == "__main__":
    main()
