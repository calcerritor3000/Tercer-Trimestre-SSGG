from typing import Any, Dict, List

from app.ollama_client import OllamaClient


class MiniSaaS:
    def __init__(self):
        self.products = {
            "A100": {"name": "Portatil Pro 14", "stock": 23, "price": 1499.0},
            "B205": {"name": "Monitor 27 4K", "stock": 8, "price": 429.0},
            "C310": {"name": "Teclado Mecanico", "stock": 61, "price": 119.0},
        }
        self.sales = [
            {"order_id": 1, "product_id": "A100", "units": 2, "total": 2998.0},
            {"order_id": 2, "product_id": "B205", "units": 1, "total": 429.0},
            {"order_id": 3, "product_id": "C310", "units": 5, "total": 595.0},
        ]

    def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get("action")
        params = request.get("params", {})

        if action == "get_product":
            product_id = params.get("product_id", "")
            product = self.products.get(product_id)
            if not product:
                return {"ok": False, "error": f"Producto no encontrado: {product_id}"}
            return {"ok": True, "data": {"product_id": product_id, **product}}

        if action == "list_low_stock":
            threshold = int(params.get("threshold", 10))
            low_stock: List[Dict[str, Any]] = []
            for pid, item in self.products.items():
                if item["stock"] < threshold:
                    low_stock.append({"product_id": pid, **item})
            return {"ok": True, "data": low_stock}

        if action == "sales_total":
            total = sum(order["total"] for order in self.sales)
            return {"ok": True, "data": {"sales_total": total, "orders": len(self.sales)}}

        return {"ok": False, "error": f"Accion no soportada: {action}"}


REQUEST_SYSTEM_PROMPT = (
    "Eres un traductor de lenguaje humano a peticiones JSON para un SaaS.\n"
    "Debes devolver SOLO JSON con esta forma:\n"
    '{"action":"get_product|list_low_stock|sales_total","params":{...}}\n'
    "Para get_product usa params.product_id.\n"
    "Para list_low_stock usa params.threshold.\n"
    "Para sales_total params vacio.\n"
)


def human_to_request(client: OllamaClient, human_question: str) -> Dict[str, Any]:
    prompt = f"Convierte esta solicitud humana en peticion JSON valida:\n{human_question}"
    raw = client.generate(prompt, system=REQUEST_SYSTEM_PROMPT, format_json=True, temperature=0.0)
    return client.extract_json(raw)


def interpret_response(client: OllamaClient, human_question: str, saas_response: Dict[str, Any]) -> str:
    prompt = (
        "Explica al usuario el resultado del SaaS en lenguaje natural, breve y claro.\n\n"
        f"Pregunta original: {human_question}\n"
        f"Respuesta SaaS: {saas_response}\n"
    )
    return client.generate(prompt, temperature=0.2)


def run_mcp_flow(client: OllamaClient, human_question: str) -> Dict[str, Any]:
    saas = MiniSaaS()
    machine_request = human_to_request(client, human_question)
    saas_response = saas.execute(machine_request)
    interpreted = interpret_response(client, human_question, saas_response)
    return {
        "human_question": human_question,
        "machine_request": machine_request,
        "saas_response": saas_response,
        "interpretation": interpreted,
    }


def run_mcp_demo(client: OllamaClient, human_question: str) -> None:
    result = run_mcp_flow(client, human_question)
    print("\n=== DEMO MCP (Humano -> Peticion -> SaaS -> Interpretacion) ===")
    print(f"Pregunta humana: {result['human_question']}")
    print(f"Peticion generada por IA: {result['machine_request']}")
    print(f"Respuesta del SaaS: {result['saas_response']}")
    print(f"Interpretacion final IA: {result['interpretation']}")
