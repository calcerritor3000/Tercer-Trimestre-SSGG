from typing import Any, Dict, List, Optional

from app.minisaas_mcp import MiniSaaS
from app.ollama_client import OllamaClient

VALID_ACTIONS = ("query_low_stock", "query_sales_total", "write_report", "finish")

AGENT_SYSTEM_PROMPT = (
    "Eres un agente empresarial autonomo. Cumple la mision paso a paso.\n"
    "Devuelve SOLO JSON valido con esta forma exacta:\n"
    '{"thought":"razon breve","action":"query_low_stock","params":{"threshold":10}}\n'
    "El campo action debe ser UNA sola palabra de esta lista (sin barras ni combinaciones):\n"
    "- query_low_stock\n"
    "- query_sales_total\n"
    "- write_report\n"
    "- finish\n"
    "Flujo recomendado: primero query_low_stock, luego query_sales_total, "
    "luego write_report y por ultimo finish.\n"
)


def normalize_action(action: Optional[str]) -> Optional[str]:
    if not action:
        return None
    text = str(action).strip().lower()
    if "low_stock" in text or "list_low_stock" in text:
        return "query_low_stock"
    if "sales" in text:
        return "query_sales_total"
    if "report" in text:
        return "write_report"
    if text == "finish":
        return "finish"
    if text in VALID_ACTIONS:
        return text
    return None


def next_planned_action(memory: List[Dict[str, Any]], report_created: bool) -> Optional[str]:
    queries = {item.get("query") for item in memory}
    if "low_stock" not in queries:
        return "query_low_stock"
    if "sales_total" not in queries:
        return "query_sales_total"
    if not report_created:
        return "write_report"
    return "finish"


def run_agentic_flow(client: OllamaClient, mission: str, max_steps: int = 8) -> Dict[str, Any]:
    saas = MiniSaaS()
    memory: List[Dict[str, Any]] = []
    steps_log: List[Dict[str, Any]] = []
    report_text = ""
    report_created = False

    for step in range(1, max_steps + 1):
        prompt = (
            f"Mision: {mission}\n"
            f"Paso actual: {step}\n"
            f"Memoria: {memory}\n"
            "Elige la siguiente accion (una sola)."
        )
        raw = client.generate(prompt, system=AGENT_SYSTEM_PROMPT, format_json=True, temperature=0.0)
        try:
            decision = client.extract_json(raw)
        except Exception:
            decision = {}

        action = normalize_action(decision.get("action"))
        params = decision.get("params", {}) or {}

        fallback = action is None
        if fallback:
            action = next_planned_action(memory, report_created)

        step_entry: Dict[str, Any] = {
            "step": step,
            "action": action,
            "decision": decision,
            "fallback": fallback,
        }

        if action == "query_low_stock":
            threshold = int(params.get("threshold", 10))
            result = saas.execute({"action": "list_low_stock", "params": {"threshold": threshold}})
            memory.append({"step": step, "query": "low_stock", "result": result})
            step_entry["result"] = result
            steps_log.append(step_entry)
            continue

        if action == "query_sales_total":
            result = saas.execute({"action": "sales_total", "params": {}})
            memory.append({"step": step, "query": "sales_total", "result": result})
            step_entry["result"] = result
            steps_log.append(step_entry)
            continue

        if action == "write_report":
            if report_created:
                step_entry["note"] = "Informe ya generado, finalizando."
                steps_log.append(step_entry)
                break
            report_prompt = (
                "Eres un analista de negocio. Redacta un informe ejecutivo breve "
                "con exactamente 3 recomendaciones numeradas y prioritarias.\n"
                f"Mision: {mission}\n"
                f"Datos recolectados: {memory}\n"
            )
            report_text = client.generate(report_prompt, temperature=0.2)
            report_created = True
            memory.append({"step": step, "report_created": True})
            step_entry["report"] = report_text
            steps_log.append(step_entry)
            continue

        if action == "finish":
            step_entry["note"] = "Mision finalizada."
            steps_log.append(step_entry)
            break

        step_entry["error"] = f"accion desconocida: {action}"
        steps_log.append(step_entry)

    return {
        "mission": mission,
        "steps": steps_log,
        "memory": memory,
        "report": report_text,
        "success": bool(report_text),
    }


def run_agentic_mission(client: OllamaClient, mission: str, max_steps: int = 8) -> None:
    print("\n=== DEMO IA AGENTICA (autonoma, no conversacional) ===")
    print(f"Mision: {mission}\n")
    result = run_agentic_flow(client, mission, max_steps)
    for step in result["steps"]:
        label = "plan autonomo de respaldo" if step.get("fallback") else "decision IA"
        print(f"Paso {step['step']} ({label}): {step.get('decision') or step.get('action')}")
        if step.get("report"):
            print("\n[REPORTE DEL AGENTE]")
            print(step["report"])
    if result["report"]:
        print("\n[REPORTE FINAL]")
        print(result["report"])
    else:
        print("\nNo se genero reporte final.")
