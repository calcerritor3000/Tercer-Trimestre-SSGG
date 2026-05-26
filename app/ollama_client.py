import json
import time
from typing import Any, Dict, Optional

import requests


class OllamaClient:
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def healthcheck(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        format_json: bool = False,
        temperature: float = 0.2,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if format_json:
            payload["format"] = "json"

        last_error = ""
        for attempt in range(1, 4):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=300,
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "").strip()

                last_error = response.text[:500] or response.reason
                if response.status_code == 500 and attempt < 3:
                    print(
                        f"Ollama aún carga el modelo (intento {attempt}/3). "
                        "Espera 15 s y reintenta..."
                    )
                    time.sleep(15)
                    continue
                response.raise_for_status()
            except requests.Timeout:
                last_error = "Tiempo de espera agotado (el modelo tarda en cargar la primera vez)."
                if attempt < 3:
                    print(f"Timeout (intento {attempt}/3). Reintentando...")
                    time.sleep(10)
                    continue
                raise

        raise RuntimeError(
            f"Ollama devolvió error al generar con '{self.model}'. Detalle: {last_error}\n"
            "Prueba: cerrar otras apps, reiniciar Ollama, o usar un modelo más ligero "
            "(phi3:mini / mistral:7b)."
        )

    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(text[start : end + 1])
