# Tercer-Trimestre-SSGG — IA aplicada a gestión empresarial

Este proyecto cubre los 4 apartados pedidos:

1. Uso de **Ollama + modelo preestablecido** (local o remoto).
2. **Mini-entrenamiento** usando pares pregunta/respuesta en JSON y comparación antes/después.
3. Patrón tipo **MCP**: humano -> petición estructurada -> SaaS -> interpretación IA.
4. **IA agéntica** no conversacional: recibe una misión y actúa de forma autónoma.

## Requisitos

- Python 3.10+
- Ollama instalado y ejecutándose (`ollama serve`)
- Tener descargado al menos un modelo, por ejemplo:
  - `ollama pull llama3.1:8b`
  - `ollama pull mistral:7b`
  - `ollama pull phi3:mini`

Instalación:

```bash
pip install -r requirements.txt
```

## Interfaz web (recomendado para el examen)

```bash
pip install -r requirements.txt
python web_app.py
```

Abre en el navegador: **http://127.0.0.1:5000**

Tendrás 4 pestañas: Modelo Ollama, Entrenamiento JSON, MCP MiniSaaS e IA Agéntica.

Requisito: Ollama debe estar ejecutándose (`ollama serve` o app Ollama abierta).

## Ejecución por terminal

### 1) Modelo Ollama y justificación técnica

```bash
python main.py --mode model --model llama3.1:8b
```

### 2) Entrenamiento mínimo con JSON (antes/después)

```bash
python main.py --mode training --model llama3.1:8b
```

### 3) MiniSaaS estilo MCP

```bash
python main.py --mode mcp --model llama3.1:8b --question "¿Qué productos están por debajo de 10 unidades de stock?"
```

### 4) IA agéntica autónoma

```bash
python main.py --mode agentic --model llama3.1:8b --mission "Analiza stock y ventas y redacta 3 acciones concretas para mejorar la operación."
```

## Qué modelo usar y por qué

Recomendación para examen: **`llama3.1:8b`** como modelo base.

Motivos (potencia, rapidez, memoria, generación):

- **Potencia**: 8B parámetros permiten buen razonamiento general para tareas empresariales (interpretación, resumen, transformación a JSON).
- **Rapidez**: más rápido que modelos de 13B+ en CPU/GPU de equipo personal, con latencia razonable para demos.
- **Memoria**: coste de RAM/VRAM moderado comparado con modelos grandes; es más viable en un entorno local.
- **Calidad de generación**: suele mantener buen equilibrio entre coherencia y precisión para prompts estructurados.

Alternativas:

- **`mistral:7b`**: normalmente rápido y ligero, buena opción si el equipo es más limitado.
- **`phi3:mini`**: muy eficiente en memoria/velocidad, útil para hardware modesto, pero con menor profundidad de respuesta.

## Cómo defender el apartado de entrenamiento

En `data/train_qa.json` hay pares pregunta/respuesta de negocio.

El script `--mode training`:

1. Hace una pregunta sin conocimiento específico cargado (**ANTES**).
2. Recupera el par más cercano del JSON y lo inyecta como conocimiento entrenado (**DESPUÉS**).
3. Muestra diferencia de calidad/precisión entre ambas respuestas.

Nota técnica: esto es un **entrenamiento mínimo por adaptación en contexto (few-shot + recuperación)**, no fine-tuning de pesos. Para el objetivo docente de demostrar mejora antes/después es válido y observable.

## Arquitectura MCP del MiniSaaS

En `app/minisaas_mcp.py`:

1. Usuario hace pregunta humana.
2. IA la traduce a petición JSON estructurada (`action`, `params`).
3. MiniSaaS ejecuta la acción.
4. IA interpreta la respuesta técnica y la devuelve en lenguaje natural.

Esto demuestra claramente el concepto de conexión entre IA y aplicación de negocio.

## IA agéntica

En `app/agentic_mission.py` el agente:

- Recibe una misión.
- Decide acciones en bucle (`query_low_stock`, `query_sales_total`, `write_report`, `finish`).
- Consulta el MiniSaaS de forma autónoma.
- Genera un informe y finaliza por sí mismo.

No es un chat guiado paso a paso; actúa orientado a objetivo.

## Estructura

- `main.py`: punto de entrada por modos.
- `app/ollama_client.py`: cliente de Ollama.
- `app/training_demo.py`: demo de entrenamiento JSON.
- `app/minisaas_mcp.py`: demo tipo MCP.
- `app/agentic_mission.py`: demo agéntica.
- `data/train_qa.json`: dataset inicial de entrenamiento.
