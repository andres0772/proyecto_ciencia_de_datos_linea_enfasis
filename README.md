# Taller: Buenas Prácticas en Ciencia de Datos e IA

Proyecto completo del taller (ver `docs/GUIA.md`): **EDA del Titanic** + **modelo de ML**
para predecir supervivencia, construido como un pipeline reproducible con arquitectura
**medallion** (Bronze → Silver → Gold) y guardado auditable del modelo.

## Datos
Nombre: Andres Esteban Vasquez Peña

Materia: Linea de enfasis 3

## Requisitos

- Python ≥ 3.11 (se recomienda 3.12), `make`, `git`
- [`uv`](https://docs.astral.sh/uv/) para el entorno virtual

## Instalación (con uv)

```bash
cd proyecto_ciencia_de_datos_linea_enfasis
uv venv --python 3.12
source .venv/bin/activate          # fish: source .venv/bin/activate.fish
uv pip install -e ".[dev]"
cp .env.example .env
```

## Ejecución

```bash
make pipeline    # ingesta -> silver -> gold -> modelo entrenado y guardado
make test        # pruebas automáticas (pytest)
make lint        # ruff
```

Por etapas: `make ingest`, `make silver`, `make gold`, `make train`.
Si la URL de datos caduca, descargar el CSV fuera del repo y poner su ruta en
`TITANIC_LOCAL_PATH` dentro de `.env`.

## Estructura

```text
flows/            código productivo por etapa (00 ingesta, 01 ingeniería, 03 agregación, 05 modelado)
flows/common/     rutas, configuración, logging y hashing compartidos
data/             capas medallion (ignorado por Git; se genera al ejecutar)
  bronze/         copia intacta de la fuente + metadatos de ingesta
  silver/         datos validados y enriquecidos (Parquet)
  gold/           tablas agregadas para informes
notebooks/        exploración y modelado (sin lógica productiva; salidas limpiadas)
tests/            pruebas de cada módulo (pytest)
reports/          conclusiones_eda.md, problema_ml.md, model_card.md
models/           modelo .joblib + .metadata.json (ignorado por Git)
```

## Buenas prácticas aplicadas

- Funciones puras con contratos de validación (falla ruidosa) y pruebas automáticas.
- Configuración externa (`.env`), rutas centralizadas (`flows/common/paths.py`), cero rutas absolutas.
- Reproducibilidad: semilla fija, huella SHA-256 de datos y modelo, versiones de librerías en metadatos.
- Evaluación sin fuga de datos: preprocesamiento dentro de un `Pipeline`, conjunto de prueba abierto una sola vez,
  métrica de prueba con intervalo de confianza por bootstrap.
- Distingue asociación de causalidad y documenta los sesgos de imputación en `reports/`.

## Verificación de extremo a extremo (clon limpio)

```bash
git clone <tu-repositorio> && cd proyecto_ciencia_de_datos_linea_enfasis
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
make test && make pipeline
ls data/bronze data/silver data/gold models
```
