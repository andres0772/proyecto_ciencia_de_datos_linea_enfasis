# Taller Titanic — proyecto completo (docs/GUIA.md)

**Objetivo**: implementar el proyecto completo de la guía (17 partes): módulos comunes, ingesta,
ingeniería, agregación, modelado, cuadernos, reports y README, con `make test` y `make pipeline` verdes.
**Rama**: `feat/taller-titanic` · **Estrategia de entrega**: single-pr (una rama, slices por unidad; merge/PR decisión del usuario).
**Runner de tests**: `pytest -q` (venv uv, Python 3.12) · **Fuente de especificación**: `docs/GUIA.md` (los tests exactos están ahí).

## Specs

S1 (GUIA §0.4): módulos comunes completos — `paths.py`, `config.py`, `logging.py`, `hashing.py` con el código de la guía verbatim; `import flows.common.*` funciona.
S2 (GUIA §0.5–0.6): `flows/00_ingestion.py` (read_source, split_labeled, run, quality_report), `flows/job.py` orquestador; tests `tests/test_00_ingestion.py` y `tests/test_job.py` (verbatim de la guía) en verde; `make ingest` genera Bronze + `_ingestion_metadata.json` + `_quality_report.json`.
S3 (GUIA Partes 2–8): `flows/01_data_engineering.py` completa (validate_bronze, cast_types, add_person, add_cabin_features, add_family_features, add_survivor_label, add_title, consolidate_titles, impute_embarked, impute_age, encode_categoricals, drop_original_columns, validate_silver, build_silver); tests `tests/test_01_data_engineering.py` (verbatim) en verde; `make silver` genera los 2 Parquet.
S4 (GUIA Partes 6, 9): `flows/03_aggregation.py` completa (group_stats, chi2_sex_survival, survival_by_class_sex, survival_by_embarked, survival_by_embarked_class, add_fare_quartile, survival_by_fare_quartile_class, survival_by_family, outlier_mask_iqr, outlier_summary, run); tests `tests/test_03_aggregation.py` (verbatim) en verde; `make gold` genera las tablas + chi2 json.
S5 (GUIA Partes 10–17): `flows/05_ml_models.py` completa (load_dataset, split, GroupAgeImputer, build_pipeline, get_candidates, cross_validate_candidates, get_param_grids, evaluate_on_test, train_final, save_model, load_model, predict, run con bootstrap CI en metadatos); tests `tests/test_05_ml_models.py` (verbatim) en verde; `make train` guarda modelo + metadatos.
S6 (GUIA Partes 1–7, cuaderno): `notebooks/02_exploracion.ipynb` — EDA completo (diagnóstico Bronze, gráficos, imputaciones, correlaciones, chi2, boxplots), corre de arriba abajo sin errores, se commitea sin salidas (nbstripout).
S7 (GUIA Partes 10–17, cuaderno): `notebooks/04_modelado.ipynb` — formulación, baselines, pipeline, CV, evaluación en profundidad, umbral, GridSearch, interpretación, ablación, evaluación final, modelo guardado; corre de arriba abajo; sin salidas al commit.
S8 (GUIA entregables): `reports/conclusiones_eda.md`, `reports/problema_ml.md`, `reports/model_card.md` con cifras OBSERVADAS del pipeline real (n, tasas, chi2, métricas de prueba con intervalo); distinguen asociación vs causalidad.
S9 (GUIA §Definición de terminado, verbatim): "make test pasa completo y make lint no reporta errores" / "make pipeline genera data/bronze, data/silver, data/gold y models/titanic_model_v1.joblib + .metadata.json desde un clon limpio" / "No hay rutas absolutas, credenciales ni datos/modelos en Git (git status limpio de data/, models/)" / README.md describe instalación (uv) y ejecución.

## Tasks

- T1 (S1, inline): llenar los 4 módulos comunes; verificación de imports. ✅ `304c025`
- T2 (S2, inline): tests RED → 00_ingestion.py + job.py → GREEN → `make ingest`. ✅ `c065790`
- T3 (S3, inline): tests RED → 01_data_engineering.py → GREEN → `make silver`. ✅ `f2ddf00`
- T4 (S4, inline): tests RED → 03_aggregation.py → GREEN → `make gold`. ✅ `9f83278`
- T5 (S5, inline): tests RED → 05_ml_models.py → GREEN → `make train`. ✅ `b1417b2` + fix `ad182c8` (unidad de alto riesgo: riesgo ítem 3 por contratos/funciones compartidas del módulo ML)
- T6 (S6, inline): cuaderno 02 ejecutado + strip. ✅ `22c3835`
- T7 (S7, inline): cuaderno 04 ejecutado + strip. ✅ `22c3835` (parches de ejecución en `ad182c8`: indexado posicional y `.tolist()`)
- T8 (S8, inline): reports con cifras observadas. ✅ `a122f8e`
- T9 (S9, inline): README + `make clean` + verificación extremo a extremo + lint. ✅ `a122f8e` + `37d9fec`

Ruta declarada: todo inline. Justificación: la guía trae tests y esqueletos verbatim ya cargados en el contexto del padre; delegar obligaría a cada worker a releer 2.000 líneas de guía por el mismo resultado.

## Log

- L1 (2026-11-13): pedido del usuario verbatim: "esto es muy demorado podes hacerlo por mi para yo al terminar ejecutarlo y leerlo porfa?" — alcance confirmado: "Todo el proyecto". El usuario construye = ya no aplica; el agente implementa, el usuario ejecuta y lee.
- L1b: contexto previo del usuario: quiere uv (no venv/pip directo), usa fish shell, prefiere explicaciones simples. El entorno ya quedó montado: .venv 3.12.16, .gitignore, .env, Makefile (arreglado: TABs y erratas).
- L2 (T5): el test corrigió el diseño: `predict()` debe devolver EXACTAMENTE 2 columnas (`proba_survive`, `pred`); se quitó `PassengerId`.
- L3 (T7): cuaderno 04 ejecutado con 3 iteraciones de parche: (1) indexado de etiquetas sobre array posicional en errores de alta confianza → `pd.Series(conf, index=errores.index)`; (2) `dif.to_list()` es de Series, ndarray usa `.tolist()`; lección: ejecutar el cuaderno mentalmente antes de nbconvert no sustituye la ejecución.
- L4 (decisión congelada real): el GridSearch del Paso 57 eligió lr=0.1, max_depth=4, max_iter=200, l2=1.0; comparación pareada Δ=+0.0057±0.0103 (mejora dentro del ruido entre pliegues). Se congelaron esos valores (no los del ejemplo de la guía). Ablación: cabina aporta +0.0004 AUC → excluida (posible artefacto de registro). THRESHOLD=0.5 (umbral F1 óptimo 0.609, marginal).
- L5 (verificación final): `make lint` limpio (ruff excluye notebooks, ignora N999), 29 tests verdes, `make pipeline` desde `make clean` genera bronze/silver/gold + modelo. Métrica de prueba: exactitud 0.810, F1 0.742, ROC-AUC 0.824 [IC 95%: 0.747–0.894], n_test 179. git status limpio.
- L6 (cierre): 9 commits en `feat/taller-titanic` (304c025…37d9fec). Pendiente del usuario: merge/PR a main (remote `upstream` del profesor dfmarin). Los reports son textos reflexivos personalizables por el usuario.
