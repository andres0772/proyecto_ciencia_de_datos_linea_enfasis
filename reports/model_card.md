# Ficha del modelo — titanic_model_v1

## Propósito y uso previsto

Modelo de clasificación binaria que estima la **probabilidad de sobrevivir** de un pasajero
del Titanic a partir de características conocidas al embarcar. **Uso previsto: educativo y
de análisis histórico** (patrones de evacuación de 1912). **Uso NO recomendado**: asignar
prioridades de rescate, fijar primas individuales o cualquier decisión diferenciada por
género o clase (ver Limitaciones).

## Datos de entrenamiento

- **Fuente**: `TITANIC_URL` (paste.c-net.org/MarvelJinxed), ingesta reproducible con `make ingest`.
- **Conjunto**: 891 pasajeros etiquetados (1912). La capa Silver usada tiene huella
  SHA-256 `1169b3fa24d4b86ae51e4b7402d10c24d464c234da3560e691dd56bf02073a0e`
  (auditada en `models/titanic_model_v1.metadata.json`).
- **Particiones**: 712 entrenamiento / 179 prueba (estratificada, semilla 42); el modelo
  final se reentrenó con los 891 tras validar el procedimiento.

## Variables

- **Numéricas**: `Age` (original, con NaN; el pipeline imputa por grupo Sex×Pclass aprendido
  solo en entrenamiento), `SibSp`, `Parch`, `Fare`, `Familia`.
- **Categóricas** (one-hot, categorías raras agrupadas): `Pclass`, `Sex`, `Embarked`,
  `TTitulo`, `Alone`.
- **Excluidas y por qué**: identificadores (`PassengerId`, `Name`, `Ticket`), codificaciones
  ordinales (`NSex`, `NEmbarked`, `NCabin`, `NTitulo`), `Age` imputada en Silver (fuga),
  `Person` (redundante), variables de cabina (`HasCabin`, `TCabin`: +0.0004 AUC y posible
  artefacto del registro — ablación en `reports/problema_ml.md` §4).
- **Objetivo**: `Survived` (1 = sobrevivió).

## Métricas

Validación cruzada (5 pliegues estratificados, entrenamiento): ROC-AUC 0.887 ± 0.019,
exactitud 0.820 ± 0.028, F1 0.755 ± 0.033.

**Evaluación en prueba (abierta una sola vez, n = 179):**
exactitud **0.810**, F1 **0.742**, ROC-AUC **0.824** con IC 95 % por bootstrap
**[0.747 – 0.894]**. Umbral de decisión: 0.5.

Desempeño por subgrupos (error fuera de muestra): error mayor en 1ª clase (0.211) y en
adultos (0.204); menor en 2ª clase (0.087) y niños (0.132). Ver §6 de `problema_ml.md`.

## Limitaciones

- Un solo suceso histórico; n pequeño → intervalos anchos; no comparar modelos por décimas.
- 19.9 % de edades y 77.1 % de cabinas faltantes; la imputación reduce la varianza (sesgo
  documentado) y la ausencia de cabina se asocia al resultado.
- Usa `Sex` y `Pclass`: variables sensibles que dominaron las decisiones de 1912. El modelo
  describe lo que ocurrió; no legitima que deba repetirse ni debe usarse para decidir hoy.

## Versiones y reproducibilidad

- Python 3.12.16 · scikit-learn 1.9.1 · pandas 3.1.1 (registradas en los metadatos; el
  cargador advierte si la versión de scikit-learn cambia).
- Hiperparámetros congelados: `learning_rate=0.1, max_depth=4, max_iter=200,
  l2_regularization=1.0` (HistGradientBoostingClassifier).
- Huella SHA-256 del modelo: `8f338eb6…` (completa en los metadatos).
- Reproducir desde cero: `make pipeline` (ingesta → silver → gold → train).
- Seguridad: los `.joblib` ejecutan código al cargarse; no cargar modelos de fuentes no confiables.
