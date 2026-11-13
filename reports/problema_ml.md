# Problema de ML — diseño, decisiones y limitaciones

Complementa `docs/GUIA.md` (Partes 10–17) y `reports/conclusiones_eda.md`.
Todas las cifras son observadas en ejecuciones reales (semilla 42, Python 3.12.16).

## 1. Formulación (Paso 39)

| Decisión | Respuesta | Justificación |
|---|---|---|
| Tipo de problema | Clasificación binaria supervisada | `Survived` es 0/1 y hay 891 ejemplos etiquetados. |
| Variable objetivo | `Survived` (1 = sobrevivió) | Es la pregunta del negocio (¿quién sobrevive en un desastre?). |
| Unidad de análisis y momento | Un pasajero, *al embarcar* | Solo se usan datos que existían antes del naufragio (con la excepción auditada de la cabina, ver §4). |
| Métrica principal | **ROC-AUC** | No depende del umbral, funciona bien con 38.4 % de positivos y permite comparar modelos por ranking. |
| Métricas secundarias | exactitud, precisión, recall, F1 | Con 618 "no sobrevivió" y 341 "sí" (train), la exactitud sola engaña (un dummy acierta 61.7 %). |
| Costo de errores | Para la aseguradora, un falso "sobrevive" sobreestima la prima de riesgo; un falso "no sobrevive" subestima la exposición y la responsabilidad actuarial. Con costos simétricos asumidos, se mantiene umbral 0.5. | Ver §5. |
| Criterio de éxito | Superar la regla de género (exactitud 0.789, F1 0.715) y al dummy (ROC-AUC 0.5) por un margen defendible. | Logrado: ver §3 y §6. |

## 2. Auditoría de variables y fuga de datos (Paso 40)

- Excluidas del modelo: `Survived`/`Survivor` (objetivo), `PassengerId`/`Ticket`/`Name`
  (identificadores que memorizan), `NSex`/`NEmbarked`/`NCabin`/`NTitulo` (duplican categorías
  con un orden arbitrario; se usa one-hot), `Age` imputada (estadística calculada con todos
  los datos → fuga), `Person` (redundante con `Age`+`Sex`).
- Se usa `AgeOriginal` (con NaN): el `GroupAgeImputer` del pipeline aprende los promedios
  **solo con datos de entrenamiento** dentro de cada fold (diferencia observada entre la media
  de mujeres en todo el dataset y solo en train: pequeña aquí, pero el método es el error).
- `HasCabin`/`TCabin` se marcan como "sospechosas" (posible artefacto del registro) y se
  evalúan con ablación (ver §4).

## 3. Baselines y selección de modelos (Pasos 42, 47–50)

Baselines sobre el entrenamiento (5 pliegues estratificados, semilla 42):
- **Dummy** (`most_frequent`): exactitud 0.617, F1 0.000, ROC-AUC 0.500.
- **Regla de género** (predice sobrevivir si `Sex == female`): exactitud 0.789,
  precisión 0.743, recall 0.689, F1 0.715.

Validación cruzada de candidatos (media ± desv., ROC-AUC en validación):

| Modelo | Exactitud | ROC-AUC | Brecha train−val |
|---|---|---|---|
| **hgb** (HistGradientBoosting) | 0.820 ± 0.028 | **0.887 ± 0.019** | 0.107 |
| forest | 0.803 ± 0.012 | 0.866 ± 0.016 | 0.133 |
| logreg | 0.792 ± 0.022 | 0.858 ± 0.024 | 0.006 |
| tree (max_depth 4) | 0.805 ± 0.026 | 0.847 ± 0.026 | 0.046 |
| knn | 0.787 ± 0.017 | 0.841 ± 0.017 | 0.087 |
| dummy | 0.617 | 0.500 | 0.000 |

**Finalistas y justificación**: `hgb` (mejor desempeño medio, brecha moderada),
`forest` (segundo, muy estable pero con la mayor brecha: memoriza) y `logreg`
(ligeramente peor pero casi sin brecha y totalmente interpretable: buena opción si el
negocio necesita explicaciones). El más sobreajustado es `forest` (brecha 0.133), el más
estable `logreg` (0.006). Se eligió **hgb** como modelo final por desempeño y porque su
brecha se controla con regularización (ver §4).

## 4. Ajuste de hiperparámetros y ablation (Pasos 56–58, 61)

GridSearch (5 pliegues, solo entrenamiento): `hgb` ganó con
`learning_rate=0.1, max_depth=4, max_iter=200, l2_regularization=1.0` (0.8925);
`forest` 0.8822; `logreg` 0.8585 (el ajuste no aportó: C=1 es el valor por defecto).

**Comparación pareada (mismos pliegues, hgb por defecto vs. ajustado):**
diferencia pliegue a pliegue [-0.007, -0.004, +0.009, +0.011, +0.021]; media +0.0057 ± 0.0103.
**Conclusión honesta: la mejora del ajuste está dentro de la variabilidad entre pliegues.**
Se congelan igualmente los hiperparámetros ajustados porque limitan profundidad (max_depth=4)
y agregan regularización L2, lo que reduce la brecha train−val; pero se reconoce que el
modelo por defecto era estadísticamente equivalente.

**Ablación (ROC-AUC en CV, mismos pliegues):**

| Configuración | ROC-AUC | Δ vs. base |
|---|---|---|
| A base | 0.8867 ± 0.019 | — |
| B base + cabina (`HasCabin`, `TCabin`) | 0.8871 ± 0.021 | +0.0004 |
| C sin `TTitulo` | 0.8772 ± 0.017 | −0.0095 |
| D sin `Sex` | 0.8835 ± 0.020 | −0.0032 |

**Decisión**: se **excluyen** las variables de cabina del modelo final (`USE_CABIN_FEATURES=False`).
Aportan ~nada (+0.0004) y cargan el riesgo del artefacto de registro (Paso 40): un modelo que
explota "tener cabina anotada" no estima riesgo real. `TTitulo` aporta algo (+0.0095 al
quitarlo baja) y se conserva; `Sex` sigue siendo la variable más fuerte.

## 5. Calibración y umbral (Paso 53)

Brier score 0.1302 (cerca de 0 = bien calibrado; el azar daría ≈0.25 con esta base).
El umbral que maximiza F1 fuera de muestra es **0.609** (F1 0.759) frente a 0.5 (F1 ≈ 0.755):
la diferencia es marginal. **Decisión congelada: THRESHOLD = 0.5**, porque no definimos
costos asimétricos de error y un umbral por defecto es auditable. Si la aseguradora
definiera que subestimar exposición es más caro que sobrestimarla, se justificaría subirlo.

## 6. Evaluación final (una sola vez, Paso 62) y subgrupos (Paso 54)

| Modelo | Exactitud | ROC-AUC |
|---|---|---|
| dummy (CV) | 0.617 | 0.500 |
| regla de género | 0.789 | — |
| hgb ajustado (CV) | 0.820 ± 0.028 | 0.887 ± 0.019 |
| **hgb ajustado (prueba, n = 179)** | **0.810** | **0.824 [IC 95 %: 0.747 – 0.894]** |

La estimación de prueba es coherente con la de validación cruzada (0.824 dentro del
intervalo). Con ~179 pasajeros de prueba el IC es ancho (±0.07): **no comparar modelos
por décimas**. La exactitud de prueba (0.810) supera el criterio de éxito (0.789).

Errores fuera de muestra por subgrupo (tasa de error, n):
`Sex`: female 0.194 (253) / male 0.172 (459) · `Pclass`: 1ª 0.211 (171) / 2ª 0.087 (150) /
3ª 0.202 (391) · edad: niño 0.132 (76) / adulto 0.204 (378) / mayor 0.182 (121).
El modelo falla más en 1ª clase (donde la regla de género es menos suficiente) y menos en
niños y 2ª clase. Detalle de 10 errores de alta confianza en el cuaderno 04.

## 7. Limitaciones y ética (Paso 63)

- **Tamaño**: 891 pasajeros, ~179 de prueba; intervalos anchos; diferencias pequeñas
  entre modelos no son reales.
- **Un solo suceso histórico** (1912): no generaliza a otros desastres sin más evidencia.
- **Datos incompletos y no aleatorios**: 19.9 % de edades y 77.1 % de cabinas faltan;
  la ausencia se asocia al resultado (posible fuga de registro, auditada).
- **Variables sensibles**: `Sex` y `Pclass` dominaron las decisiones de evacuación de
  1912. Predecir bien *lo que ocurrió* no implica que *deba* ocurrir así. Este modelo
  **NO debería usarse** para asignar prioridades de rescate, primas individuales ni
  ningún tratamiento diferenciado por género o clase; sirve como estudio histórico y
  didáctico de patrones de evacuación.
- La asociación documentada (mujeres/niños/1ª clase sobrevivieron más) refleja protocolos
  sociales de la época, no causas físicas.
