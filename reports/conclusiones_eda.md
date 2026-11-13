# Conclusiones del EDA — Titanic (1912)

Informe para la gerencia de la aseguradora. Fuente: `data/silver/titanic_silver.parquet`
(891 pasajeros etiquetados, generado por `make silver`). Todas las cifras son observadas,
no estimadas, salvo donde se indica. Tablas de apoyo en `data/gold/` (regenerar con `make gold`).

## 1. Hallazgos (cada uno con su cifra y su n)

1. **El género fue el factor dominante.** Sobrevivieron 74.2 % de las mujeres (233/314)
   frente a 18.9 % de los hombres (109/577). La asociación es estadísticamente sólida
   (chi² = 260.7, p ≈ 1.2e-58), pero eso solo dice que es muy improbable que sea azar;
   no dice que el género "cause" la supervivencia.
2. **La clase social marcó una brecha enorme.** 1ª: 63.0 % (216), 2ª: 47.3 % (184),
   3ª: 24.2 % (491). La interacción con el género es extrema: mujeres de 1ª 96.8 % (n = 94)
   frente a hombres de 3ª 13.5 % (n = 347). La clase amplificó —o frenó— el acceso a los botes.
3. **Los niños fueron prioridad de evacuación, pero no alcanzó para todos.**
   `child` (edad < 16): 59.0 % de supervivencia (49/83), muy por encima del 16.4 % de los
   hombres adultos (88/537) y por debajo del 75.6 % de las mujeres adultas (205/271).
4. **Viajar acompañado ayudó (hasta cierto tamaño).** Con familia: 50.6 % (179/354) vs.
   solos: 30.4 % (163/537). Pero las familias de 4+ miembros cayeron a 16.1 % (10/62):
   grupos grandes se separaron o perdieron tiempo buscándose.
5. **Tener cabina registrada se asocia a sobrevivir (66.7 % vs. 30.0 %)**, pero es
   sospechoso de **artefacto del registro**: 77.1 % de las cabinas faltan, y es plausible
   que se registraran más las de quienes sobrevivieron para contarlas. No lo interpretamos
   como efecto protector (ver `reports/problema_ml.md`, Paso 61).

## 2. Decisiones de limpieza y su sesgo

| Decisión | Qué hace | Sesgo que introduce |
|---|---|---|
| `Age` imputada con el promedio por `Sex` | Rellena 177 nulos (19.9 %) | La desviación estándar cae de 14.53 a 13.01: la distribución se estrecha artificialmente y las conclusiones sobre edad quedan algo "más seguras" de lo real. Se conserva `AgeOriginal` para auditarlo. |
| `Person` sin edad usa el `Sex` | 177 pasajeros sin edad nunca pueden ser `child` | Subestima a los niños. Si los 177 sin edad se hubieran comportado como los que sí tienen edad (11.6 % child), faltarían ≈ 21 niños más en `Person`. |
| `Master` fusionado con `Mr` (consolidación de títulos) | Títulos raros se agrupan | Mezcla niños varones con adultos en `TTitulo`; la columna `Person` compensa parcialmente. Nos parece defendible por el tamaño de grupo (estadística necesita n), pero documentado como compromiso. |
| `TCabin` = 'X' para cabina desconocida | "No sé" se vuelve categoría explícita | Los 687 'X' mezclan pasajeros muy distintos; `NCabin` les asigna el valor más alto del mapa, un orden que no existe realmente. Para modelar se usa one-hot, no este ordinal. |
| Cabina anómala 'T' (n = 1) | Se excluye SOLO del gráfico | Un grupo de 1 no sirve para estadística, pero la fila se conserva: no es un error de datos. |

## 3. Confusiones detectadas y cómo las separamos

- **Puerto de embarque parece importar (C: 55.4 %, n = 168; S: 33.9 %, n = 646), pero la
  composición de clases explica casi todo**: desde Cherbourg embarcó 50.6 % de 1ª clase;
  desde Queenstown, 93.5 % de 3ª. Estratificado por clase (tabla
  `survival_by_embarked_class`), el "efecto puerto" se encoge (en 1ª clase: C 69.4 % vs.
  S 58.9 % — mucho menor que la brecha bruta). La clase es la variable confusante principal.
- **Tarifa**: dentro de cada clase, el cuartil más caro supera al más barato
  (1ª: 73.6 % vs. 46.3 %; 3ª: 28.3 % vs. 22.7 %, tabla `survival_by_fare_quartile_class`),
  así que pagar más ayudó *dentro* de la misma clase, pero el grueso de la brecha lo
  explica la clase misma.
- **Atípicos de tarifa (116 por IQR)**: sobrevivieron 68.1 % frente a 33.9 % del resto.
  No son errores de captura: son pasajeros de 1ª clase con suites. No se eliminan.

## 4. Qué NO podemos afirmar

- **Asociación ≠ causalidad**: la matriz de correlación y las tasas describen lo que
  ocurrió, no lo que causó qué. Nadie "sobrevivió porque era mujer"; el protocolo de
  evacuación priorizó a mujeres y niños.
- **Un solo naufragio, una sola noche**: no hay réplica; cualquier conclusión es sobre
  este evento y no generaliza a otros desastres marítimos sin más evidencia.
- **Datos incompletos**: 19.9 % de edades y 77.1 % de cabinas faltan, y no sabemos si
  faltan al azar (la ausencia de cabina se asocia al resultado, ver hallazgo 5).
- **Las barras de error de familias grandes (6+ personas) son enormes** (n = 6 a 25):
  esas tasas casi no son informativas.
- Este análisis **no respalda decisiones de asignación de riesgo individual** por
  género o clase: describe un evento histórico con sesgos de registro documentados.
