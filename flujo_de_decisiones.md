# Flujo de decisiones: cómo se construyó el machine learning

```
Datos crudos (frutales.csv)
   │  preparacion.py: texto→número · 39 duplicados fuera · 0 vacíos · 202 ceros conservados
   ▼
Datos limpios (9.953 filas)
   │  Sumar ciclos → 1 fila por municipio–cultivo–año
   │  Agregar lo conocido 3 años antes: cosechas de hace 3, 4 y 5 años y tendencias
   ▼
Tabla de modelado (7.477 casos)
   │  Partición 80/20, random_state=42, ANTES de cualquier ajuste
   ├──► Entrenamiento (5.981)        └──► Prueba (1.496), guardada hasta el final
   ▼
Líneas base ─ Modelo tonto (promedio): 2.688 t · Regla simple (repetir última cosecha): 979 t
   ▼
Modelo A ─ Árbol de decisión · curva de profundidad 1–10 · elegido 9 · 905 t
Modelo B ─ Árboles en cadena · curva de profundidad 1–8 · elegido 5 · 770 t  ◄── GANADOR
   │  Regla del Gap: solo profundidades con Gap ≤ 0,10 · Gap final −0,003
   ▼
Validación extra en el tiempo: entrenar ≤2019 → pronosticar 2020–2022 (684 t frente a 708 t)
   ▼
Interpretabilidad: importancia por permutación
   │  Cosechas previas ≈ 80% · Cultivo 9% · Municipio 7%
   ▼
Uso para el negocio
   ├─ Pronóstico 2025 sin plan (4 cultivos): 66.845 t → 59.868 t (−10%)
   └─ Agrupamiento K-Means (k=3) de 42 municipios → dónde replicar el plan
```

## Decisiones y por qué
| # | Decisión | Por qué |
|---|---|---|
| 1 | Pronosticar a **3 años** y no a 1 | A 1 año ningún modelo supera "repetir el año anterior", porque la producción cambia poco de un año a otro. Además, un plan de fomento tarda años en dar resultados. |
| 2 | **Regresión** | La respuesta es una cantidad en toneladas. |
| 3 | Solo usar información de 3 años antes | Evita la fuga de datos: el modelo no puede ver el futuro. |
| 4 | Partición 80/20 antes de todo | El examen tiene que hacerse con datos que el modelo nunca vio. |
| 5 | Comparar con 2 líneas base | Un modelo solo vale si le gana al promedio y a la regla simple. |
| 6 | Probar 2 modelos | El árbol simple es fácil de leer. Los árboles en cadena pronostican mejor (21% menos error), así que fue el elegido. |
| 7 | Error en toneladas (MAE) como métrica principal | Es fácil de explicar a quien decide: cuántas toneladas se equivoca en promedio. |
| 8 | Profundidad con Gap ≤ 0,10 y menor error | Evita que el modelo memorice. |
| 9 | Importancia por permutación | Mide cuánto empeora el pronóstico si se "desordena" cada variable. |
| 10 | Agrupamiento K-Means con datos escalados | No hay etiquetas de "tipo de municipio", así que se dejó que el algoritmo encontrara los grupos. |

## Límites
- En los años de pandemia (2020–2022) el modelo apenas le ganó a la regla simple.
- Pronosticar no es explicar: el modelo anticipa cuánto se producirá, pero no por qué.
- No hay precios ni área sembrada.
