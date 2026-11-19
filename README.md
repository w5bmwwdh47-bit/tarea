# Producción frutícola del Valle del Cauca 2000–2022

**El Valle nunca produjo tanta fruta, ni dependió de tan pocos cultivos**

Analítica de Datos · Universidad Cooperativa de Colombia · Carlos Ignacio Grueso P. · Jean Carlos Cárdenas

## Procedencia del dataset
- **Fuente:** [datos.gov.co](https://www.datos.gov.co), Ministerio de Agricultura y Desarrollo Rural, *Evaluaciones Agropecuarias Municipales*: producción de frutales del Valle del Cauca.
- **Descarga:** 28 de septiembre de 2026. Archivo `frutales.csv`, sin modificar.
- **Contenido:** 9.992 filas y 8 columnas (año, municipio, subregión, cultivo, ciclo y producción en toneladas), 2000–2022. Tras la limpieza quedan 9.953 filas.

## Estructura
| Archivo / carpeta | Qué es |
|---|---|
| `frutales.csv` | Dataset original |
| `01_EDA_momento1.ipynb` | Análisis exploratorio (Momento 1) |
| `preparacion.py` | Limpieza de datos, usada por todos los demás archivos |
| `02_modelado_y_evaluacion.ipynb` | Modelado y evaluación con machine learning (Momento 3) |
| `modelo.py` | Funciones del modelo, el pronóstico y el agrupamiento, usadas por el notebook y el dashboard |
| `app.py` | Dashboard interactivo en Streamlit |
| `presentacion/presentacion_frutales.html` | Presentación final: abrir con doble clic |
| `presentacion/dashboard_frutales.html` | Versión web del dashboard, sin instalar nada |
| `capturas/` | Respaldo del dashboard |
| `docs/flujo_de_decisiones.md` | Cómo se construyó el modelo y por qué se tomó cada decisión |
| `docs/guion_presentacion.txt` | Guion de la exposición |

## Cómo correrlo
```bash
pip install -r requirements.txt
```
Orden de ejecución:
1. `python preparacion.py`: limpia los datos y genera `frutales_limpio.csv` (opcional; los demás archivos llaman la limpieza solos).
2. `01_EDA_momento1.ipynb`: análisis exploratorio.
3. `02_modelado_y_evaluacion.ipynb`: entrenamiento, evaluación, pronóstico 2025 y agrupamiento. Ejecutar todas las celdas en orden.
4. `streamlit run app.py`: abre el dashboard en http://localhost:8501.

Todo usa `random_state=42`, así que los resultados se pueden repetir.

## Resultados del modelo (conjunto de prueba, 1.496 casos)
| Método | Error promedio (t) |
|---|---|
| Árboles en cadena (HistGradientBoosting) — **elegido** | **770** |
| Árbol de decisión | 905 |
| Regla simple (repetir la última cosecha) | 979 |
| Modelo tonto (promedio) | 2.688 |

R² en entrenamiento 0,739 y en prueba 0,743. Gap −0,003, es decir, el modelo no memorizó.

## Recomendaciones de negocio
1. **Destinar el próximo fondo de fomento a aguacate, mango, papaya y lulo.**
   *Evidencia:* crecieron +224%, +215%, +262% y +194% entre 2000–04 y 2018–22. Sin intervención, el modelo anticipa que caerían de 66.845 t (2022) a 59.868 t (2025), un 10% menos.
2. **Empezar por Dagua, Jamundí, La Cumbre y Vijes, en la subregión Sur.**
   *Evidencia:* el Sur es la subregión que más creció (+121%) y aporta solo el 18,1% de la producción. El agrupamiento ubica a estos municipios entre los que crecen rápido con poca participación de los 4 cultivos.
3. **Fijar como meta superar 59.900 t de los 4 cultivos en 2025 y revisarla cada año en el dashboard.**
   *Evidencia:* es el pronóstico sin plan, con 21% menos error que la mejor regla simple. Lo que se produzca por encima de esa cifra es el efecto del fomento.

## Limitaciones
- Los datos son toneladas, sin área sembrada ni precios.
- La correlación y el pronóstico no indican causas.
- En 2020–2022 la ventaja del modelo sobre la regla simple fue pequeña: 684 t frente a 708 t.
- El número de registros creció con los años, así que parte del crecimiento puede deberse a mejor cobertura.

## Uso de IA
Usamos un asistente de IA (Claude) como apoyo para escribir el código y estructurar la presentación. El equipo formuló las preguntas, verificó cada cifra, interpretó los resultados y tomó las decisiones.
