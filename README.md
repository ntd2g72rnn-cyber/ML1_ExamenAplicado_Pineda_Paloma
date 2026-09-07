# ML1 Examen Aplicado — Pineda, Paloma

Repositorio del examen aplicado de Machine Learning I. **Estado actual: en progreso** (entorno configurado, dataset seleccionado, analisis exploratorio, preprocesamiento sin leakage, PCA, K-Means y modelado supervisado con validacion cruzada completos; diagnostico del mejor modelo, justificacion, conclusiones ejecutivas y video se agregan en etapas siguientes).

## Dataset seleccionado

- **Nombre:** California Housing (scikit-learn / StatLib)
- **URL:** https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- **Tarea:** regresion supervisada
- **Observaciones:** 20.640 filas, 8 variables predictoras numericas + 1 objetivo
- **Objetivo:** `MedHouseValue`, valor mediano de vivienda en cientos de miles de dolares

Ver el detalle completo en [ml1_examen_aplicado.ipynb](ml1_examen_aplicado.ipynb): carga inicial, analisis exploratorio, division train/test y preprocesamiento sin leakage (`ColumnTransformer`), PCA, K-Means y modelado supervisado (Ridge y Random Forest) con `GridSearchCV`. Los graficos generados quedan en [figures](figures/) y la tabla comparativa de modelos en [results/model_comparison.csv](results/model_comparison.csv).

## Metodologia

Los datos se separan en train/test (80/20, `random_state=42`) antes de ajustar cualquier imputador, codificador o escalador. El `ColumnTransformer` aplica imputacion mediana y `StandardScaler` a las variables numericas, dejando declarados (aunque vacios para este dataset) los ramales nominal y ordinal. Se aplica PCA eligiendo el numero de componentes cuya varianza acumulada cae entre 80% y 90%, y K-Means evaluando K entre 2 y 10 con inercia y silhouette. Para el modelado supervisado se comparan Ridge y Random Forest con `GridSearchCV`, cinco folds y `scoring='neg_root_mean_squared_error'`; el conjunto de test permanece reservado hasta la evaluacion final.

## Resultado principal (modelado supervisado)

| Modelo | RMSE | MAE | R2 | MAPE (%) | Tiempo entrenamiento (s) |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.5045 | 0.3271 | 0.8058 | 18.8549 | 102.8226 |
| Ridge | 0.7456 | 0.5332 | 0.5758 | 31.9522 | 3.2978 |

El mejor modelo por RMSE es Random Forest. El diagnostico detallado (residuales, importancia de variables, observaciones con mayor error) y la justificacion del modelo seleccionado se agregan en la etapa siguiente.

## Entorno de trabajo

Este proyecto usa [uv](https://docs.astral.sh/uv/) para gestionar el entorno Python.

```bash
uv sync
uv run jupyter notebook
```

Alternativamente, con pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

### Hugging Face (para etapas futuras)

El dataset actual no requiere autenticacion, pero el proyecto deja preparada una cuenta y token de Hugging Face para posibles pasos futuros (ver seccion 1 de [ml1_examen_aplicado.ipynb](ml1_examen_aplicado.ipynb)). El token se guarda localmente en un archivo `.env` con la variable `HF_TOKEN`, el cual **no se sube a git** (ver `.gitignore`).

## Bitacora de trabajo

### 2026-09-06
- Se crea el repositorio `ML1_ExamenAplicado_Pineda_Paloma` en GitHub.
- Se copian los archivos de configuracion del entorno (`pyproject.toml`, `uv.lock`, `.python-version`, `requirements.txt`, `src/exam/`) desde el proyecto de trabajo, sin incluir secretos (`.env`, `accessToken.txt`).
- Se valida que `uv sync` reconstruye el entorno correctamente en este repositorio.
- Se selecciona el dataset **California Housing** como dataset del examen, cumpliendo los minimos exigidos por la ficha (>500 filas, >=6 predictoras, >=3 numericas continuas, no pertenece a la lista de datasets excluidos).
- Se documentan los pasos para crear una cuenta y un token de acceso en Hugging Face, dejando el mecanismo listo para etapas futuras del proyecto.
- Se crea `seleccion_dataset.ipynb` con la declaracion del dataset y su carga inicial con pandas.
- Se amplia `seleccion_dataset.ipynb` con el analisis exploratorio hasta el punto "3. Analisis exploratorio": valores faltantes (0% en todas las columnas), tratamiento diagnostico de outliers con IQR, distribucion y skewness del objetivo, correlaciones de Pearson, scatterplots y pairplot. Notebook ejecutado sin errores; graficos exportados a `figures/`.

### 2026-09-07
- Se renombra `seleccion_dataset.ipynb` a `ml1_examen_aplicado.ipynb` para reflejar que el notebook ya cubre mas alla de la seleccion del dataset.
- Se agrega la seccion "5. Division train/test y preprocesamiento sin leakage": separacion 80/20 con `random_state=42` antes de ajustar cualquier transformador, y `ColumnTransformer` con imputacion mediana + `StandardScaler` para numericas (ramales nominal y ordinal declarados y documentados aunque vacios para este dataset).
- Se agrega la seccion "6. PCA y contribuciones de variables": scree plot y varianza acumulada, seleccion automatica del numero de componentes entre 80% y 90%, proyeccion PC1 vs PC2 y loadings mas relevantes por componente.
- Se agrega la seccion "7. K-Means y perfil de clusters": barrido de K entre 2 y 10 con inercia y silhouette, seleccion del K optimo, visualizacion de clusters en el espacio PCA e interpretacion de perfiles por grupo.
- Se agrega la seccion "8. Modelos supervisados con validacion cruzada": `GridSearchCV` (5 folds) para Ridge y Random Forest, metricas RMSE/MAE/R2/MAPE sobre test y exportacion de la tabla comparativa a `results/model_comparison.csv`.
- Notebook ejecutado de punta a punta sin errores; graficos nuevos (`pca_scree_plot.png`, `pca_pc1_pc2.png`, `kmeans_elbow_silhouette.png`, `kmeans_clusters_pca.png`) exportados a `figures/`.
- Mejor modelo por RMSE en test: Random Forest (RMSE=0.5045, R2=0.8058) frente a Ridge (RMSE=0.7456, R2=0.5758).

### Proximos pasos pendientes
- Diagnostico del mejor modelo: residuales, importancia de variables y observaciones con mayor error absoluto.
- Justificacion del modelo seleccionado.
- Conclusiones ejecutivas.
- Video de presentacion.

## Declaracion de uso de IA

Se utilizo IA generativa como apoyo para estructurar el entorno, redactar la documentacion y revisar la cobertura de los requisitos de la ficha del examen. La seleccion del dataset y la responsabilidad sobre el contenido corresponden a la estudiante.
