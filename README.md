# ML1 Examen Aplicado — Pineda, Paloma

Repositorio del examen aplicado de Machine Learning I. **Estado actual: en progreso** (entorno configurado, dataset seleccionado y analisis exploratorio inicial completo; preprocesamiento sin leakage, PCA, K-Means, modelado e interpretacion se agregan en etapas siguientes).

## Dataset seleccionado

- **Nombre:** California Housing (scikit-learn / StatLib)
- **URL:** https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- **Tarea:** regresion supervisada
- **Observaciones:** 20.640 filas, 8 variables predictoras numericas + 1 objetivo
- **Objetivo:** `MedHouseValue`, valor mediano de vivienda en cientos de miles de dolares

Ver el detalle, la carga inicial y el analisis exploratorio (hasta el punto "3. Analisis exploratorio") en [seleccion_dataset.ipynb](seleccion_dataset.ipynb). Los graficos generados quedan en [figures](figures/).

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

El dataset actual no requiere autenticacion, pero el proyecto deja preparada una cuenta y token de Hugging Face para posibles pasos futuros (ver seccion 1 de [seleccion_dataset.ipynb](seleccion_dataset.ipynb)). El token se guarda localmente en un archivo `.env` con la variable `HF_TOKEN`, el cual **no se sube a git** (ver `.gitignore`).

## Bitacora de trabajo

### 2026-09-06
- Se crea el repositorio `ML1_ExamenAplicado_Pineda_Paloma` en GitHub.
- Se copian los archivos de configuracion del entorno (`pyproject.toml`, `uv.lock`, `.python-version`, `requirements.txt`, `src/exam/`) desde el proyecto de trabajo, sin incluir secretos (`.env`, `accessToken.txt`).
- Se valida que `uv sync` reconstruye el entorno correctamente en este repositorio.
- Se selecciona el dataset **California Housing** como dataset del examen, cumpliendo los minimos exigidos por la ficha (>500 filas, >=6 predictoras, >=3 numericas continuas, no pertenece a la lista de datasets excluidos).
- Se documentan los pasos para crear una cuenta y un token de acceso en Hugging Face, dejando el mecanismo listo para etapas futuras del proyecto.
- Se crea `seleccion_dataset.ipynb` con la declaracion del dataset y su carga inicial con pandas.
- Se amplia `seleccion_dataset.ipynb` con el analisis exploratorio hasta el punto "3. Analisis exploratorio": valores faltantes (0% en todas las columnas), tratamiento diagnostico de outliers con IQR, distribucion y skewness del objetivo, correlaciones de Pearson, scatterplots y pairplot. Notebook ejecutado sin errores; graficos exportados a `figures/`.

### Proximos pasos pendientes
- Pipeline de preprocesamiento sin leakage (`ColumnTransformer`).
- PCA y K-Means.
- Modelado supervisado (Ridge y Random Forest) con validacion cruzada.
- Interpretacion, conclusiones ejecutivas y video de presentacion.

## Declaracion de uso de IA

Se utilizo IA generativa como apoyo para estructurar el entorno, redactar la documentacion y revisar la cobertura de los requisitos de la ficha del examen. La seleccion del dataset y la responsabilidad sobre el contenido corresponden a la estudiante.
