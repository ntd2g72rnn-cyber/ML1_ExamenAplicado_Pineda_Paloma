# ML1 Examen Aplicado — Pineda, Paloma

Repositorio del examen aplicado de Machine Learning I. **Estado actual: en progreso** (entorno configurado y dataset seleccionado; EDA, modelado e interpretacion se agregan en etapas siguientes).

## Dataset seleccionado

- **Nombre:** California Housing (scikit-learn / StatLib)
- **URL:** https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- **Tarea:** regresion supervisada
- **Observaciones:** 20.640 filas, 8 variables predictoras numericas + 1 objetivo
- **Objetivo:** `MedHouseValue`, valor mediano de vivienda en cientos de miles de dolares

Ver el detalle y la carga inicial en [seleccion_dataset.ipynb](seleccion_dataset.ipynb).

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

### Proximos pasos pendientes
- EDA completo (nulos, outliers, distribucion del objetivo, correlaciones).
- Pipeline de preprocesamiento sin leakage (`ColumnTransformer`).
- PCA y K-Means.
- Modelado supervisado (Ridge y Random Forest) con validacion cruzada.
- Interpretacion, conclusiones ejecutivas y video de presentacion.

## Declaracion de uso de IA

Se utilizo IA generativa como apoyo para estructurar el entorno, redactar la documentacion y revisar la cobertura de los requisitos de la ficha del examen. La seleccion del dataset y la responsabilidad sobre el contenido corresponden a la estudiante.
