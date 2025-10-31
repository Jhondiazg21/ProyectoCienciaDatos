# Predictor de Riesgo de Insuficiencia Cardíaca

Proyecto de ciencia de datos para la predicción de eventos fatales en pacientes con insuficiencia cardíaca mediante modelos de machine learning. Este proyecto incluye un análisis exploratorio completo, preprocesamiento de datos, modelado predictivo y una aplicación web interactiva con Streamlit.

## Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Características](#-características)
- [Dataset](#-dataset)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Modelos Implementados](#-modelos-implementados)
- [Resultados](#-resultados)

## Descripción del Proyecto

Este proyecto se enfoca en desarrollar modelos predictivos para identificar pacientes con mayor riesgo de eventos fatales por insuficiencia cardíaca. Utiliza técnicas de machine learning para analizar variables clínicas y generar predicciones precisas que pueden ayudar en la toma de decisiones médicas. 

**Destacado**: El proyecto trabaja con un enfoque dual, utilizando tanto datos originales limpios como datos sintéticos con ruido para replicar desafíos reales en la transformación y limpieza de datos.

### Objetivos

- Realizar un análisis exploratorio exhaustivo de los datos clínicos
- Probar el preprocesamiento tanto con datos originales limpios como con datos sintéticos con ruido
- Crear pipelines robustos de limpieza y transformación de datos
- Comparar el rendimiento de múltiples algoritmos de clasificación
- Desplegar un modelo predictivo en una aplicación web interactiva

## Características

- **Enfoque dual con datos**: Trabajo con dataset original limpio y datos sintéticos modificados con ruido
- **EDA completo** con visualizaciones interactivas (Matplotlib, Seaborn, Plotly)
- **Generación de datos sintéticos** usando Gaussian Mixture Models (GMM)
- **Preprocesamiento robusto** que maneja:
  - Valores faltantes
  - Inconsistencias de tipo de datos
  - Valores extremos y atípicos
  - Errores comunes en datos reales
- **Modelado con scikit-learn**: Regresión Logística y Random Forest
- **Aplicación web** con Streamlit para predicciones interactivas
- **Análisis de importancia de variables** y métricas de evaluación

## Dataset

### Fuente
- **Nombre**: Heart Failure Clinical Records Dataset
- **Origen**: [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) / [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
- **Licencia**: Apache 2.0
- **Tipo**: Clasificación binaria

### Variables

El dataset contiene información clínica de pacientes con insuficiencia cardíaca:

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `age` | Numérico | Edad del paciente en años |
| `anaemia` | Binario | Disminución de glóbulos rojos o hemoglobina (1 = sí, 0 = no) |
| `creatinine_phosphokinase` | Numérico | Nivel de la enzima CPK en sangre (mcg/L) |
| `diabetes` | Binario | Indica si el paciente tiene diabetes (1 = sí, 0 = no) |
| `ejection_fraction` | Numérico | Porcentaje de sangre expulsada por el corazón en cada contracción |
| `high_blood_pressure` | Binario | Indica si el paciente tiene hipertensión (1 = sí, 0 = no) |
| `platelets` | Numérico | Número de plaquetas en sangre (kiloplaquetas/mL) |
| `serum_creatinine` | Numérico | Nivel de creatinina en sangre (mg/dL) |
| `serum_sodium` | Numérico | Nivel de sodio en sangre (mEq/L) |
| `sex` | Binario | Sexo del paciente (1 = masculino, 0 = femenino) |
| `smoking` | Binario | Indica si el paciente fuma (1 = sí, 0 = no) |
| `time` | Numérico | Tiempo de seguimiento en días |
| `DEATH_EVENT` | Binario | Variable objetivo (0 = sobrevivió, 1 = falleció) |

### Enfoque de Trabajo con Datos

Este proyecto implementa un enfoque dual para el tratamiento de datos, permitiendo trabajar tanto con información original como con datos sintéticos modificados:

#### 1. Dataset Original
- **Archivo**: `heart_failure_clinical_records_dataset.csv`
- **Registros**: 299 pacientes
- **Calidad**: Datos limpios y consistentes de Kaggle/UCI
- **Uso**: Baseline y análisis exploratorio inicial

#### 2. Datos Sintéticos con Ruido
- **Registros**: Ampliación a 2,299 (299 originales + 2,000 sintéticos)
- **Técnica**: Gaussian Mixture Models (GMM) preservando correlaciones
- **Modificaciones introducidas para mayor realismo**:
  - Valores faltantes aleatorios (5-15% por columna)
  - Cambios de tipo de dato (numéricos a texto con unidades)
  - Inconsistencias en codificaciones (ej: 'YES', 'NO', 'N/A' para booleanos)
  - Valores atípicos (valores negativos o fuera de rango)
  - Conversiones de formato (sexo como 'MALE'/'FEMALE' en vez de binario)
  
#### ¿Por qué datos sintéticos modificados?
Este enfoque retador replica situaciones reales donde los datos llegan "sucios" desde diferentes fuentes, exigiendo robustez en:
- **Limpieza y transformación**: Manejo de múltiples formatos e inconsistencias
- **Imputación inteligente**: Decisión sobre qué imputar y qué eliminar
- **Validación de pipelines**: Asegurar que el proceso sea reproducible y confiable

## 📁 Estructura del Proyecto

```
ProyectoCienciaDatos/
│
├── data/                          # Datos del proyecto
│   ├── raw/                       # Datos originales
│   │   └── heart_failure_clinical_records_dataset.csv
│   └── processed/                 # Datos procesados
│       ├── heart_failure_augmented.csv
│       └── heart_failure_clinical_records_extended_cleaned.csv
│
├── models/                        # Modelos entrenados
│   └── trained/
│       ├── heart_failure_clinical_records-logistic_regression.joblib
│       └── heart_failure_clinical_records-random-forest.joblib
│
├── src/                           # Código fuente
│   ├── data_exploration/
│   │   └── eda.ipynb              # Exploratory Data Analysis
│   ├── notebook/
│   │   ├── Notebook_dataset_base.ipynb
│   │   ├── Notebook_dataset_sintetico.ipynb
│   │   └── Notebook_dataset_sintetico_transformados.ipynb
│   └── preprocessing_modeling/
│       └── preprocessing_and_models.ipynb  # Preprocesamiento y modelado
│
├── reports/                       # Reportes y aplicaciones
│   └── app.py                     # Aplicación Streamlit
│
├── Notebook/                      # Carpeta adicional de notebooks
│
├── .venv/                         # Entorno virtual (no versionar)
├── pyproject.toml                 # Configuración del proyecto y dependencias
├── uv.lock                        # Lock file de dependencias
├── .python-version                # Versión de Python
└── README.md                      # Este archivo

```

## Requisitos

- **Python**: >= 3.13
- **Gestor de paquetes**: uv (recomendado) o pip

### Dependencias Principales

- `pandas`: Manipulación y análisis de datos
- `numpy`: Operaciones numéricas
- `scikit-learn`: Machine learning
- `matplotlib`: Visualización básica
- `seaborn`: Visualización estadística
- `plotly`: Visualizaciones interactivas
- `streamlit`: Aplicación web
- `autoviz`: Visualización automatizada

Ver todas las dependencias en `pyproject.toml`.

## Instalación

### Opción 1: Usando uv (Recomendado)

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd ProyectoCienciaDatos

# Instalar uv si no está instalado
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
# o descargar desde https://github.com/astral-sh/uv

# Crear e instalar el entorno virtual
uv venv
uv pip install -e .
```

### Opción 2: Usando pip tradicional

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd ProyectoCienciaDatos

# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt  # o desde pyproject.toml
```

## Uso

### 1. Exploración de Datos (EDA)

Ejecutar los notebooks de exploración para entender los datos:

```bash
jupyter notebook src/data_exploration/eda.ipynb
```

### 2. Preprocesamiento y Modelado

Ejecutar el notebook principal para entrenar modelos:

```bash
jupyter notebook src/preprocessing_modeling/preprocessing_and_models.ipynb
```

Este notebook incluye:
- Carga de datos originales limpios
- Generación de datos sintéticos con introducción controlada de errores
- Preprocesamiento robusto aplicado a datos con ruido
- Manejo de valores faltantes, inconsistencias de tipo y valores atípicos
- Entrenamiento de modelos con ambos conjuntos de datos
- Evaluación y comparación de rendimiento

### 3. Aplicación Web Interactiva

Desplegar la aplicación Streamlit para hacer predicciones:

```bash
streamlit run reports/app.py
```

La aplicación se abrirá automáticamente en tu navegador (por defecto `http://localhost:8501`).

#### Uso de la App

1. Ingresa los datos clínicos del paciente en los campos correspondientes
2. Haz clic en "Predecir Riesgo"
3. Revisa el resultado y la probabilidad calculada
4. Consulta las recomendaciones generadas

### 4. Usar Modelos Guardados

```python
import joblib
import pandas as pd

# Cargar modelo
model = joblib.load('models/trained/heart_failure_clinical_records-random-forest.joblib')

# Preparar datos de entrada
data = {
    'age': 75,
    'anaemia': 1,
    'creatinine_phosphokinase': 582,
    'diabetes': 0,
    'ejection_fraction': 20,
    'high_blood_pressure': 1,
    'platelets': 265000.0,
    'serum_creatinine': 1.9,
    'serum_sodium': 130,
    'sex': 0,
    'smoking': 0,
    'time': 7
}

# Hacer predicción
df = pd.DataFrame([data])
prediction = model.predict(df)
probability = model.predict_proba(df)

print(f"Predicción: {'Alto Riesgo' if prediction[0] else 'Bajo Riesgo'}")
print(f"Probabilidad de evento fatal: {probability[0][1]:.2%}")
```

## 🤖 Modelos Implementados

### 1. Regresión Logística
- **Ventajas**: Interpretabilidad, rápido entrenamiento, probabilidades calibradas
- **Uso**: Baseline y casos donde la interpretabilidad es crítica

### 2. Random Forest
- **Ventajas**: Alta precisión, manejo automático de no linealidades, importancia de variables
- **Uso**: Modelo principal para predicciones optimizadas

## 📈 Resultados

### Variables Más Importantes
Basado en el análisis de correlación y importancia de características:

1. **serum_creatinine** (correlación ≈ +0.27) - Mayor nivel asociado a mayor riesgo
2. **age** (correlación ≈ +0.22) - Mayor edad implica mayor riesgo
3. **ejection_fraction** (correlación ≈ -0.21) - Menor fracción de eyección aumenta el riesgo
4. **serum_sodium** (correlación ≈ -0.12) - Niveles bajos aumentan el riesgo

### Hallazgos Clínicos

- **Edad**: Los grupos de mayor edad (71-95 años) muestran tasas de mortalidad significativamente más altas
- **Anemia**: Pacientes con anemia presentan mayor proporción de eventos fatales
- **Tabaquismo**: Claramente asociado con mayor mortalidad
- **Diabetes e Hipertensión**: Factores de riesgo moderados
- **Sexo**: Mayor número absoluto de muertes en hombres, principalmente debido al mayor tamaño de la muestra

## 👥 Autores

**Presentado por:**

- Angélica Órtiz Álvarez (`aortiz016@soyudemedellin.edu.co`)
- Jhon Jader Díaz Gómez (`jdiaz510@soyudemedellin.edu.co`)
- Cristian Camilo Ospina Metaute (`cospina149@soyudemedellin.edu.co`)

---

### **Proyecto de Ciencia de Datos**

**Análisis del Dataset de Insuficiencia Cardíaca**  
`heart_failure_clinical_records_dataset.csv` de Kaggle / UCI Irvine



