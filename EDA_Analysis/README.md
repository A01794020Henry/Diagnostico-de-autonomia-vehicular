# 📊 Análisis Exploratorio de Datos (EDA) - Sistema de Diagnóstico Vehicular

Este directorio contiene herramientas especializadas para realizar análisis exploratorio de datos de los sistemas de comunicación CAN y diagnóstico de autonomía vehicular.

## 🎯 Objetivos del EDA

1. **Análisis Descriptivo**: Estadísticas resumidas y distribuciones
2. **Detección de Anomalías**: Identificación de valores atípicos y patrones anómalos
3. **Análisis Temporal**: Tendencias y patrones temporales en señales vehiculares
4. **Correlaciones**: Relaciones entre diferentes señales y sistemas
5. **Calidad de Datos**: Valores faltantes y consistencia

## 📁 Estructura de Archivos

```
EDA_Analysis/
├── README.md                      # Este archivo
├── eda_main.py                    # Script principal de análisis
├── data_quality_analyzer.py       # Análisis de calidad de datos
├── statistical_analyzer.py        # Análisis estadístico avanzado
├── visualization_engine.py        # Motor de visualizaciones
├── temporal_analyzer.py           # Análisis temporal especializado
├── correlation_analyzer.py        # Análisis de correlaciones
├── outlier_detector.py           # Detección de valores atípicos
├── preprocessing_toolkit.py       # Herramientas de preprocesamiento
├── eda_config.py                 # Configuración del análisis
├── notebooks/
│   ├── EDA_Complete_Analysis.ipynb    # Notebook principal
│   ├── Vehicle_Signals_Analysis.ipynb # Análisis de señales vehiculares
│   └── Temporal_Patterns.ipynb        # Patrones temporales
├── reports/                       # Reportes generados
└── visualizations/               # Gráficos y visualizaciones
```

## 🚀 Uso Rápido

### 1. Análisis Completo Automatizado
```python
python eda_main.py --data-source blf --blf-dir "ruta/a/archivos/blf" --dbc-file "archivo.dbc"
```

### 2. Análisis Específico
```python
# Análisis de calidad de datos
python data_quality_analyzer.py --input datos.csv

# Análisis temporal
python temporal_analyzer.py --input datos.csv --time-column timestamp

# Detección de outliers
python outlier_detector.py --input datos.csv --method isolation_forest
```

### 3. Usando Notebooks
1. Abrir `notebooks/EDA_Complete_Analysis.ipynb`
2. Configurar ruta de datos en la primera celda
3. Ejecutar todas las celdas

## 📈 Preguntas que Responde el EDA

- ✅ **Valores Faltantes**: Patrones y estrategias de imputación
- ✅ **Estadísticas Descriptivas**: Medidas de tendencia y dispersión
- ✅ **Valores Atípicos**: Detección multivariada y temporal
- ✅ **Cardinalidad**: Análisis de variables categóricas
- ✅ **Distribuciones**: Normalidad, asimetría y transformaciones
- ✅ **Tendencias Temporales**: Estacionalidad y patrones cíclicos
- ✅ **Correlaciones**: Matrices de correlación y dependencias
- ✅ **Análisis Bivariado**: Relaciones entre pares de variables
- ✅ **Normalización**: Estrategias de escalado y transformación
- ✅ **Desequilibrio de Clases**: Análisis de distribución de categorías

## 🔧 Dependencias

```bash
pip install -r requirements_eda.txt
```

## 📊 Tipos de Visualizaciones

- Histogramas y distribuciones
- Box plots y violin plots
- Matrices de correlación (heatmaps)
- Gráficos de dispersión (scatter plots)
- Series temporales
- Gráficos de anomalías
- Mapas de calor de valores faltantes
- Análisis de componentes principales (PCA)

## 🎨 Configuración de Estilo

El análisis utiliza un estilo visual consistente definido en `eda_config.py`:
- Paleta de colores para vehículos eléctricos
- Configuración de gráficos optimizada para datos CAN
- Templates de reportes profesionales