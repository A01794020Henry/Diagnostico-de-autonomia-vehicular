# Procesador de Archivos BLF para Diagnóstico de Autonomía Vehicular

Sistema completo para procesar archivos BLF de pruebas de ruta CAN bus, unificarlos, decodificarlos usando archivos DBC y visualizar las señales de forma interactiva.

## 🚗 Características

- **Procesamiento de múltiples archivos BLF**: Unifica archivos BLF ordenándolos cronológicamente
- **Decodificación DBC múltiple**: Soporte para cargar y usar múltiples archivos DBC simultáneamente
- **Decodificación inteligente**: Intenta decodificar mensajes con todas las bases de datos DBC disponibles
- **Interfaz gráfica interactiva**: Visualiza señales con filtros por mensaje y señal
- **Exportación**: Guarda datasets procesados y gráficos generados
- **Modo CLI avanzado**: Procesamiento por lotes con soporte para múltiples DBCs

## 📋 Requisitos

### Dependencias Python
```bash
pip install cantools pandas matplotlib PyQt5 pyqtgraph python-can numpy
```

### Archivos necesarios

- **Archivos BLF**: Logs de pruebas de ruta CAN bus (formato .blf)
- **Archivos DBC**: Una o múltiples bases de datos CAN para decodificación (formato .dbc) - *opcional*

## 🚀 Uso Rápido

### Interfaz Gráfica (Recomendado)
```bash
python main_blf_processor.py
```

### Línea de Comandos
```bash
# Procesamiento básico
python main_blf_processor.py --cli --blf-dir "C:/ruta/archivos/blf"

# Con archivo DBC y exportación
python main_blf_processor.py --cli --blf-dir "C:/ruta/blf" --dbc "archivo.dbc" --output "resultado.csv"
```

## 📁 Estructura del Proyecto

```
├── main_blf_processor.py      # Script principal
├── ProcessorBLF_v2.py         # Clase para procesamiento de BLF
├── CAN_Visualizer_GUI.py      # Interfaz gráfica
├── Procesador_BLF.py          # Script original (obsoleto)
├── README.md                  # Este archivo
└── blf_processor.log          # Log de ejecución
```

## � Múltiples Archivos DBC

### Nuevas Capacidades

- **Carga múltiple**: Soporte para cargar varios archivos DBC simultáneamente
- **Decodificación inteligente**: Intenta decodificar cada mensaje con todas las bases de datos hasta encontrar coincidencia
- **Cobertura mejorada**: Mayor probabilidad de decodificar mensajes de diferentes sistemas

### Interfaz Gráfica

1. **Agregar archivos DBC**:
   - Usar botón "Agregar DBC" para seleccionar archivos individuales
   - Los archivos aparecen en la lista con nombre y tooltip de ruta completa
   - Usar "Quitar DBC" y "Limpiar Todo" para gestionar la lista

2. **Procesamiento**:
   - El sistema intenta decodificar con todas las bases de datos cargadas
   - Estadísticas muestran cobertura y éxito de decodificación

### Línea de Comandos

```bash
# Múltiples archivos DBC individuales
python main_blf_processor.py --cli --blf-dir "archivos_blf" --dbc "vehiculo.dbc" --dbc "motor.dbc" --dbc "bateria.dbc"

# Desde archivo de lista
python main_blf_processor.py --cli --blf-dir "archivos_blf" --dbc-list "lista_dbc.txt"

# Combinando ambos métodos
python main_blf_processor.py --cli --blf-dir "archivos_blf" --dbc "principal.dbc" --dbc-list "adicionales.txt"
```

### Archivo de Lista DBC

Crear un archivo de texto (ej: `lista_dbc.txt`) con rutas de archivos DBC:

```text
# Archivos DBC del proyecto
C:/proyecto/dbc/vehiculo_principal.dbc
C:/proyecto/dbc/sistema_motor.dbc
C:/proyecto/dbc/bateria_bms.dbc
# Líneas que empiecen con # son comentarios
C:/proyecto/dbc/diagnosticos.dbc
```

## �🔧 Guía de Uso

### 1. Interfaz Gráfica

1. **Ejecutar el programa**:
   ```bash
   python main_blf_processor.py
   ```

2. **Cargar archivos**:
   - Selecciona el directorio que contiene archivos BLF
   - Opcionalmente, selecciona un archivo DBC para decodificación
   - Haz clic en "Procesar Archivos"

3. **Visualizar señales**:
   - Selecciona un mensaje del dropdown
   - Elige señales de la lista
   - Haz clic en "Agregar Señal" para graficar
   - Usa las pestañas "Gráficos" y "Datos" para diferentes vistas

4. **Exportar resultados**:
   - "Exportar a CSV": Guarda el dataset procesado
   - "Exportar Gráfico": Guarda la visualización como imagen

### 2. Línea de Comandos

```bash
# Ayuda completa
python main_blf_processor.py --help

# Opciones principales
--cli              # Modo línea de comandos
--blf-dir DIR      # Directorio con archivos BLF
--dbc FILE         # Archivo DBC (opcional)
--output FILE      # Archivo CSV de salida
--verbose          # Información detallada
```

## 📊 Funcionalidades del Procesador

### ProcessorBLF_v2.py

**Clase principal `ProcessorBLF`**:
- `load_dbc(path)`: Carga archivo DBC
- `find_blf_files(directory)`: Encuentra archivos BLF
- `unify_blf_files(files)`: Unifica y ordena archivos BLF
- `decode_messages(df)`: Decodifica mensajes usando DBC
- `get_signal_data(message, signal)`: Extrae datos de señal específica
- `save_dataset(filename)`: Exporta a CSV

**Características**:
- ✅ Ordenamiento cronológico automático
- ✅ Manejo robusto de errores
- ✅ Progreso de procesamiento
- ✅ Estadísticas detalladas
- ✅ Soporte para múltiples formatos

### CAN_Visualizer_GUI.py

**Interfaz gráfica completa**:
- 🖥️ Panel de control lateral
- 📊 Visualización interactiva con PyQtGraph
- 🔍 Filtros por mensaje y señal
- 📈 Zoom y pan en gráficos
- 📋 Vista tabular de datos
- 💾 Exportación de datos y gráficos

## 🎯 Casos de Uso Típicos

### 1. Análisis de Autonomía Vehicular
```python
# Procesar logs de prueba completa
processor = ProcessorBLF()
dataset = processor.process_directory("logs_prueba_ruta", "vehicle_can.dbc")

# Analizar señal específica
battery_data = processor.get_signal_data(message_name="BMS_Info", signal_name="Battery_SOC")
```

### 2. Análisis con Múltiples Sistemas DBC
```python
# Procesar con múltiples archivos DBC para diferentes sistemas
processor = ProcessorBLF()
dbc_files = ["vehiculo.dbc", "motor.dbc", "bateria.dbc", "diagnosticos.dbc"]
results = processor.load_multiple_dbc(dbc_files)

# Procesar datos con cobertura completa
dataset = processor.process_directory("logs_prueba", dbc_paths=dbc_files)

# Obtener información de cobertura
dbc_info = processor.get_loaded_dbc_info()
print(f"Archivos DBC cargados: {len(dbc_info)}")
```

### 3. Diagnóstico de Fallas
```python
# Buscar anomalías en señales específicas
motor_temp = processor.get_signal_data(signal_name="Motor_Temperature")
anomalies = motor_temp[motor_temp['signal_value'] > 80]
```

### 4. Generación de Reportes
```bash
# Procesar y exportar automáticamente
python main_blf_processor.py --cli --blf-dir "logs_diarios" --dbc "config.dbc" --output "reporte_$(date).csv"
```

## 📈 Ejemplo de Salida

### Estadísticas de Procesamiento
```
==================================================
ESTADÍSTICAS DEL PROCESAMIENTO
==================================================
Total de registros: 125,847
Mensajes únicos: 23
Señales únicas: 156
Duración total: 1,847.25 segundos
Desde: 2025-09-21 14:30:15.123
Hasta: 2025-09-21 15:00:42.373

Mensajes más frecuentes:
  BMS_Status: 18,234 registros
  Motor_Control: 15,892 registros
  Vehicle_Speed: 12,456 registros
```

### Dataset Decodificado
| timestamp | datetime | message_name | signal_name | signal_value | unit |
|-----------|----------|--------------|-------------|--------------|------|
| 1632225015.123 | 2025-09-21 14:30:15.123 | BMS_Status | Battery_SOC | 87.5 | % |
| 1632225015.124 | 2025-09-21 14:30:15.124 | Motor_Control | Torque_Request | 150.2 | Nm |

## 🐛 Solución de Problemas

### Error: "No se encontraron archivos BLF"
- Verifica que el directorio contenga archivos .blf
- Asegúrate de que los archivos no estén corruptos

### Error: "Error cargando DBC"
- Verifica que el archivo .dbc existe y es válido
- El programa puede funcionar sin DBC (datos crudos)

### Error: "PyQt5 not found"
```bash
pip install PyQt5 pyqtgraph
```

### Rendimiento lento con archivos grandes
- Usa modo CLI para archivos muy grandes
- Considera procesar archivos por lotes

## 🔄 Actualizaciones y Mantenimiento

### Log de Cambios
- **v2.0**: Interfaz gráfica completa
- **v1.5**: Decodificación DBC robusta  
- **v1.0**: Procesamiento básico de BLF

### Mejoras Futuras
- [ ] Soporte para más formatos de logs
- [ ] Análisis estadístico avanzado
- [ ] Dashboard web interactivo
- [ ] Alertas automáticas por umbrales

## 📞 Soporte

Para problemas, mejoras o consultas:
1. Revisa este README
2. Consulta los logs en `blf_processor.log`
3. Ejecuta con `--verbose` para más detalles

---

**Sistema de Diagnóstico de Autonomía Vehicular - 2025**