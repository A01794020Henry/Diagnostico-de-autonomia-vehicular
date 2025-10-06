# Procesador de Archivos DBC a Excel

## 📋 Descripción

Este proyecto proporciona un procesador completo para archivos DBC (Database CAN) que extrae toda la información relevante y la exporta a archivos Excel con formato profesional. Es ideal para ingenieros automotrices, desarrolladores de sistemas CAN y cualquier persona que trabaje con redes de comunicación vehicular.

## ✨ Características

- **Procesamiento completo de archivos DBC**: Extrae mensajes, señales, nodos, atributos y comentarios
- **Exportación a Excel profesional**: Múltiples hojas con formato y estilos
- **Análisis detallado**: Estadísticas completas del archivo DBC
- **Logging completo**: Registro detallado del proceso de conversión
- **Manejo de errores robusto**: Validación y manejo de archivos corruptos
- **Interfaz simple**: Scripts fáciles de usar desde línea de comandos

## 📦 Instalación

### Prerrequisitos

- Python 3.7 o superior
- pip (administrador de paquetes de Python)

### Instalar dependencias

```bash
pip install -r requirements.txt
```

Las dependencias principales son:
- `pandas`: Manipulación de datos
- `openpyxl`: Generación de archivos Excel
- `cantools`: Procesamiento de archivos CAN (opcional)

## 🚀 Uso

### Método 1: Script Simple

```bash
# Uso básico - genera archivo Excel automáticamente
python procesar_dbc.py mi_archivo.dbc

# Especificar archivo de salida
python procesar_dbc.py mi_archivo.dbc resultado.xlsx
```

### Método 2: Script de Demostración Completa

```bash
python demo_dbc_processor.py
```

Este script:
- Crea un archivo DBC de ejemplo si no existe
- Muestra el procesamiento paso a paso
- Genera estadísticas detalladas
- Crea un archivo Excel completo

### Método 3: Uso Programático

```python
from dbc_processor_xlsx import process_dbc_to_excel, DBCProcessor

# Procesamiento directo
success = process_dbc_to_excel("archivo.dbc", "salida.xlsx")

# Uso detallado
processor = DBCProcessor("archivo.dbc")
if processor.process_dbc():
    # Obtener DataFrames para análisis personalizado
    messages_df = processor.get_messages_dataframe()
    signals_df = processor.get_signals_dataframe()
    
    print(f"Mensajes encontrados: {len(messages_df)}")
    print(f"Señales encontradas: {len(signals_df)}")
```

## 📊 Estructura del Archivo Excel Generado

El archivo Excel de salida contiene las siguientes hojas:

### 1. **Resumen**
- Estadísticas generales del archivo DBC
- Contadores de elementos encontrados
- Información del archivo original

### 2. **Mensajes**
- ID del mensaje (decimal y hexadecimal)
- Nombre del mensaje
- Tamaño en bytes
- Nodo transmisor
- Número de señales

### 3. **Señales**
- Información detallada de cada señal
- Factores de escala y offset
- Rangos de valores (mín/máx)
- Unidades de medida
- Configuración de bits

### 4. **Nodos**
- Lista de nodos en la red CAN
- Conteo de mensajes por nodo

### 5. **Tablas_Valores** (si existen)
- Mapeo de valores numéricos a texto descriptivo
- Enumeraciones de señales

### 6. **Atributos** (si existen)
- Atributos personalizados del archivo DBC
- Configuraciones específicas del protocolo

### 7. **Comentarios** (si existen)
- Documentación y comentarios del archivo DBC

## 📁 Archivos del Proyecto

```
📦 Procesador DBC
├── 📄 dbc_processor_xlsx.py      # Clase principal del procesador
├── 📄 demo_dbc_processor.py      # Script de demostración completa
├── 📄 procesar_dbc.py            # Script simple de uso directo
├── 📄 requirements.txt           # Dependencias del proyecto
├── 📄 README_DBC.md             # Esta documentación
└── 📁 Procesamiento_H2/
    └── 📄 Procesamiento.DBC      # Archivo DBC de entrada
```

## 🔧 Características Técnicas

### Elementos DBC Soportados

- ✅ **Mensajes (BO_)**: ID, nombre, tamaño, nodo transmisor
- ✅ **Señales (SG_)**: Configuración completa de bits, factores, rangos
- ✅ **Nodos (BS_)**: Lista de ECUs en la red
- ✅ **Tablas de Valores (VAL_)**: Enumeraciones de señales
- ✅ **Atributos (BA_)**: Configuraciones personalizadas
- ✅ **Comentarios (CM_)**: Documentación interna

### Formatos de Señal Soportados

- Endianness: Big Endian y Little Endian
- Tipos: Signed y Unsigned
- Longitudes: 1 a 64 bits
- Factores de escala y offset
- Rangos de valores personalizados

## 📝 Ejemplo de Archivo DBC

El script de demostración crea automáticamente un archivo DBC de ejemplo con:

```dbc
# Mensajes de ejemplo
BO_ 512 Motor_Status: 8 ECU_Motor
 SG_ RPM : 0|16@1+ (1,0) [0|8000] "rpm" Gateway,ECU_Frenos
 SG_ Temperatura_Motor : 16|8@1+ (1,-40) [-40|150] "°C" Gateway

BO_ 513 Velocidad_Vehiculo: 4 ECU_Transmision
 SG_ Velocidad : 0|16@1+ (0.1,0) [0|300] "km/h" Gateway,ECU_Frenos
```

## 🔍 Logging y Depuración

El procesador incluye logging completo:

```
2025-09-24 10:30:15 - INFO - Archivo DBC cargado exitosamente
2025-09-24 10:30:15 - INFO - Nodos encontrados: 4
2025-09-24 10:30:15 - INFO - Mensajes encontrados: 4
2025-09-24 10:30:15 - INFO - Señales encontradas: 12
2025-09-24 10:30:16 - INFO - Archivo Excel guardado exitosamente
```

Los logs se guardan en `dbc_processing.log` para análisis posterior.

## ⚠️ Manejo de Errores

- **Archivo no encontrado**: Mensaje claro y sugerencias
- **Formato DBC inválido**: Logging detallado del error
- **Problemas de escritura**: Verificación de permisos
- **Memoria insuficiente**: Procesamiento en lotes para archivos grandes

## 🤝 Contribuciones

Este código está diseñado para ser extensible. Áreas de mejora:

1. **Soporte para más elementos DBC**: SIG_VALTYPE_, ENVVAR_DATA_, etc.
2. **Validación de archivos DBC**: Verificación de sintaxis completa
3. **Interfaz gráfica**: GUI para usuarios no técnicos
4. **Exportación adicional**: JSON, CSV, XML

## 📞 Soporte

Para problemas o preguntas:

1. Revisar los logs en `dbc_processing.log`
2. Verificar que el archivo DBC tenga formato válido
3. Confirmar que todas las dependencias están instaladas
4. Ejecutar el script de demostración para verificar el funcionamiento

## 📄 Licencia

Este código se proporciona como ejemplo educativo y puede ser modificado según las necesidades del proyecto.

---

**Desarrollado por GitHub Copilot - Septiembre 2025** 🤖