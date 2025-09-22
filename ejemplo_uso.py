"""
Script de Ejemplo y Prueba del Procesador BLF
===========================================

Script para demostrar el uso del procesador de archivos BLF y realizar
pruebas básicas del sistema.

Uso:
    python ejemplo_uso.py

Autor: Sistema de diagnóstico de autonomía vehicular
Fecha: 2025
"""

import os
import sys
from pathlib import Path

# Agregar directorio actual al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from ProcessorBLF_v2 import ProcessorBLF
import logging

# Configurar logging para ejemplo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ejemplo_basico():
    """
    Ejemplo básico de uso del procesador BLF.
    """
    print("="*60)
    print("EJEMPLO BÁSICO - PROCESADOR BLF")
    print("="*60)
    
    # Rutas de ejemplo (ajustar según tu sistema)
    dbc_path = r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Desarrollo de Software EV\Datos\DBC\IP_JZ - CAN EV.DBC"
    blf_directory = current_dir  # Buscar en directorio actual
    
    print(f"Directorio BLF: {blf_directory}")
    print(f"Archivo DBC: {dbc_path if os.path.exists(dbc_path) else 'No encontrado - se procesarán datos crudos'}")
    
    # Crear procesador
    processor = ProcessorBLF()
    
    # Buscar archivos BLF
    blf_files = processor.find_blf_files(blf_directory)
    print(f"\nArchivos BLF encontrados: {len(blf_files)}")
    
    if not blf_files:
        print("\n⚠️  No se encontraron archivos BLF en el directorio actual")
        print("Para probar el sistema:")
        print("1. Coloca archivos .blf en el directorio del proyecto")
        print("2. O especifica un directorio diferente en el código")
        print("3. Ejecuta: python main_blf_processor.py --gui")
        return
    
    for blf_file in blf_files[:5]:  # Mostrar primeros 5
        print(f"  - {os.path.basename(blf_file)}")
    
    if len(blf_files) > 5:
        print(f"  ... y {len(blf_files) - 5} más")
    
    # Procesar archivos
    print(f"\n🔄 Procesando {len(blf_files)} archivos BLF...")
    
    try:
        # Unificar archivos BLF
        unified_df = processor.unify_blf_files(blf_files)
        
        if unified_df.empty:
            print("❌ No se pudieron procesar los archivos BLF")
            return
        
        print(f"✅ Dataset unificado creado: {len(unified_df):,} mensajes")
        
        # Cargar DBC si existe
        if os.path.exists(dbc_path):
            print(f"\n🔧 Cargando archivo DBC...")
            if processor.load_dbc(dbc_path):
                print("✅ DBC cargado exitosamente")
                
                # Decodificar mensajes
                print(f"\n🔍 Decodificando mensajes...")
                decoded_df = processor.decode_messages(unified_df)
                
                if not decoded_df.empty:
                    print(f"✅ Decodificación completada: {len(decoded_df):,} señales")
                    mostrar_estadisticas(processor, decoded_df)
                    ejemplo_filtrado(processor)
                else:
                    print("❌ No se pudieron decodificar mensajes")
            else:
                print("❌ Error cargando archivo DBC")
        else:
            print(f"\n⚠️  Archivo DBC no encontrado, mostrando datos crudos")
            mostrar_estadisticas_crudas(unified_df)
        
        # Sugerir próximos pasos
        print(f"\n" + "="*60)
        print("PRÓXIMOS PASOS SUGERIDOS")
        print("="*60)
        print("1. 📊 Ejecutar interfaz gráfica:")
        print("   python main_blf_processor.py")
        print("\n2. 💾 Exportar datos procesados:")
        print("   python main_blf_processor.py --cli --blf-dir . --output datos_procesados.csv")
        print("\n3. 🔍 Explorar señales específicas usando la interfaz gráfica")
        
    except Exception as e:
        logger.error(f"Error durante el procesamiento: {str(e)}")
        print(f"❌ Error: {str(e)}")

def mostrar_estadisticas(processor, decoded_df):
    """
    Muestra estadísticas detalladas del dataset decodificado.
    """
    print(f"\n" + "="*40)
    print("ESTADÍSTICAS DEL DATASET")
    print("="*40)
    
    print(f"📊 Registros totales: {len(decoded_df):,}")
    print(f"📨 Mensajes únicos: {decoded_df['message_name'].nunique()}")
    print(f"📡 Señales únicas: {decoded_df['signal_name'].nunique()}")
    
    # Rango temporal
    if not decoded_df.empty:
        time_range = decoded_df['timestamp'].max() - decoded_df['timestamp'].min()
        print(f"⏱️  Duración: {time_range:.2f} segundos")
        print(f"📅 Desde: {decoded_df['datetime'].min()}")
        print(f"📅 Hasta: {decoded_df['datetime'].max()}")
    
    # Mensajes más frecuentes
    print(f"\n📈 Mensajes más frecuentes:")
    message_counts = decoded_df['message_name'].value_counts().head(5)
    for i, (msg, count) in enumerate(message_counts.items(), 1):
        print(f"  {i}. {msg}: {count:,} registros")
    
    # Señales más frecuentes
    print(f"\n🔢 Señales más frecuentes:")
    signal_counts = decoded_df['signal_name'].value_counts().head(5)
    for i, (signal, count) in enumerate(signal_counts.items(), 1):
        print(f"  {i}. {signal}: {count:,} registros")

def mostrar_estadisticas_crudas(unified_df):
    """
    Muestra estadísticas de datos crudos (sin decodificar).
    """
    print(f"\n" + "="*40)
    print("ESTADÍSTICAS DE DATOS CRUDOS")
    print("="*40)
    
    print(f"📊 Mensajes totales: {len(unified_df):,}")
    print(f"📨 IDs únicos: {unified_df['arbitration_id'].nunique()}")
    
    # Rango temporal
    if not unified_df.empty:
        time_range = unified_df['timestamp'].max() - unified_df['timestamp'].min()
        print(f"⏱️  Duración: {time_range:.2f} segundos")
    
    # IDs más frecuentes
    print(f"\n📈 IDs de mensaje más frecuentes:")
    id_counts = unified_df['arbitration_id'].value_counts().head(5)
    for i, (msg_id, count) in enumerate(id_counts.items(), 1):
        print(f"  {i}. 0x{msg_id:X}: {count:,} mensajes")

def ejemplo_filtrado(processor):
    """
    Ejemplo de filtrado y extracción de señales específicas.
    """
    print(f"\n" + "="*40)
    print("EJEMPLO DE FILTRADO DE SEÑALES")
    print("="*40)
    
    # Obtener mensajes disponibles
    messages = processor.get_available_messages()
    
    if not messages:
        print("No hay mensajes decodificados disponibles")
        return
    
    print(f"📋 Mensajes disponibles: {len(messages)}")
    
    # Mostrar primeros mensajes y sus señales
    for i, message in enumerate(messages[:3]):
        signals = processor.get_available_signals(message)
        print(f"\n  {i+1}. {message}")
        print(f"     Señales: {len(signals)}")
        
        # Mostrar algunas señales de ejemplo
        for j, signal in enumerate(signals[:3]):
            print(f"       - {signal}")
        
        if len(signals) > 3:
            print(f"       ... y {len(signals) - 3} más")
        
        # Ejemplo de extracción de datos para el primer mensaje
        if i == 0 and signals:
            print(f"\n🔍 Ejemplo de datos para '{signals[0]}':")
            signal_data = processor.get_signal_data(message, signals[0])
            
            if not signal_data.empty:
                print(f"    Registros: {len(signal_data):,}")
                if 'signal_value' in signal_data.columns:
                    values = signal_data['signal_value'].dropna()
                    if len(values) > 0:
                        try:
                            numeric_values = pd.to_numeric(values, errors='coerce').dropna()
                            if len(numeric_values) > 0:
                                print(f"    Valor mín: {numeric_values.min()}")
                                print(f"    Valor máx: {numeric_values.max()}")
                                print(f"    Valor promedio: {numeric_values.mean():.2f}")
                        except:
                            print(f"    Valores: {values.iloc[0]} ... {values.iloc[-1]}")

def verificar_dependencias():
    """
    Verifica que todas las dependencias estén instaladas.
    """
    dependencias = {
        'cantools': 'cantools',
        'can': 'python-can', 
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib'
    }
    
    print("🔍 Verificando dependencias...")
    faltantes = []
    
    for modulo, paquete in dependencias.items():
        try:
            __import__(modulo)
            print(f"  ✅ {paquete}")
        except ImportError:
            print(f"  ❌ {paquete}")
            faltantes.append(paquete)
    
    if faltantes:
        print(f"\n⚠️  Dependencias faltantes: {', '.join(faltantes)}")
        print("Instalar con: pip install " + " ".join(faltantes))
        return False
    
    print("✅ Todas las dependencias están instaladas")
    return True

def main():
    """
    Función principal del ejemplo.
    """
    print("🚗 Procesador de Archivos BLF - Sistema de Diagnóstico de Autonomía Vehicular")
    print("=" * 80)
    
    # Verificar dependencias
    if not verificar_dependencias():
        print("\nPor favor instala las dependencias faltantes antes de continuar")
        return 1
    
    print()
    
    # Ejecutar ejemplo
    try:
        ejemplo_basico()
        return 0
    except KeyboardInterrupt:
        print("\n\n❌ Ejemplo cancelado por el usuario")
        return 1
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        print(f"\n❌ Error inesperado: {str(e)}")
        return 1

if __name__ == "__main__":
    import pandas as pd
    sys.exit(main())