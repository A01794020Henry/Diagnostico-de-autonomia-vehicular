"""
Script Principal - Procesador y Visualizador de Archivos BLF
===========================================================

Script principal que integra el procesamiento de archivos BLF y la interfaz
gráfica para visualización de señales CAN decodificadas.

Uso:
    python main_blf_processor.py

Características:
- Procesamiento de múltiples archivos BLF
- Decodificación usando archivos DBC  
- Interfaz gráfica interactiva
- Filtrado por mensaje y señal
- Exportación de datos y gráficos

Autor: Sistema de diagnóstico de autonomía vehicular
Fecha: 2025
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Agregar el directorio actual al path para importar nuestros módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from ProcessorBLF_v2 import ProcessorBLF
    from CAN_Visualizer_GUI import CANVisualizerGUI, main as gui_main
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de que todos los archivos estén en el mismo directorio")
    sys.exit(1)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blf_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_example_paths():
    """
    Configura rutas de ejemplo para archivos DBC y BLF.
    """
    # Ruta actual del archivo DBC desde tu código original
    default_dbc = r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Desarrollo de Software EV\Datos\DBC\IP_JZ - CAN EV.DBC"
    
    # Directorio actual para buscar archivos BLF
    current_dir = Path(__file__).parent
    
    # Buscar archivos BLF en el directorio actual y subdirectorios
    blf_files = list(current_dir.glob("*.blf"))
    blf_files.extend(list(current_dir.glob("**/*.blf")))
    
    example_blf_dir = str(current_dir) if blf_files else None
    
    return default_dbc, example_blf_dir

def run_cli_mode(blf_directory, dbc_path=None, output_file=None):
    """
    Ejecuta el procesador en modo línea de comandos.
    
    Args:
        blf_directory (str): Directorio con archivos BLF
        dbc_path (str): Ruta al archivo DBC (opcional)
        output_file (str): Archivo de salida CSV (opcional)
    """
    logger.info("=== INICIANDO PROCESAMIENTO EN MODO CLI ===")
    
    # Crear procesador
    processor = ProcessorBLF()
    
    try:
        # Procesar archivos
        decoded_df = processor.process_directory(blf_directory, dbc_path)
        
        if decoded_df.empty:
            logger.error("No se pudieron procesar los archivos o no se encontraron datos")
            return False
        
        # Mostrar estadísticas
        print("\n" + "="*50)
        print("ESTADÍSTICAS DEL PROCESAMIENTO")
        print("="*50)
        print(f"Total de registros: {len(decoded_df):,}")
        print(f"Mensajes únicos: {decoded_df['message_name'].nunique()}")
        print(f"Señales únicas: {decoded_df['signal_name'].nunique()}")
        
        # Rango temporal
        if not decoded_df.empty:
            time_min = decoded_df['timestamp'].min()
            time_max = decoded_df['timestamp'].max()
            duration = time_max - time_min
            print(f"Duración total: {duration:.2f} segundos")
            print(f"Desde: {decoded_df['datetime'].min()}")
            print(f"Hasta: {decoded_df['datetime'].max()}")
        
        # Mostrar mensajes más frecuentes
        print(f"\nMensajes más frecuentes:")
        message_counts = decoded_df['message_name'].value_counts().head(10)
        for msg, count in message_counts.items():
            print(f"  {msg}: {count:,} registros")
        
        # Guardar archivo si se especifica
        if output_file:
            if processor.save_dataset(output_file, 'decoded'):
                print(f"\nDatos guardados en: {output_file}")
            else:
                logger.error("Error guardando archivo de salida")
                return False
        
        logger.info("Procesamiento completado exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"Error durante el procesamiento: {str(e)}")
        return False

def run_gui_mode():
    """
    Ejecuta la interfaz gráfica.
    """
    logger.info("=== INICIANDO MODO INTERFAZ GRÁFICA ===")
    
    try:
        gui_main()
    except Exception as e:
        logger.error(f"Error ejecutando interfaz gráfica: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Verifica que tengas PyQt5 instalado: pip install PyQt5")

def main():
    """
    Función principal del script.
    """
    parser = argparse.ArgumentParser(
        description="Procesador y Visualizador de archivos BLF para diagnóstico de autonomía vehicular",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Modo interfaz gráfica (recomendado)
  python main_blf_processor.py

  # Modo línea de comandos básico
  python main_blf_processor.py --cli --blf-dir "C:/ruta/archivos/blf"

  # Con archivo DBC y salida CSV
  python main_blf_processor.py --cli --blf-dir "C:/ruta/blf" --dbc "archivo.dbc" --output "resultado.csv"
        """
    )
    
    parser.add_argument(
        '--cli', 
        action='store_true',
        help='Ejecutar en modo línea de comandos (sin interfaz gráfica)'
    )
    
    parser.add_argument(
        '--blf-dir', 
        type=str,
        help='Directorio que contiene archivos BLF'
    )
    
    parser.add_argument(
        '--dbc', 
        type=str,
        help='Ruta al archivo DBC para decodificación'
    )
    
    parser.add_argument(
        '--output', 
        type=str,
        help='Archivo de salida CSV para guardar datos procesados'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada de procesamiento'
    )
    
    args = parser.parse_args()
    
    # Configurar nivel de logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Obtener rutas de ejemplo
    default_dbc, example_blf_dir = setup_example_paths()
    
    if args.cli:
        # Modo línea de comandos
        blf_directory = args.blf_dir
        
        if not blf_directory:
            if example_blf_dir:
                print(f"No se especificó directorio BLF, usando directorio actual: {example_blf_dir}")
                blf_directory = example_blf_dir
            else:
                print("Error: Debe especificar --blf-dir en modo CLI")
                print("Ejemplo: python main_blf_processor.py --cli --blf-dir 'C:/ruta/archivos/blf'")
                return 1
        
        if not os.path.exists(blf_directory):
            print(f"Error: El directorio {blf_directory} no existe")
            return 1
        
        dbc_path = args.dbc or (default_dbc if os.path.exists(default_dbc) else None)
        
        if dbc_path:
            print(f"Usando archivo DBC: {dbc_path}")
        else:
            print("Procesando sin archivo DBC (solo datos crudos)")
        
        success = run_cli_mode(blf_directory, dbc_path, args.output)
        return 0 if success else 1
        
    else:
        # Modo interfaz gráfica (por defecto)
        print("Iniciando interfaz gráfica...")
        print("Si prefieres modo línea de comandos, usa: python main_blf_processor.py --cli --help")
        
        run_gui_mode()
        return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        print(f"Error inesperado: {str(e)}")
        sys.exit(1)