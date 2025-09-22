"""
Procesador de archivos BLF para análisis de autonomía vehicular
=============================================================

Este script procesa múltiples archivos BLF de pruebas de ruta CAN bus,
los unifica en un dataset único ordenado cronológicamente, decodifica
los mensajes usando archivos DBC y proporciona una interfaz gráfica
interactiva para visualizar las señales.

Autor: Sistema de diagnóstico de autonomía vehicular
Fecha: 2025
"""

import os
import glob
import cantools
import can
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessorBLF:
    """
    Clase principal para procesar archivos BLF y generar datasets unificados.
    """
    
    def __init__(self, dbc_path: str = None, dbc_paths: List[str] = None):
        """
        Inicializa el procesador BLF.
        
        Args:
            dbc_path (str): Ruta al archivo DBC para decodificación (compatibilidad)
            dbc_paths (List[str]): Lista de rutas a archivos DBC para decodificación múltiple
        """
        # Compatibilidad con versión anterior (un solo archivo DBC)
        self.dbc_path = dbc_path  # Mantener para compatibilidad
        self.database = None      # Mantener para compatibilidad
        
        # Nuevos atributos para múltiples archivos DBC
        self.dbc_paths = dbc_paths or []
        self.databases = {}  # Diccionario: nombre_archivo -> database
        self.loaded_dbc_files = []  # Lista de archivos DBC cargados exitosamente
        
        self.unified_dataset = pd.DataFrame()
        self.decoded_dataset = pd.DataFrame()
        
        # Cargar archivo DBC individual si se especifica (compatibilidad)
        if dbc_path and os.path.exists(dbc_path):
            self.load_dbc(dbc_path)
        
        # Cargar múltiples archivos DBC si se especifican
        if dbc_paths:
            self.load_multiple_dbc(dbc_paths)
    
    def load_dbc(self, dbc_path: str) -> bool:
        """
        Carga el archivo DBC para decodificación de mensajes.
        
        Args:
            dbc_path (str): Ruta al archivo DBC
            
        Returns:
            bool: True si se cargó exitosamente, False en caso contrario
        """
        try:
            database = cantools.database.load_file(dbc_path)
            
            # Mantener compatibilidad con versión anterior
            self.database = database
            self.dbc_path = dbc_path
            
            # Agregar a la nueva estructura de múltiples archivos
            dbc_filename = os.path.basename(dbc_path)
            self.databases[dbc_filename] = database
            
            # Agregar a la lista de archivos cargados si no está ya
            if dbc_path not in self.loaded_dbc_files:
                self.loaded_dbc_files.append(dbc_path)
            
            # Agregar a dbc_paths si no está ya
            if dbc_path not in self.dbc_paths:
                self.dbc_paths.append(dbc_path)
            
            logger.info(f"DBC cargado exitosamente: {dbc_path}")
            logger.info(f"Mensajes disponibles: {len(database.messages)}")
            return True
        except Exception as e:
            logger.error(f"Error cargando DBC {dbc_path}: {str(e)}")
            return False
    
    def load_multiple_dbc(self, dbc_paths: List[str]) -> Dict[str, bool]:
        """
        Carga múltiples archivos DBC para decodificación de mensajes.
        
        Args:
            dbc_paths (List[str]): Lista de rutas a archivos DBC
            
        Returns:
            Dict[str, bool]: Diccionario con el resultado de carga para cada archivo
        """
        results = {}
        successful_loads = 0
        
        logger.info(f"Iniciando carga de {len(dbc_paths)} archivos DBC...")
        
        for dbc_path in dbc_paths:
            if not os.path.exists(dbc_path):
                logger.warning(f"Archivo DBC no encontrado: {dbc_path}")
                results[dbc_path] = False
                continue
            
            try:
                database = cantools.database.load_file(dbc_path)
                dbc_filename = os.path.basename(dbc_path)
                
                # Agregar a la estructura de múltiples archivos
                self.databases[dbc_filename] = database
                
                # Agregar a la lista de archivos cargados si no está ya
                if dbc_path not in self.loaded_dbc_files:
                    self.loaded_dbc_files.append(dbc_path)
                
                # Actualizar lista de paths si no está ya
                if dbc_path not in self.dbc_paths:
                    self.dbc_paths.append(dbc_path)
                
                # Mantener compatibilidad: el primer archivo exitoso será el principal
                if successful_loads == 0:
                    self.database = database
                    self.dbc_path = dbc_path
                
                logger.info(f"DBC cargado: {dbc_path} ({len(database.messages)} mensajes)")
                results[dbc_path] = True
                successful_loads += 1
                
            except Exception as e:
                logger.error(f"Error cargando DBC {dbc_path}: {str(e)}")
                results[dbc_path] = False
        
        logger.info(f"Carga completada: {successful_loads}/{len(dbc_paths)} archivos DBC exitosos")
        
        # Log de resumen de mensajes totales disponibles
        total_messages = sum(len(db.messages) for db in self.databases.values())
        logger.info(f"Total de mensajes disponibles: {total_messages}")
        
        return results
    
    def find_blf_files(self, directory: str, pattern: str = "*.blf") -> List[str]:
        """
        Encuentra todos los archivos BLF en un directorio.
        
        Args:
            directory (str): Directorio donde buscar archivos BLF
            pattern (str): Patrón de búsqueda (por defecto "*.blf")
            
        Returns:
            List[str]: Lista de rutas de archivos BLF encontrados
        """
        blf_files = glob.glob(os.path.join(directory, pattern))
        blf_files.extend(glob.glob(os.path.join(directory, "**", pattern), recursive=True))
        
        logger.info(f"Encontrados {len(blf_files)} archivos BLF en {directory}")
        return sorted(blf_files)
    
    def get_file_creation_time(self, file_path: str) -> datetime:
        """
        Obtiene la fecha de creación del archivo para ordenamiento.
        
        Args:
            file_path (str): Ruta del archivo
            
        Returns:
            datetime: Fecha de creación del archivo
        """
        try:
            return datetime.fromtimestamp(os.path.getctime(file_path))
        except:
            return datetime.now()
    
    def read_blf_file(self, blf_path: str) -> pd.DataFrame:
        """
        Lee un archivo BLF individual y lo convierte a DataFrame.
        
        Args:
            blf_path (str): Ruta al archivo BLF
            
        Returns:
            pd.DataFrame: DataFrame con los datos del archivo BLF
        """
        messages_data = []
        
        try:
            with can.BLFReader(blf_path) as blf_reader:
                for msg in blf_reader:
                    if hasattr(msg, 'arbitration_id'):
                        message_info = {
                            'timestamp': msg.timestamp,
                            'arbitration_id': msg.arbitration_id,
                            'dlc': len(msg.data) if msg.data else 0,
                            'data': msg.data.hex() if msg.data else '',
                            'data_bytes': msg.data if msg.data else b'',
                            'file_source': os.path.basename(blf_path)
                        }
                        messages_data.append(message_info)
            
            df = pd.DataFrame(messages_data)
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                logger.info(f"Procesado {blf_path}: {len(df)} mensajes")
            
            return df
            
        except Exception as e:
            logger.error(f"Error procesando {blf_path}: {str(e)}")
            return pd.DataFrame()
    
    def unify_blf_files(self, blf_files: List[str]) -> pd.DataFrame:
        """
        Unifica múltiples archivos BLF en un solo dataset ordenado cronológicamente.
        
        Args:
            blf_files (List[str]): Lista de rutas de archivos BLF
            
        Returns:
            pd.DataFrame: Dataset unificado y ordenado
        """
        all_dataframes = []
        
        # Ordenar archivos por fecha de creación
        blf_files_sorted = sorted(blf_files, key=self.get_file_creation_time)
        
        logger.info("Iniciando procesamiento de archivos BLF...")
        for i, blf_file in enumerate(blf_files_sorted, 1):
            logger.info(f"Procesando archivo {i}/{len(blf_files_sorted)}: {os.path.basename(blf_file)}")
            df = self.read_blf_file(blf_file)
            if not df.empty:
                all_dataframes.append(df)
        
        if all_dataframes:
            # Concatenar todos los DataFrames
            unified_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Ordenar por timestamp
            unified_df = unified_df.sort_values('timestamp').reset_index(drop=True)
            
            self.unified_dataset = unified_df
            logger.info(f"Dataset unificado creado: {len(unified_df)} mensajes totales")
            logger.info(f"Rango temporal: {unified_df['datetime'].min()} a {unified_df['datetime'].max()}")
            
            return unified_df
        else:
            logger.warning("No se pudieron procesar archivos BLF")
            return pd.DataFrame()
    
    def decode_messages(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Decodifica los mensajes CAN usando todos los archivos DBC cargados.
        
        Args:
            df (pd.DataFrame): DataFrame con mensajes a decodificar
            
        Returns:
            pd.DataFrame: DataFrame con mensajes decodificados
        """
        if df is None:
            df = self.unified_dataset
        
        if df.empty:
            logger.warning("No hay datos para decodificar")
            return pd.DataFrame()
        
        # Verificar si hay bases de datos DBC cargadas
        if not self.databases and not self.database:
            logger.warning("No hay bases de datos DBC cargadas")
            return df
        
        decoded_data = []
        successful_decodings = 0
        failed_decodings = 0
        
        logger.info("Iniciando decodificación de mensajes...")
        if self.databases:
            logger.info(f"Usando {len(self.databases)} archivos DBC cargados")
        
        for index, row in df.iterrows():
            if index % 10000 == 0:
                logger.info(f"Decodificando mensaje {index}/{len(df)}")
            
            decoded_successfully = False
            
            # Intentar decodificar con todas las bases de datos disponibles
            databases_to_try = []
            
            # Agregar bases de datos múltiples
            if self.databases:
                databases_to_try.extend(list(self.databases.values()))
            
            # Agregar base de datos principal para compatibilidad (si no está ya en múltiples)
            if self.database and self.database not in databases_to_try:
                databases_to_try.append(self.database)
            
            # Intentar decodificar con cada base de datos hasta encontrar coincidencia
            for database in databases_to_try:
                try:
                    message = database.get_message_by_frame_id(row['arbitration_id'])
                    decoded_msg = database.decode_message(row['arbitration_id'], row['data_bytes'])
                    
                    # Crear entrada para cada señal decodificada
                    for signal_name, signal_value in decoded_msg.items():
                        signal_entry = {
                            'timestamp': row['timestamp'],
                            'datetime': row['datetime'],
                            'message_name': message.name,
                            'message_id': row['arbitration_id'],
                            'signal_name': signal_name,
                            'signal_value': signal_value,
                            'file_source': row['file_source']
                        }
                        
                        # Agregar información adicional de la señal si está disponible
                        try:
                            signal_obj = message.get_signal_by_name(signal_name)
                            signal_entry.update({
                                'unit': signal_obj.unit or '',
                                'minimum': signal_obj.minimum,
                                'maximum': signal_obj.maximum,
                                'scale': signal_obj.scale,
                                'offset': signal_obj.offset
                            })
                        except:
                            signal_entry.update({
                                'unit': '',
                                'minimum': None,
                                'maximum': None,
                                'scale': 1,
                                'offset': 0
                            })
                        
                        decoded_data.append(signal_entry)
                    
                    decoded_successfully = True
                    successful_decodings += 1
                    break  # Salir del bucle de bases de datos si se decodificó exitosamente
                    
                except Exception:
                    # Continuar con la siguiente base de datos
                    continue
            
            # Si no se pudo decodificar con ninguna base de datos, mantener como datos crudos
            if not decoded_successfully:
                raw_entry = {
                    'timestamp': row['timestamp'],
                    'datetime': row['datetime'],
                    'message_name': f'Unknown_0x{row["arbitration_id"]:X}',
                    'message_id': row['arbitration_id'],
                    'signal_name': 'raw_data',
                    'signal_value': row['data'],
                    'file_source': row['file_source'],
                    'unit': '',
                    'minimum': None,
                    'maximum': None,
                    'scale': 1,
                    'offset': 0
                }
                decoded_data.append(raw_entry)
                failed_decodings += 1
        
        decoded_df = pd.DataFrame(decoded_data)
        self.decoded_dataset = decoded_df
        
        logger.info(f"Decodificación completada: {len(decoded_df)} señales decodificadas")
        logger.info(f"Mensajes exitosamente decodificados: {successful_decodings}")
        logger.info(f"Mensajes no decodificados (datos crudos): {failed_decodings}")
        
        if not decoded_df.empty:
            logger.info(f"Mensajes únicos: {decoded_df['message_name'].nunique()}")
            logger.info(f"Señales únicas: {decoded_df['signal_name'].nunique()}")
        
        return decoded_df
    
    def get_available_messages(self) -> List[str]:
        """
        Obtiene la lista de mensajes disponibles en el dataset decodificado.
        
        Returns:
            List[str]: Lista de nombres de mensajes únicos
        """
        if not self.decoded_dataset.empty:
            return sorted(self.decoded_dataset['message_name'].unique().tolist())
        return []
    
    def get_available_signals(self, message_name: str = None) -> List[str]:
        """
        Obtiene la lista de señales disponibles, opcionalmente filtradas por mensaje.
        
        Args:
            message_name (str): Nombre del mensaje para filtrar señales
            
        Returns:
            List[str]: Lista de nombres de señales
        """
        if self.decoded_dataset.empty:
            return []
        
        if message_name:
            filtered_df = self.decoded_dataset[self.decoded_dataset['message_name'] == message_name]
            return sorted(filtered_df['signal_name'].unique().tolist())
        else:
            return sorted(self.decoded_dataset['signal_name'].unique().tolist())
    
    def get_loaded_dbc_info(self) -> Dict[str, Dict]:
        """
        Obtiene información sobre todos los archivos DBC cargados.
        
        Returns:
            Dict[str, Dict]: Información de cada archivo DBC cargado
        """
        dbc_info = {}
        
        for dbc_filename, database in self.databases.items():
            dbc_info[dbc_filename] = {
                'messages_count': len(database.messages),
                'message_names': [msg.name for msg in database.messages],
                'total_signals': sum(len(msg.signals) for msg in database.messages)
            }
        
        return dbc_info
    
    def get_loaded_dbc_count(self) -> int:
        """
        Obtiene el número de archivos DBC cargados.
        
        Returns:
            int: Número de archivos DBC cargados
        """
        return len(self.databases)
    
    def get_all_available_messages_from_dbc(self) -> List[str]:
        """
        Obtiene todos los mensajes disponibles de todas las bases de datos DBC cargadas.
        
        Returns:
            List[str]: Lista de nombres de mensajes únicos de todas las DBCs
        """
        all_messages = set()
        
        for database in self.databases.values():
            for message in database.messages:
                all_messages.add(message.name)
        
        # Agregar de la base de datos principal si existe y no está en múltiples
        if self.database and self.database not in self.databases.values():
            for message in self.database.messages:
                all_messages.add(message.name)
        
        return sorted(list(all_messages))
    
    def get_signal_data(self, message_name: str = None, signal_name: str = None) -> pd.DataFrame:
        """
        Obtiene los datos de una señal específica o conjunto de señales.
        
        Args:
            message_name (str): Nombre del mensaje (opcional)
            signal_name (str): Nombre de la señal (opcional)
            
        Returns:
            pd.DataFrame: DataFrame filtrado con los datos de la señal
        """
        df = self.decoded_dataset.copy()
        
        if message_name:
            df = df[df['message_name'] == message_name]
        
        if signal_name:
            df = df[df['signal_name'] == signal_name]
        
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def save_dataset(self, filename: str, dataset_type: str = 'decoded') -> bool:
        """
        Guarda el dataset en un archivo CSV.
        
        Args:
            filename (str): Nombre del archivo de salida
            dataset_type (str): Tipo de dataset ('unified' o 'decoded')
            
        Returns:
            bool: True si se guardó exitosamente
        """
        try:
            if dataset_type == 'unified':
                df = self.unified_dataset
            else:
                df = self.decoded_dataset
            
            if df.empty:
                logger.warning(f"No hay datos {dataset_type} para guardar")
                return False
            
            df.to_csv(filename, index=False)
            logger.info(f"Dataset {dataset_type} guardado en {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando dataset: {str(e)}")
            return False
    
    def process_directory(self, blf_directory: str, dbc_path: str = None, dbc_paths: List[str] = None) -> pd.DataFrame:
        """
        Procesa todos los archivos BLF en un directorio.
        
        Args:
            blf_directory (str): Directorio con archivos BLF
            dbc_path (str): Ruta opcional al archivo DBC (compatibilidad)
            dbc_paths (List[str]): Lista opcional de rutas a archivos DBC
            
        Returns:
            pd.DataFrame: Dataset procesado y decodificado
        """
        # Cargar archivo DBC individual (compatibilidad)
        if dbc_path:
            self.load_dbc(dbc_path)
        
        # Cargar múltiples archivos DBC
        if dbc_paths:
            self.load_multiple_dbc(dbc_paths)
        
        # Encontrar archivos BLF
        blf_files = self.find_blf_files(blf_directory)
        
        if not blf_files:
            logger.warning(f"No se encontraron archivos BLF en {blf_directory}")
            return pd.DataFrame()
        
        # Unificar archivos BLF
        unified_df = self.unify_blf_files(blf_files)
        
        if unified_df.empty:
            return pd.DataFrame()
        
        # Decodificar mensajes
        decoded_df = self.decode_messages(unified_df)
        
        return decoded_df


def main():
    """
    Función principal de demostración.
    """
    # Configuración de ejemplo
    dbc_path = r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Desarrollo de Software EV\Datos\DBC\IP_JZ - CAN EV.DBC"
    blf_directory = r"C:\ruta\a\archivos\blf"  # Cambiar por la ruta real
    
    # Crear procesador
    processor = ProcessorBLF()
    
    # Verificar si existe el archivo DBC
    if os.path.exists(dbc_path):
        logger.info("Procesando con archivo DBC...")
        decoded_df = processor.process_directory(blf_directory, dbc_path)
    else:
        logger.warning("Archivo DBC no encontrado, procesando solo datos crudos...")
        blf_files = processor.find_blf_files(blf_directory)
        unified_df = processor.unify_blf_files(blf_files)
    
    # Mostrar estadísticas si hay datos
    if not processor.decoded_dataset.empty:
        print("\n=== ESTADÍSTICAS DEL DATASET ===")
        print(f"Total de registros: {len(processor.decoded_dataset)}")
        print(f"Mensajes únicos: {len(processor.get_available_messages())}")
        print(f"Señales únicas: {len(processor.get_available_signals())}")
        print(f"\nMensajes disponibles:")
        for msg in processor.get_available_messages()[:10]:  # Mostrar primeros 10
            signals = processor.get_available_signals(msg)
            print(f"  {msg}: {len(signals)} señales")


if __name__ == "__main__":
    main()