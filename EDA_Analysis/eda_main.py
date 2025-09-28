"""
Script Principal de Análisis Exploratorio de Datos (EDA)
======================================================

Este script coordina todo el análisis exploratorio de datos del sistema
de diagnóstico vehicular, desde la carga hasta la generación de reportes.

Uso:
    python eda_main.py --data-source blf --blf-dir "ruta" --dbc-file "archivo.dbc"
    python eda_main.py --data-source csv --csv-file "datos.csv"

Autor: Sistema de diagnóstico de autonomía vehicular
Fecha: Septiembre 2025
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Agregar directorio padre al path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Importar módulos del sistema principal
try:
    from ProcessorBLF_v2 import ProcessorBLF
except ImportError as e:
    print(f"Error importando ProcessorBLF_v2: {e}")
    print("Asegúrate de que el archivo ProcessorBLF_v2.py esté en el directorio padre")
    sys.exit(1)

# Importar módulos EDA locales
from data_quality_analyzer import DataQualityAnalyzer
from statistical_analyzer import StatisticalAnalyzer
from visualization_engine import VisualizationEngine
from temporal_analyzer import TemporalAnalyzer
from correlation_analyzer import CorrelationAnalyzer
from outlier_detector import OutlierDetector
from preprocessing_toolkit import PreprocessingToolkit
from eda_config import EDAConfig

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('eda_analysis.log')
    ]
)
logger = logging.getLogger(__name__)


class EDAMainAnalyzer:
    """Coordinador principal del análisis exploratorio de datos."""
    
    def __init__(self, config=None):
        """
        Inicializar el analizador principal.
        
        Args:
            config: Configuración personalizada para el análisis
        """
        self.config = config or EDAConfig()
        self.data = None
        self.metadata = {}
        
        # Inicializar analizadores especializados
        self.data_quality = DataQualityAnalyzer(self.config)
        self.statistical = StatisticalAnalyzer(self.config)
        self.visualization = VisualizationEngine(self.config)
        self.temporal = TemporalAnalyzer(self.config)
        self.correlation = CorrelationAnalyzer(self.config)
        self.outlier = OutlierDetector(self.config)
        self.preprocessing = PreprocessingToolkit(self.config)
        
        # Crear directorios de salida
        self._setup_output_directories()
    
    def _setup_output_directories(self):
        """Crear directorios necesarios para el análisis."""
        directories = [
            self.config.reports_dir,
            self.config.visualizations_dir,
            os.path.join(self.config.visualizations_dir, 'distributions'),
            os.path.join(self.config.visualizations_dir, 'correlations'),
            os.path.join(self.config.visualizations_dir, 'temporal'),
            os.path.join(self.config.visualizations_dir, 'outliers'),
            os.path.join(self.config.visualizations_dir, 'quality')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info(f"Directorios de salida configurados en: {self.config.output_dir}")
    
    def load_data_from_blf(self, blf_dir, dbc_file=None, dbc_list=None):
        """
        Cargar datos desde archivos BLF.
        
        Args:
            blf_dir: Directorio con archivos BLF
            dbc_file: Archivo DBC único
            dbc_list: Lista de archivos DBC
        """
        logger.info(f"Cargando datos BLF desde: {blf_dir}")
        
        try:
            # Crear procesador BLF
            processor = ProcessorBLF()
            
            # Cargar archivos DBC si se especifican
            if dbc_file:
                logger.info(f"Cargando DBC: {dbc_file}")
                if not processor.load_dbc(dbc_file):
                    logger.warning("No se pudo cargar el archivo DBC")
            elif dbc_list:
                logger.info(f"Cargando múltiples DBCs desde: {dbc_list}")
                with open(dbc_list, 'r') as f:
                    dbc_files = [line.strip() for line in f if line.strip()]
                processor.load_multiple_dbc(dbc_files)
            
            # Buscar y procesar archivos BLF
            blf_files = processor.find_blf_files(blf_dir)
            logger.info(f"Encontrados {len(blf_files)} archivos BLF")
            
            if not blf_files:
                raise ValueError("No se encontraron archivos BLF en el directorio")
            
            # Unificar archivos BLF
            unified_df = processor.unify_blf_files(blf_files)
            
            if unified_df.empty:
                raise ValueError("No se pudieron procesar los archivos BLF")
            
            # Decodificar mensajes si hay DBC cargado
            if processor.db:
                logger.info("Decodificando mensajes CAN...")
                self.data = processor.decode_messages(unified_df)
                self.metadata['data_type'] = 'decoded_can'
            else:
                logger.info("Usando datos CAN sin decodificar")
                self.data = unified_df
                self.metadata['data_type'] = 'raw_can'
            
            # Guardar metadata
            self.metadata.update({
                'source': 'blf',
                'blf_files_count': len(blf_files),
                'dbc_loaded': processor.db is not None,
                'processing_timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Datos cargados exitosamente: {len(self.data):,} registros")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando datos BLF: {str(e)}")
            return False
    
    def load_data_from_csv(self, csv_file):
        """
        Cargar datos desde archivo CSV.
        
        Args:
            csv_file: Ruta al archivo CSV
        """
        logger.info(f"Cargando datos CSV desde: {csv_file}")
        
        try:
            import pandas as pd
            self.data = pd.read_csv(csv_file)
            
            # Guardar metadata
            self.metadata.update({
                'source': 'csv',
                'csv_file': csv_file,
                'data_type': 'csv_import',
                'processing_timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Datos CSV cargados exitosamente: {len(self.data):,} registros")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando datos CSV: {str(e)}")
            return False
    
    def run_complete_analysis(self):
        """
        Ejecutar análisis exploratorio completo.
        
        Returns:
            dict: Diccionario con todos los resultados del análisis
        """
        if self.data is None:
            logger.error("No hay datos cargados para analizar")
            return None
        
        logger.info("=" * 60)
        logger.info("INICIANDO ANÁLISIS EXPLORATORIO COMPLETO")
        logger.info("=" * 60)
        
        analysis_results = {
            'metadata': self.metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. Análisis de calidad de datos
        logger.info("🔍 1. Análisis de calidad de datos...")
        try:
            quality_results = self.data_quality.analyze(self.data)
            analysis_results['data_quality'] = quality_results
            logger.info("✅ Análisis de calidad completado")
        except Exception as e:
            logger.error(f"❌ Error en análisis de calidad: {str(e)}")
            analysis_results['data_quality'] = {'error': str(e)}
        
        # 2. Análisis estadístico
        logger.info("📊 2. Análisis estadístico...")
        try:
            stats_results = self.statistical.analyze(self.data)
            analysis_results['statistics'] = stats_results
            logger.info("✅ Análisis estadístico completado")
        except Exception as e:
            logger.error(f"❌ Error en análisis estadístico: {str(e)}")
            analysis_results['statistics'] = {'error': str(e)}
        
        # 3. Detección de valores atípicos
        logger.info("🚨 3. Detección de valores atípicos...")
        try:
            outlier_results = self.outlier.detect_outliers(self.data)
            analysis_results['outliers'] = outlier_results
            logger.info("✅ Detección de outliers completada")
        except Exception as e:
            logger.error(f"❌ Error en detección de outliers: {str(e)}")
            analysis_results['outliers'] = {'error': str(e)}
        
        # 4. Análisis de correlaciones
        logger.info("🔗 4. Análisis de correlaciones...")
        try:
            correlation_results = self.correlation.analyze(self.data)
            analysis_results['correlations'] = correlation_results
            logger.info("✅ Análisis de correlaciones completado")
        except Exception as e:
            logger.error(f"❌ Error en análisis de correlaciones: {str(e)}")
            analysis_results['correlations'] = {'error': str(e)}
        
        # 5. Análisis temporal (si aplica)
        if self._has_temporal_data():
            logger.info("⏰ 5. Análisis temporal...")
            try:
                temporal_results = self.temporal.analyze(self.data)
                analysis_results['temporal'] = temporal_results
                logger.info("✅ Análisis temporal completado")
            except Exception as e:
                logger.error(f"❌ Error en análisis temporal: {str(e)}")
                analysis_results['temporal'] = {'error': str(e)}
        else:
            logger.info("⏰ 5. Análisis temporal omitido (no hay datos temporales)")
        
        # 6. Generar visualizaciones
        logger.info("📈 6. Generando visualizaciones...")
        try:
            viz_results = self.visualization.create_all_visualizations(
                self.data, analysis_results
            )
            analysis_results['visualizations'] = viz_results
            logger.info("✅ Visualizaciones generadas")
        except Exception as e:
            logger.error(f"❌ Error generando visualizaciones: {str(e)}")
            analysis_results['visualizations'] = {'error': str(e)}
        
        # 7. Generar reporte final
        logger.info("📋 7. Generando reporte final...")
        try:
            report_path = self._generate_final_report(analysis_results)
            analysis_results['report_path'] = report_path
            logger.info(f"✅ Reporte generado: {report_path}")
        except Exception as e:
            logger.error(f"❌ Error generando reporte: {str(e)}")
        
        logger.info("=" * 60)
        logger.info("ANÁLISIS EXPLORATORIO COMPLETADO")
        logger.info("=" * 60)
        
        return analysis_results
    
    def _has_temporal_data(self):
        """Verificar si los datos tienen componente temporal."""
        temporal_columns = ['timestamp', 'datetime', 'time', 'date']
        return any(col in self.data.columns for col in temporal_columns)
    
    def _generate_final_report(self, analysis_results):
        """
        Generar reporte final del análisis.
        
        Args:
            analysis_results: Resultados del análisis completo
            
        Returns:
            str: Ruta al archivo de reporte generado
        """
        report_path = os.path.join(
            self.config.reports_dir, 
            f"EDA_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        
        # Generar HTML del reporte
        html_content = self._create_report_html(analysis_results)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def _create_report_html(self, results):
        """Crear contenido HTML del reporte."""
        html = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reporte EDA - Sistema de Diagnóstico Vehicular</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                          border: 1px solid #ddd; border-radius: 5px; }}
                .error {{ color: red; font-style: italic; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚗 Análisis Exploratorio de Datos</h1>
                <h2>Sistema de Diagnóstico de Autonomía Vehicular</h2>
                <p>Generado: {results.get('timestamp', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h3>📊 Resumen de Datos</h3>
                <div class="metric">
                    <strong>Registros:</strong> {len(self.data):,}
                </div>
                <div class="metric">
                    <strong>Columnas:</strong> {len(self.data.columns)}
                </div>
                <div class="metric">
                    <strong>Tipo de Fuente:</strong> {results['metadata'].get('source', 'N/A')}
                </div>
            </div>
            
            {self._generate_quality_section_html(results.get('data_quality', {}))}
            {self._generate_stats_section_html(results.get('statistics', {}))}
            {self._generate_outliers_section_html(results.get('outliers', {}))}
            {self._generate_correlations_section_html(results.get('correlations', {}))}
            
            <div class="section">
                <h3>📁 Archivos Generados</h3>
                <p>Los gráficos y visualizaciones se encuentran en: <code>{self.config.visualizations_dir}</code></p>
            </div>
            
        </body>
        </html>
        """
        return html
    
    def _generate_quality_section_html(self, quality_results):
        """Generar sección HTML de calidad de datos."""
        if 'error' in quality_results:
            return f'<div class="section"><h3>🔍 Calidad de Datos</h3><p class="error">Error: {quality_results["error"]}</p></div>'
        
        return f"""
        <div class="section">
            <h3>🔍 Calidad de Datos</h3>
            <div class="metric">
                <strong>Valores Faltantes:</strong> {quality_results.get('missing_percentage', 'N/A')}%
            </div>
            <div class="metric">
                <strong>Duplicados:</strong> {quality_results.get('duplicates_count', 'N/A')}
            </div>
        </div>
        """
    
    def _generate_stats_section_html(self, stats_results):
        """Generar sección HTML de estadísticas."""
        if 'error' in stats_results:
            return f'<div class="section"><h3>📊 Estadísticas</h3><p class="error">Error: {stats_results["error"]}</p></div>'
        
        return f"""
        <div class="section">
            <h3>📊 Estadísticas Descriptivas</h3>
            <p>Estadísticas calculadas para {stats_results.get('numeric_columns_count', 'N/A')} columnas numéricas</p>
        </div>
        """
    
    def _generate_outliers_section_html(self, outlier_results):
        """Generar sección HTML de valores atípicos."""
        if 'error' in outlier_results:
            return f'<div class="section"><h3>🚨 Valores Atípicos</h3><p class="error">Error: {outlier_results["error"]}</p></div>'
        
        return f"""
        <div class="section">
            <h3>🚨 Valores Atípicos</h3>
            <div class="metric">
                <strong>Outliers Detectados:</strong> {outlier_results.get('total_outliers', 'N/A')}
            </div>
        </div>
        """
    
    def _generate_correlations_section_html(self, corr_results):
        """Generar sección HTML de correlaciones."""
        if 'error' in corr_results:
            return f'<div class="section"><h3>🔗 Correlaciones</h3><p class="error">Error: {corr_results["error"]}</p></div>'
        
        return f"""
        <div class="section">
            <h3>🔗 Análisis de Correlaciones</h3>
            <p>Correlaciones calculadas para variables numéricas</p>
        </div>
        """


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Análisis Exploratorio de Datos - Sistema de Diagnóstico Vehicular"
    )
    
    # Argumentos de fuente de datos
    parser.add_argument('--data-source', choices=['blf', 'csv'], required=True,
                       help='Tipo de fuente de datos')
    
    # Argumentos para datos BLF
    parser.add_argument('--blf-dir', help='Directorio con archivos BLF')
    parser.add_argument('--dbc-file', help='Archivo DBC para decodificación')
    parser.add_argument('--dbc-list', help='Archivo con lista de DBCs')
    
    # Argumentos para datos CSV
    parser.add_argument('--csv-file', help='Archivo CSV con datos')
    
    # Argumentos de configuración
    parser.add_argument('--output-dir', default='./EDA_output',
                       help='Directorio de salida para resultados')
    parser.add_argument('--config-file', help='Archivo de configuración personalizada')
    
    args = parser.parse_args()
    
    try:
        # Configurar analizador
        config = EDAConfig()
        if args.output_dir:
            config.output_dir = args.output_dir
        
        analyzer = EDAMainAnalyzer(config)
        
        # Cargar datos según la fuente
        if args.data_source == 'blf':
            if not args.blf_dir:
                parser.error("--blf-dir es requerido para fuente 'blf'")
            
            success = analyzer.load_data_from_blf(
                args.blf_dir, args.dbc_file, args.dbc_list
            )
            
        elif args.data_source == 'csv':
            if not args.csv_file:
                parser.error("--csv-file es requerido para fuente 'csv'")
            
            success = analyzer.load_data_from_csv(args.csv_file)
        
        if not success:
            logger.error("Error cargando datos. Terminando análisis.")
            return 1
        
        # Ejecutar análisis completo
        results = analyzer.run_complete_analysis()
        
        if results:
            print("\n" + "="*60)
            print("🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")
            print("="*60)
            print(f"📊 Registros analizados: {len(analyzer.data):,}")
            print(f"📁 Resultados en: {config.output_dir}")
            if 'report_path' in results:
                print(f"📋 Reporte principal: {results['report_path']}")
            print(f"📈 Visualizaciones: {config.visualizations_dir}")
            return 0
        else:
            logger.error("Error durante el análisis")
            return 1
            
    except KeyboardInterrupt:
        print("\n❌ Análisis cancelado por el usuario")
        return 1
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())