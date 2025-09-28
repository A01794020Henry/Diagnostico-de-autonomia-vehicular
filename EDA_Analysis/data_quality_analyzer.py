"""
Analizador de Calidad de Datos
=============================

Módulo especializado para evaluar la calidad de los datos del sistema
vehicular, identificando problemas comunes y patrones en los datos.

Autor: Sistema de diagnóstico de autonomía vehicular
Fecha: Septiembre 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataQualityAnalyzer:
    """Analizador de calidad de datos para el sistema vehicular."""
    
    def __init__(self, config):
        """Inicializar analizador de calidad."""
        self.config = config
        self.quality_report = {}
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Realizar análisis completo de calidad de datos.
        
        Args:
            data: DataFrame a analizar
            
        Returns:
            Diccionario con resultados del análisis de calidad
        """
        logger.info("Iniciando análisis de calidad de datos")
        
        quality_results = {
            'basic_info': self._get_basic_info(data),
            'missing_values': self._analyze_missing_values(data),
            'duplicates': self._analyze_duplicates(data),
            'data_types': self._analyze_data_types(data),
            'cardinality': self._analyze_cardinality(data),
            'consistency': self._analyze_consistency(data),
            'completeness_score': self._calculate_completeness_score(data)
        }
        
        # Generar visualizaciones de calidad
        self._create_quality_visualizations(data, quality_results)
        
        # Generar recomendaciones
        quality_results['recommendations'] = self._generate_recommendations(quality_results)
        
        logger.info("Análisis de calidad completado")
        return quality_results
    
    def _get_basic_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Obtener información básica del dataset."""
        return {
            'rows': len(data),
            'columns': len(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2,
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(data.select_dtypes(include=['datetime64']).columns)
        }
    
    def _analyze_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analizar valores faltantes en el dataset."""
        missing_info = {
            'total_missing': data.isnull().sum().sum(),
            'missing_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'columns_with_missing': {},
            'missing_patterns': {}
        }
        
        # Análisis por columna
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                missing_info['columns_with_missing'][col] = {
                    'count': missing_count,
                    'percentage': (missing_count / len(data)) * 100
                }
        
        # Patrones de valores faltantes
        if missing_info['columns_with_missing']:
            # Crear matriz de valores faltantes
            missing_matrix = data.isnull()
            
            # Encontrar patrones comunes
            pattern_counts = missing_matrix.value_counts()
            missing_info['missing_patterns'] = {
                'unique_patterns': len(pattern_counts),
                'most_common_patterns': pattern_counts.head(5).to_dict()
            }
        
        return missing_info
    
    def _analyze_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analizar registros duplicados."""
        duplicates_info = {
            'total_duplicates': data.duplicated().sum(),
            'duplicate_percentage': (data.duplicated().sum() / len(data)) * 100,
            'unique_rows': len(data.drop_duplicates()),
            'duplicate_subsets': {}
        }
        
        # Analizar duplicados en subconjuntos de columnas importantes
        important_columns = self._identify_important_columns(data)
        
        for subset in important_columns:
            if len(subset) > 1:
                subset_name = '_'.join(subset[:2])  # Usar primeras 2 columnas como nombre
                duplicates_info['duplicate_subsets'][subset_name] = {
                    'columns': subset,
                    'duplicates': data.duplicated(subset=subset).sum()
                }
        
        return duplicates_info
    
    def _analyze_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analizar tipos de datos y posibles inconsistencias."""
        type_info = {
            'type_distribution': data.dtypes.value_counts().to_dict(),
            'potential_type_issues': {},
            'memory_optimization': {}
        }
        
        for col in data.columns:
            col_type = str(data[col].dtype)
            
            # Verificar posibles problemas de tipo
            if col_type == 'object':
                # Verificar si podría ser numérico
                try:
                    pd.to_numeric(data[col], errors='coerce')
                    non_numeric_count = pd.to_numeric(data[col], errors='coerce').isnull().sum()
                    if non_numeric_count < len(data) * 0.1:  # Menos del 10% no numérico
                        type_info['potential_type_issues'][col] = 'could_be_numeric'
                except:
                    pass
                
                # Verificar si podría ser categórico
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio < 0.05:  # Menos del 5% de valores únicos
                    type_info['potential_type_issues'][col] = 'could_be_categorical'
            
            # Sugerencias de optimización de memoria
            if col_type in ['int64', 'float64']:
                current_memory = data[col].memory_usage(deep=True)
                
                if col_type == 'int64':
                    # Verificar si puede usar int32 o int16
                    min_val, max_val = data[col].min(), data[col].max()
                    if min_val >= -2147483648 and max_val <= 2147483647:
                        type_info['memory_optimization'][col] = 'int32'
                    elif min_val >= -32768 and max_val <= 32767:
                        type_info['memory_optimization'][col] = 'int16'
                
                elif col_type == 'float64':
                    # Verificar si puede usar float32
                    type_info['memory_optimization'][col] = 'float32'
        
        return type_info
    
    def _analyze_cardinality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analizar cardinalidad de variables categóricas."""
        cardinality_info = {
            'high_cardinality_columns': {},
            'low_cardinality_columns': {},
            'cardinality_distribution': {}
        }
        
        for col in data.columns:
            unique_count = data[col].nunique()
            unique_ratio = unique_count / len(data)
            
            cardinality_info['cardinality_distribution'][col] = {
                'unique_count': unique_count,
                'unique_ratio': unique_ratio
            }
            
            # Clasificar por cardinalidad
            if unique_count > self.config.high_cardinality_threshold:
                cardinality_info['high_cardinality_columns'][col] = {
                    'unique_count': unique_count,
                    'unique_ratio': unique_ratio,
                    'top_values': data[col].value_counts().head(5).to_dict()
                }
            elif unique_ratio < 0.1:  # Menos del 10% de valores únicos
                cardinality_info['low_cardinality_columns'][col] = {
                    'unique_count': unique_count,
                    'unique_ratio': unique_ratio,
                    'value_distribution': data[col].value_counts().to_dict()
                }
        
        return cardinality_info
    
    def _analyze_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analizar consistencia de datos específicos del dominio vehicular."""
        consistency_info = {
            'timestamp_consistency': {},
            'signal_consistency': {},
            'range_violations': {}
        }
        
        # Verificar consistencia temporal
        timestamp_cols = [col for col in data.columns if 'timestamp' in col.lower() or 'time' in col.lower()]
        for col in timestamp_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                consistency_info['timestamp_consistency'][col] = {
                    'monotonic_increasing': data[col].is_monotonic_increasing,
                    'gaps_detected': self._detect_time_gaps(data[col]),
                    'duplicate_timestamps': data[col].duplicated().sum()
                }
        
        # Verificar rangos de señales vehiculares conocidas
        signal_ranges = {
            'voltage': (0, 1000),  # Voltios
            'current': (-500, 500),  # Amperios
            'temperature': (-50, 150),  # Celsius
            'speed': (0, 300),  # km/h
            'soc': (0, 100)  # Porcentaje
        }
        
        for col in data.columns:
            col_lower = col.lower()
            for signal_type, (min_val, max_val) in signal_ranges.items():
                if signal_type in col_lower and pd.api.types.is_numeric_dtype(data[col]):
                    violations = ((data[col] < min_val) | (data[col] > max_val)).sum()
                    if violations > 0:
                        consistency_info['range_violations'][col] = {
                            'expected_range': (min_val, max_val),
                            'violations': violations,
                            'violation_percentage': (violations / len(data)) * 100
                        }
        
        return consistency_info
    
    def _detect_time_gaps(self, time_series: pd.Series) -> Dict[str, Any]:
        """Detectar gaps en series temporales."""
        if len(time_series) < 2:
            return {'gaps_found': False}
        
        # Calcular diferencias entre timestamps consecutivos
        diffs = time_series.diff().dropna()
        
        # Detectar gaps (diferencias anormalmente grandes)
        median_diff = diffs.median()
        threshold = median_diff * 5  # 5 veces la diferencia mediana
        
        gaps = diffs[diffs > threshold]
        
        return {
            'gaps_found': len(gaps) > 0,
            'gap_count': len(gaps),
            'largest_gap': gaps.max() if len(gaps) > 0 else None,
            'median_interval': median_diff
        }
    
    def _calculate_completeness_score(self, data: pd.DataFrame) -> float:
        """Calcular puntuación de completitud de los datos."""
        total_cells = len(data) * len(data.columns)
        non_null_cells = total_cells - data.isnull().sum().sum()
        return (non_null_cells / total_cells) * 100 if total_cells > 0 else 0
    
    def _identify_important_columns(self, data: pd.DataFrame) -> List[List[str]]:
        """Identificar combinaciones importantes de columnas para análisis."""
        important_combinations = []
        
        # Combinaciones específicas del dominio vehicular
        vehicular_combinations = [
            ['timestamp', 'arbitration_id'],
            ['message_name', 'signal_name'],
            ['latitude', 'longitude'],
            ['voltage', 'current']
        ]
        
        for combo in vehicular_combinations:
            available_cols = [col for col in combo if col in data.columns]
            if len(available_cols) > 1:
                important_combinations.append(available_cols)
        
        return important_combinations
    
    def _create_quality_visualizations(self, data: pd.DataFrame, quality_results: Dict[str, Any]):
        """Crear visualizaciones de calidad de datos."""
        quality_viz_dir = Path(self.config.visualizations_dir) / 'quality'
        quality_viz_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Mapa de valores faltantes
            if quality_results['missing_values']['columns_with_missing']:
                self._plot_missing_values_heatmap(data, quality_viz_dir)
            
            # 2. Distribución de tipos de datos
            self._plot_data_types_distribution(quality_results['data_types'], quality_viz_dir)
            
            # 3. Cardinalidad de variables
            self._plot_cardinality_analysis(quality_results['cardinality'], quality_viz_dir)
            
            # 4. Completitud por columna
            self._plot_completeness_by_column(data, quality_viz_dir)
            
        except Exception as e:
            logger.error(f"Error creando visualizaciones de calidad: {str(e)}")
    
    def _plot_missing_values_heatmap(self, data: pd.DataFrame, output_dir: Path):
        """Crear mapa de calor de valores faltantes."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Seleccionar solo columnas con valores faltantes
        missing_cols = data.columns[data.isnull().any()]
        if len(missing_cols) > 0:
            missing_data = data[missing_cols].isnull()
            
            sns.heatmap(
                missing_data.T, 
                cbar=True, 
                ax=ax, 
                cmap='Reds',
                yticklabels=True,
                xticklabels=False
            )
            
            ax.set_title('Mapa de Valores Faltantes', fontsize=16, fontweight='bold')
            ax.set_xlabel('Registros')
            ax.set_ylabel('Variables')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'missing_values_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_data_types_distribution(self, type_info: Dict[str, Any], output_dir: Path):
        """Crear gráfico de distribución de tipos de datos."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        type_dist = type_info['type_distribution']
        types = list(type_dist.keys())
        counts = list(type_dist.values())
        
        colors = [self.config.get_category_color('other') for _ in types]
        
        bars = ax.bar(types, counts, color=colors)
        
        # Añadir etiquetas de valor
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   str(count), ha='center', va='bottom')
        
        ax.set_title('Distribución de Tipos de Datos', fontsize=16, fontweight='bold')
        ax.set_xlabel('Tipo de Dato')
        ax.set_ylabel('Número de Columnas')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'data_types_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cardinality_analysis(self, cardinality_info: Dict[str, Any], output_dir: Path):
        """Crear análisis de cardinalidad."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico 1: Distribución de cardinalidad
        cardinality_data = cardinality_info['cardinality_distribution']
        columns = list(cardinality_data.keys())
        unique_counts = [info['unique_count'] for info in cardinality_data.values()]
        
        ax1.hist(unique_counts, bins=20, color=self.config.color_palette['primary'], alpha=0.7)
        ax1.set_title('Distribución de Cardinalidad', fontweight='bold')
        ax1.set_xlabel('Número de Valores Únicos')
        ax1.set_ylabel('Frecuencia')
        
        # Gráfico 2: Top columnas por cardinalidad
        top_cardinality = sorted(
            [(col, info['unique_count']) for col, info in cardinality_data.items()],
            key=lambda x: x[1], reverse=True
        )[:10]
        
        if top_cardinality:
            cols, counts = zip(*top_cardinality)
            ax2.barh(range(len(cols)), counts, color=self.config.color_palette['secondary'])
            ax2.set_yticks(range(len(cols)))
            ax2.set_yticklabels(cols)
            ax2.set_title('Top 10 Columnas por Cardinalidad', fontweight='bold')
            ax2.set_xlabel('Valores Únicos')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cardinality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_completeness_by_column(self, data: pd.DataFrame, output_dir: Path):
        """Crear gráfico de completitud por columna."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calcular completitud por columna
        completeness = ((len(data) - data.isnull().sum()) / len(data) * 100).sort_values()
        
        colors = [self.config.color_palette['success'] if x >= 95 else 
                 self.config.color_palette['warning'] if x >= 80 else 
                 self.config.color_palette['error'] for x in completeness.values]
        
        bars = ax.barh(range(len(completeness)), completeness.values, color=colors)
        
        ax.set_yticks(range(len(completeness)))
        ax.set_yticklabels(completeness.index, fontsize=8)
        ax.set_xlabel('Porcentaje de Completitud (%)')
        ax.set_title('Completitud por Variable', fontsize=16, fontweight='bold')
        
        # Añadir líneas de referencia
        ax.axvline(x=95, color='green', linestyle='--', alpha=0.5, label='95% (Excelente)')
        ax.axvline(x=80, color='orange', linestyle='--', alpha=0.5, label='80% (Aceptable)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'completeness_by_column.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_recommendations(self, quality_results: Dict[str, Any]) -> List[str]:
        """Generar recomendaciones basadas en el análisis de calidad."""
        recommendations = []
        
        # Recomendaciones para valores faltantes
        missing_info = quality_results['missing_values']
        if missing_info['missing_percentage'] > 10:
            recommendations.append(
                f"⚠️ Alto porcentaje de valores faltantes ({missing_info['missing_percentage']:.1f}%). "
                "Considera estrategias de imputación o eliminación de variables/registros."
            )
        
        # Recomendaciones para duplicados
        duplicates_info = quality_results['duplicates']
        if duplicates_info['duplicate_percentage'] > 5:
            recommendations.append(
                f"🔄 Se detectaron duplicados ({duplicates_info['duplicate_percentage']:.1f}% del dataset). "
                "Revisa y elimina registros duplicados innecesarios."
            )
        
        # Recomendaciones para tipos de datos
        type_info = quality_results['data_types']
        if type_info['potential_type_issues']:
            recommendations.append(
                "📊 Se detectaron posibles mejoras en tipos de datos. "
                "Considera convertir variables a tipos más apropiados para optimizar memoria y rendimiento."
            )
        
        # Recomendaciones para cardinalidad
        cardinality_info = quality_results['cardinality']
        if cardinality_info['high_cardinality_columns']:
            recommendations.append(
                "🔢 Variables con alta cardinalidad detectadas. "
                "Considera técnicas de encoding o agrupación para variables categóricas."
            )
        
        # Recomendaciones para consistencia
        consistency_info = quality_results['consistency']
        if consistency_info['range_violations']:
            recommendations.append(
                "⚡ Se detectaron valores fuera de rangos esperados. "
                "Revisa y valida estos valores atípicos en señales vehiculares."
            )
        
        # Recomendación general de completitud
        completeness = quality_results['completeness_score']
        if completeness < 80:
            recommendations.append(
                f"📈 Completitud general baja ({completeness:.1f}%). "
                "Prioriza la mejora de calidad de datos antes del modelado."
            )
        
        if not recommendations:
            recommendations.append("✅ La calidad general de los datos es buena. Procede con el análisis.")
        
        return recommendations
