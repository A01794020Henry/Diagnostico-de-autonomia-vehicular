"""
Motor de Visualizaciones Simplificado
=====================================

Módulo para crear visualizaciones básicas usando matplotlib
para el análisis exploratorio de datos del sistema vehicular.

Autor: Sistema de diagnóstico de autonomía vehicular
Fecha: Septiembre 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """Motor de visualizaciones simplificado para análisis EDA."""
    
    def __init__(self, config):
        """Inicializar motor de visualizaciones."""
        self.config = config
        
    def create_all_visualizations(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crear todas las visualizaciones del análisis EDA.
        
        Args:
            data: DataFrame original
            analysis_results: Resultados del análisis EDA
            
        Returns:
            Diccionario con información de visualizaciones creadas
        """
        logger.info("Iniciando generación de visualizaciones")
        
        viz_results = {
            'static_plots_created': [],
            'summary_dashboard': None
        }
        
        try:
            # 1. Visualizaciones estáticas con Matplotlib
            static_plots = self._create_static_visualizations(data, analysis_results)
            viz_results['static_plots_created'] = static_plots
            
            # 2. Dashboard resumen
            dashboard_path = self._create_summary_dashboard(data, analysis_results)
            viz_results['summary_dashboard'] = dashboard_path
            
            logger.info(f"Visualizaciones creadas: {len(static_plots)} estáticas")
            
        except Exception as e:
            logger.error(f"Error creando visualizaciones: {str(e)}")
            viz_results['error'] = str(e)
        
        return viz_results
    
    def _create_static_visualizations(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
        """Crear visualizaciones estáticas."""
        static_plots = []
        
        try:
            # 1. Overview del dataset
            overview_path = self._create_dataset_overview(data)
            if overview_path:
                static_plots.append(overview_path)
            
            # 2. Análisis de distribuciones
            dist_path = self._create_distribution_analysis(data)
            if dist_path:
                static_plots.append(dist_path)
            
            # 3. Matriz de correlación
            corr_path = self._create_correlation_matrix(data)
            if corr_path:
                static_plots.append(corr_path)
            
            # 4. Análisis de valores faltantes
            missing_path = self._create_missing_values_analysis(data)
            if missing_path:
                static_plots.append(missing_path)
            
            # 5. Análisis de outliers
            outliers_path = self._create_outliers_visualization(data)
            if outliers_path:
                static_plots.append(outliers_path)
            
            # 6. Análisis temporal (si aplica)
            if self._has_temporal_data(data):
                temporal_path = self._create_temporal_analysis(data)
                if temporal_path:
                    static_plots.append(temporal_path)
            
        except Exception as e:
            logger.error(f"Error creando visualizaciones estáticas: {str(e)}")
        
        return static_plots
    
    def _create_dataset_overview(self, data: pd.DataFrame) -> Optional[str]:
        """Crear overview general del dataset."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Información básica del dataset
            basic_info = {
                'Registros': len(data),
                'Columnas': len(data.columns),
                'Memoria (MB)': data.memory_usage(deep=True).sum() / 1024**2,
                'Valores Faltantes': data.isnull().sum().sum()
            }
            
            ax1.axis('off')
            info_text = '\\n'.join([f'{k}: {v:,.2f}' if isinstance(v, float) else f'{k}: {v:,}' 
                                  for k, v in basic_info.items()])
            ax1.text(0.1, 0.5, f'INFORMACIÓN DEL DATASET\\n\\n{info_text}', 
                    fontsize=12, transform=ax1.transAxes, verticalalignment='center')
            
            # 2. Distribución de tipos de datos
            type_counts = data.dtypes.value_counts()
            colors = ['#2E8B57', '#1E90FF', '#FF6B35'][:len(type_counts)]
            ax2.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', colors=colors)
            ax2.set_title('Distribución de Tipos de Datos', fontweight='bold')
            
            # 3. Completitud por columna
            missing_by_col = (data.isnull().sum() / len(data) * 100).sort_values(ascending=False)
            top_missing = missing_by_col.head(10)
            
            if not top_missing.empty and top_missing.iloc[0] > 0:
                colors = ['#DC143C' if x > 50 else '#FFD700' if x > 20 else '#228B22' 
                         for x in top_missing.values]
                
                ax3.barh(range(len(top_missing)), top_missing.values, color=colors)
                ax3.set_yticks(range(len(top_missing)))
                ax3.set_yticklabels(top_missing.index, fontsize=8)
                ax3.set_xlabel('Porcentaje de Valores Faltantes')
                ax3.set_title('Variables con Más Valores Faltantes', fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'No hay valores faltantes\\nen el dataset', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Valores Faltantes', fontweight='bold')
            
            # 4. Distribución de cardinalidad
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                cardinality = [data[col].nunique() for col in data.columns]
                ax4.hist(cardinality, bins=20, color='#2E8B57', alpha=0.7)
                ax4.set_xlabel('Cardinalidad (Valores Únicos)')
                ax4.set_ylabel('Frecuencia')
                ax4.set_title('Distribución de Cardinalidad', fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'No hay columnas\\nnuméricas', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            
            plt.suptitle('Resumen General del Dataset', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_path = Path(self.config.visualizations_dir) / 'dataset_overview.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creando overview del dataset: {str(e)}")
            plt.close()
            return None
    
    def _create_distribution_analysis(self, data: pd.DataFrame) -> Optional[str]:
        """Crear análisis de distribuciones."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return None
            
            # Seleccionar hasta 9 columnas más relevantes
            cols_to_plot = numeric_data.columns[:9]
            
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, col in enumerate(cols_to_plot):
                ax = axes[i]
                series = numeric_data[col].dropna()
                
                if len(series) == 0:
                    ax.set_visible(False)
                    continue
                
                # Histograma
                ax.hist(series, bins=30, alpha=0.7, color='#2E8B57', density=True, label='Datos')
                
                # Línea de media y mediana
                mean_val = series.mean()
                median_val = series.median()
                
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Media: {mean_val:.2f}')
                ax.axvline(median_val, color='orange', linestyle='-', linewidth=2, 
                          label=f'Mediana: {median_val:.2f}')
                
                # Información estadística
                skewness = series.skew()
                kurtosis = series.kurtosis()
                
                ax.set_title(f'{col}\\nAsimetría: {skewness:.2f}, Curtosis: {kurtosis:.2f}', 
                           fontsize=10, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Ocultar ejes vacíos
            for i in range(len(cols_to_plot), 9):
                axes[i].set_visible(False)
            
            plt.suptitle('Análisis de Distribuciones', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_path = Path(self.config.visualizations_dir) / 'distributions' / 'distribution_analysis.png'
            output_path.parent.mkdir(exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creando análisis de distribuciones: {str(e)}")
            plt.close()
            return None
    
    def _create_correlation_matrix(self, data: pd.DataFrame) -> Optional[str]:
        """Crear matriz de correlación."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) < 2:
                return None
            
            # Calcular matriz de correlación
            correlation_matrix = numeric_data.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Crear heatmap manual
            im = ax.imshow(correlation_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Configurar ejes
            ax.set_xticks(range(len(correlation_matrix.columns)))
            ax.set_yticks(range(len(correlation_matrix.columns)))
            ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(correlation_matrix.columns)
            
            # Añadir valores de correlación
            for i in range(len(correlation_matrix.columns)):
                for j in range(len(correlation_matrix.columns)):
                    if i != j:  # No mostrar diagonal
                        text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlación', rotation=270, labelpad=20)
            
            ax.set_title('Matriz de Correlación', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            output_path = Path(self.config.visualizations_dir) / 'correlations' / 'correlation_matrix.png'
            output_path.parent.mkdir(exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creando matriz de correlación: {str(e)}")
            plt.close()
            return None
    
    def _create_missing_values_analysis(self, data: pd.DataFrame) -> Optional[str]:
        """Crear análisis de valores faltantes."""
        try:
            missing_data = data.isnull()
            
            if not missing_data.any().any():
                # No hay valores faltantes
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, '✅ No se encontraron valores faltantes en el dataset', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=16, fontweight='bold', color='green')
                ax.set_title('Análisis de Valores Faltantes', fontsize=16, fontweight='bold')
                ax.axis('off')
                
                output_path = Path(self.config.visualizations_dir) / 'quality' / 'missing_values_analysis.png'
                output_path.parent.mkdir(exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                return str(output_path)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Porcentaje de valores faltantes por columna
            missing_percent = (missing_data.mean() * 100).sort_values(ascending=False)
            missing_percent = missing_percent[missing_percent > 0]
            
            if not missing_percent.empty:
                colors = ['red' if x > 50 else 'orange' if x > 20 else 'gold' for x in missing_percent.values]
                
                bars = ax1.bar(range(len(missing_percent)), missing_percent.values, color=colors)
                ax1.set_xticks(range(len(missing_percent)))
                ax1.set_xticklabels(missing_percent.index, rotation=45, ha='right')
                ax1.set_ylabel('Porcentaje de Valores Faltantes')
                ax1.set_title('Valores Faltantes por Variable', fontweight='bold')
                
                # Agregar valores en las barras
                for bar, value in zip(bars, missing_percent.values):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
            
            # 2. Heatmap simple de valores faltantes
            missing_cols = data.columns[missing_data.any()]
            if len(missing_cols) > 0 and len(missing_cols) <= 50:
                sample_size = min(1000, len(data))
                sample_data = missing_data[missing_cols].iloc[:sample_size]
                
                im = ax2.imshow(sample_data.T.values, cmap='Reds', aspect='auto')
                ax2.set_yticks(range(len(missing_cols)))
                ax2.set_yticklabels(missing_cols, fontsize=8)
                ax2.set_title('Patrones de Valores Faltantes', fontweight='bold')
                ax2.set_xlabel('Registros (muestra)')
            else:
                ax2.text(0.5, 0.5, 'Demasiadas columnas\\npara mostrar heatmap', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Patrones de Valores Faltantes', fontweight='bold')
            
            # 3. Distribución de completitud
            completeness = (1 - missing_data.mean()) * 100
            ax3.hist(completeness, bins=20, color='#2E8B57', alpha=0.7)
            ax3.axvline(completeness.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Media: {completeness.mean():.1f}%')
            ax3.set_xlabel('Porcentaje de Completitud')
            ax3.set_ylabel('Número de Variables')
            ax3.set_title('Distribución de Completitud', fontweight='bold')
            ax3.legend()
            
            # 4. Estadísticas de valores faltantes
            total_cells = len(data) * len(data.columns)
            total_missing = missing_data.sum().sum()
            missing_stats = {
                'Total de celdas': f'{total_cells:,}',
                'Celdas faltantes': f'{total_missing:,}',
                '% total faltante': f'{(total_missing/total_cells)*100:.2f}%',
                'Variables afectadas': f'{missing_data.any().sum()}/{len(data.columns)}',
                'Registros completos': f'{len(data.dropna())}/{len(data)}'
            }
            
            ax4.axis('off')
            stats_text = '\\n'.join([f'{k}: {v}' for k, v in missing_stats.items()])
            ax4.text(0.1, 0.5, f'ESTADÍSTICAS DE VALORES FALTANTES\\n\\n{stats_text}', 
                    fontsize=11, transform=ax4.transAxes, verticalalignment='center')
            
            plt.suptitle('Análisis Completo de Valores Faltantes', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_path = Path(self.config.visualizations_dir) / 'quality' / 'missing_values_analysis.png'
            output_path.parent.mkdir(exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creando análisis de valores faltantes: {str(e)}")
            plt.close()
            return None
    
    def _create_outliers_visualization(self, data: pd.DataFrame) -> Optional[str]:
        """Crear visualización de valores atípicos."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return None
            
            # Seleccionar hasta 6 columnas para boxplots
            cols_to_plot = numeric_data.columns[:6]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(cols_to_plot):
                ax = axes[i]
                series = numeric_data[col].dropna()
                
                if len(series) == 0:
                    ax.set_visible(False)
                    continue
                
                # Boxplot
                box_plot = ax.boxplot(series, patch_artist=True)
                box_plot['boxes'][0].set_facecolor('#2E8B57')
                box_plot['boxes'][0].set_alpha(0.7)
                
                # Calcular estadísticas de outliers
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(series)) * 100
                
                ax.set_title(f'{col}\\nOutliers: {outlier_count} ({outlier_percentage:.1f}%)', 
                           fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
            
            # Ocultar ejes vacíos
            for i in range(len(cols_to_plot), 6):
                axes[i].set_visible(False)
            
            plt.suptitle('Análisis de Valores Atípicos (Boxplots)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_path = Path(self.config.visualizations_dir) / 'outliers' / 'outliers_boxplots.png'
            output_path.parent.mkdir(exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creando visualización de outliers: {str(e)}")
            plt.close()
            return None
    
    def _create_temporal_analysis(self, data: pd.DataFrame) -> Optional[str]:
        """Crear análisis temporal básico."""
        try:
            # Buscar columnas temporales
            temporal_cols = [col for col in data.columns 
                           if any(temporal_keyword in col.lower() 
                                 for temporal_keyword in ['time', 'date', 'timestamp'])]
            
            if not temporal_cols:
                return None
            
            # Tomar la primera columna temporal
            time_col = temporal_cols[0]
            
            # Convertir a datetime si es necesario
            if pd.api.types.is_numeric_dtype(data[time_col]):
                try:
                    time_series = pd.to_datetime(data[time_col], unit='s', errors='coerce')
                except:
                    time_series = pd.to_datetime(data[time_col], errors='coerce')
            else:
                time_series = pd.to_datetime(data[time_col], errors='coerce')
            
            # Filtrar valores válidos
            valid_times = time_series.dropna()
            
            if len(valid_times) < 10:
                return None
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Distribución temporal general
            ax1.hist(valid_times, bins=50, color='#2E8B57', alpha=0.7)
            ax1.set_title('Distribución Temporal de los Datos', fontweight='bold')
            ax1.set_xlabel('Tiempo')
            ax1.set_ylabel('Frecuencia')
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Patrón de actividad por hora
            if len(valid_times) > 24:
                hourly_counts = valid_times.dt.hour.value_counts().sort_index()
                ax2.bar(hourly_counts.index, hourly_counts.values, 
                       color='#1E90FF', alpha=0.7)
                ax2.set_title('Patrón de Actividad por Hora', fontweight='bold')
                ax2.set_xlabel('Hora del Día')
                ax2.set_ylabel('Número de Registros')
                ax2.set_xticks(range(0, 24, 4))
            else:
                ax2.text(0.5, 0.5, 'Insuficientes datos\\npara análisis horario', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Patrón Horario', fontweight='bold')
            
            # 3. Patrón por día de la semana
            if len(valid_times) > 7:
                daily_counts = valid_times.dt.day_name().value_counts()
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_counts = daily_counts.reindex([day for day in days_order if day in daily_counts.index])
                
                ax3.bar(range(len(daily_counts)), daily_counts.values, 
                       color='#FF6B35', alpha=0.7)
                ax3.set_xticks(range(len(daily_counts)))
                ax3.set_xticklabels([day[:3] for day in daily_counts.index], rotation=45)
                ax3.set_title('Patrón por Día de la Semana', fontweight='bold')
                ax3.set_xlabel('Día de la Semana')
                ax3.set_ylabel('Número de Registros')
            else:
                ax3.text(0.5, 0.5, 'Insuficientes datos\\npara análisis semanal', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Patrón Semanal', fontweight='bold')
            
            # 4. Estadísticas temporales
            time_stats = {
                'Período': f'{valid_times.min().strftime("%Y-%m-%d")} a {valid_times.max().strftime("%Y-%m-%d")}',
                'Duración': f'{(valid_times.max() - valid_times.min()).days} días',
                'Registros': f'{len(valid_times):,}',
                'Duplicados': f'{len(valid_times) - valid_times.nunique()}'
            }
            
            ax4.axis('off')
            stats_text = '\\n'.join([f'{k}: {v}' for k, v in time_stats.items()])
            ax4.text(0.1, 0.5, f'ESTADÍSTICAS TEMPORALES\\n\\n{stats_text}', 
                    fontsize=11, transform=ax4.transAxes, verticalalignment='center')
            
            plt.suptitle('Análisis Temporal del Dataset', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_path = Path(self.config.visualizations_dir) / 'temporal' / 'temporal_analysis.png'
            output_path.parent.mkdir(exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creando análisis temporal: {str(e)}")
            plt.close()
            return None
    
    def _create_summary_dashboard(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> Optional[str]:
        """Crear dashboard HTML resumen."""
        try:
            html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard EDA - Sistema Diagnóstico Vehicular</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2E8B57, #1E90FF);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2E8B57;
            margin-bottom: 10px;
        }}
        .metric-label {{
            color: #6c757d;
            font-size: 1em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .section {{
            margin: 30px;
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
        }}
        .section h3 {{
            color: #2E8B57;
            border-bottom: 2px solid #2E8B57;
            padding-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🚗 Dashboard EDA - Sistema de Diagnóstico Vehicular</h1>
            <p>Análisis Exploratorio Completo | {analysis_results.get('timestamp', 'N/A')}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{len(data):,}</div>
                <div class="metric-label">Registros</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(data.columns)}</div>
                <div class="metric-label">Variables</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data.memory_usage(deep=True).sum() / 1024**2:.1f}</div>
                <div class="metric-label">MB en Memoria</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{((len(data) - data.isnull().sum().sum()) / (len(data) * len(data.columns)) * 100):.1f}%</div>
                <div class="metric-label">Completitud</div>
            </div>
        </div>
        
        <div class="section">
            <h3>📁 Archivos Generados</h3>
            <p>📈 <strong>Visualizaciones:</strong> {self.config.visualizations_dir}</p>
            <p>📋 <strong>Reportes:</strong> {self.config.reports_dir}</p>
            <ul>
                <li>📊 Análisis de distribuciones</li>
                <li>🔗 Matrices de correlación</li>
                <li>🚨 Detección de valores atípicos</li>
                <li>⏰ Análisis temporal (si aplica)</li>
                <li>🔍 Análisis de calidad de datos</li>
            </ul>
        </div>
    </div>
</body>
</html>"""
            
            output_path = Path(self.config.visualizations_dir) / 'eda_dashboard.html'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creando dashboard resumen: {str(e)}")
            return None
    
    def _has_temporal_data(self, data: pd.DataFrame) -> bool:
        """Verificar si hay datos temporales."""
        temporal_keywords = ['time', 'date', 'timestamp']
        return any(any(keyword in col.lower() for keyword in temporal_keywords) 
                  for col in data.columns)