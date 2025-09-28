"""
Analizador Temporal
==================

Módulo especializado para analizar patrones temporales en los datos
del sistema de diagnóstico vehicular.

Autor: Sistema de diagnóstico de autonomía vehicular
Fecha: Septiembre 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
from scipy import signal
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """Analizador temporal para datos vehiculares."""
    
    def __init__(self, config):
        """Inicializar analizador temporal."""
        self.config = config
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Realizar análisis temporal completo.
        
        Args:
            data: DataFrame con datos temporales
            
        Returns:
            Diccionario con resultados del análisis temporal
        """
        logger.info("Iniciando análisis temporal")
        
        # Buscar columnas temporales
        time_columns = self._identify_time_columns(data)
        
        if not time_columns:
            return {'error': 'No se encontraron columnas temporales en el dataset'}
        
        results = {
            'time_columns_found': time_columns,
            'temporal_patterns': {},
            'seasonality_analysis': {},
            'trend_analysis': {},
            'gap_analysis': {},
            'frequency_analysis': {}
        }
        
        # Analizar cada columna temporal
        for time_col in time_columns[:2]:  # Limitar a 2 columnas temporales
            try:
                col_results = self._analyze_time_column(data, time_col)
                results['temporal_patterns'][time_col] = col_results
            except Exception as e:
                logger.error(f"Error analizando columna temporal {time_col}: {str(e)}")
                continue
        
        # Crear visualizaciones temporales
        self._create_temporal_visualizations(data, results)
        
        logger.info("Análisis temporal completado")
        return results
    
    def _identify_time_columns(self, data: pd.DataFrame) -> List[str]:
        """Identificar columnas temporales."""
        time_keywords = ['time', 'date', 'timestamp', 'datetime']
        time_columns = []
        
        for col in data.columns:
            col_lower = col.lower()
            
            # Verificar por nombre
            if any(keyword in col_lower for keyword in time_keywords):
                time_columns.append(col)
                continue
            
            # Verificar por tipo de datos
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                time_columns.append(col)
                continue
            
            # Verificar si es numérico que podría ser timestamp
            if pd.api.types.is_numeric_dtype(data[col]):
                try:
                    # Intentar convertir una muestra a datetime
                    sample = data[col].dropna().iloc[:10]
                    if len(sample) > 0:
                        # Probar diferentes formatos de timestamp
                        pd.to_datetime(sample, unit='s')  # Unix timestamp
                        time_columns.append(col)
                        continue
                except:
                    pass
        
        return time_columns
    
    def _analyze_time_column(self, data: pd.DataFrame, time_col: str) -> Dict[str, Any]:
        """Analizar una columna temporal específica."""
        # Preparar serie temporal
        time_series = self._prepare_time_series(data[time_col])
        
        if time_series is None:
            return {'error': f'No se pudo procesar la columna temporal {time_col}'}
        
        valid_times = time_series.dropna()
        
        if len(valid_times) < 10:
            return {'error': f'Insuficientes datos temporales válidos en {time_col}'}
        
        analysis = {
            'basic_info': self._temporal_basic_info(valid_times),
            'patterns': self._detect_temporal_patterns(valid_times),
            'gaps': self._analyze_time_gaps(valid_times),
            'frequency': self._analyze_frequency(valid_times)
        }
        
        # Análisis de series numéricas relacionadas
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['series_analysis'] = self._analyze_time_series(data, time_col, numeric_cols[:5])
        
        return analysis
    
    def _prepare_time_series(self, time_data: pd.Series) -> Optional[pd.Series]:
        """Preparar serie temporal para análisis."""
        try:
            if pd.api.types.is_datetime64_any_dtype(time_data):
                return time_data
            
            # Si es numérico, intentar convertir
            if pd.api.types.is_numeric_dtype(time_data):
                # Probar diferentes unidades
                try:
                    return pd.to_datetime(time_data, unit='s')
                except:
                    try:
                        return pd.to_datetime(time_data, unit='ms')
                    except:
                        return pd.to_datetime(time_data, unit='us')
            
            # Si es string, intentar parsear
            return pd.to_datetime(time_data, errors='coerce')
            
        except Exception as e:
            logger.error(f"Error preparando serie temporal: {str(e)}")
            return None
    
    def _temporal_basic_info(self, time_series: pd.Series) -> Dict[str, Any]:
        """Información básica de la serie temporal."""
        return {
            'start_time': time_series.min().isoformat(),
            'end_time': time_series.max().isoformat(),
            'duration_days': (time_series.max() - time_series.min()).days,
            'total_points': len(time_series),
            'unique_timestamps': time_series.nunique(),
            'duplicate_timestamps': len(time_series) - time_series.nunique(),
            'is_monotonic': time_series.is_monotonic_increasing
        }
    
    def _detect_temporal_patterns(self, time_series: pd.Series) -> Dict[str, Any]:
        """Detectar patrones temporales."""
        patterns = {
            'hourly_pattern': {},
            'daily_pattern': {},
            'weekly_pattern': {},
            'monthly_pattern': {}
        }
        
        # Patrón por hora del día
        if len(time_series) >= 24:
            hourly_counts = time_series.dt.hour.value_counts().sort_index()
            patterns['hourly_pattern'] = {
                'peak_hour': int(hourly_counts.idxmax()),
                'min_hour': int(hourly_counts.idxmin()),
                'hourly_distribution': hourly_counts.to_dict(),
                'coefficient_variation': hourly_counts.std() / hourly_counts.mean()
            }
        
        # Patrón por día de la semana
        if len(time_series) >= 7:
            daily_counts = time_series.dt.day_name().value_counts()
            patterns['daily_pattern'] = {
                'peak_day': daily_counts.idxmax(),
                'min_day': daily_counts.idxmin(),
                'daily_distribution': daily_counts.to_dict(),
                'weekday_vs_weekend': {
                    'weekday_avg': daily_counts[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].mean(),
                    'weekend_avg': daily_counts[['Saturday', 'Sunday']].mean()
                }
            }
        
        # Patrón mensual
        if (time_series.max() - time_series.min()).days >= 30:
            monthly_counts = time_series.dt.month.value_counts().sort_index()
            patterns['monthly_pattern'] = {
                'peak_month': int(monthly_counts.idxmax()),
                'min_month': int(monthly_counts.idxmin()),
                'monthly_distribution': monthly_counts.to_dict()
            }
        
        return patterns
    
    def _analyze_time_gaps(self, time_series: pd.Series) -> Dict[str, Any]:
        """Analizar gaps temporales."""
        if len(time_series) < 2:
            return {'gaps_detected': False}
        
        # Calcular diferencias entre timestamps consecutivos
        sorted_times = time_series.sort_values()
        time_diffs = sorted_times.diff().dropna()
        
        # Estadísticas de intervalos
        median_interval = time_diffs.median()
        mean_interval = time_diffs.mean()
        
        # Detectar gaps (intervalos anormalmente largos)
        gap_threshold = median_interval * 5  # 5 veces el intervalo mediano
        gaps = time_diffs[time_diffs > gap_threshold]
        
        gap_info = {
            'total_intervals': len(time_diffs),
            'median_interval_seconds': median_interval.total_seconds(),
            'mean_interval_seconds': mean_interval.total_seconds(),
            'gaps_detected': len(gaps) > 0,
            'number_of_gaps': len(gaps),
            'gap_threshold_seconds': gap_threshold.total_seconds()
        }
        
        if len(gaps) > 0:
            gap_info.update({
                'largest_gap_seconds': gaps.max().total_seconds(),
                'total_gap_time_seconds': gaps.sum().total_seconds(),
                'gap_locations': [{
                    'start': (sorted_times.iloc[i-1]).isoformat(),
                    'end': (sorted_times.iloc[i]).isoformat(),
                    'duration_seconds': gaps.iloc[j].total_seconds()
                } for j, i in enumerate(gaps.index[:10])]  # Top 10 gaps
            })
        
        return gap_info
    
    def _analyze_frequency(self, time_series: pd.Series) -> Dict[str, Any]:
        """Analizar frecuencia de muestreo."""
        if len(time_series) < 10:
            return {'error': 'Insuficientes datos para análisis de frecuencia'}
        
        sorted_times = time_series.sort_values()
        intervals = sorted_times.diff().dropna()
        
        # Convertir a segundos
        intervals_seconds = intervals.dt.total_seconds()
        
        # Encontrar frecuencia más común
        from collections import Counter
        
        # Redondear intervalos para agrupar similares
        rounded_intervals = np.round(intervals_seconds, 1)
        interval_counts = Counter(rounded_intervals)
        most_common_interval = interval_counts.most_common(1)[0][0]
        
        frequency_info = {
            'most_common_interval_seconds': most_common_interval,
            'estimated_frequency_hz': 1.0 / most_common_interval if most_common_interval > 0 else 0,
            'interval_statistics': {
                'min_seconds': float(intervals_seconds.min()),
                'max_seconds': float(intervals_seconds.max()),
                'mean_seconds': float(intervals_seconds.mean()),
                'median_seconds': float(intervals_seconds.median()),
                'std_seconds': float(intervals_seconds.std())
            },
            'regularity_score': interval_counts[most_common_interval] / len(intervals) * 100,
            'unique_intervals': len(set(rounded_intervals))
        }
        
        return frequency_info
    
    def _analyze_time_series(self, data: pd.DataFrame, time_col: str, numeric_cols: List[str]) -> Dict[str, Any]:
        """Analizar series temporales numéricas."""
        time_series = self._prepare_time_series(data[time_col])
        
        if time_series is None:
            return {'error': 'No se pudo preparar serie temporal'}
        
        series_analysis = {}
        
        for col in numeric_cols:
            try:
                # Crear DataFrame con tiempo y valores
                valid_mask = time_series.notna() & data[col].notna()
                
                if valid_mask.sum() < 10:
                    continue
                
                ts_data = pd.DataFrame({
                    'time': time_series[valid_mask],
                    'value': data[col][valid_mask]
                }).sort_values('time')
                
                # Análisis básico de la serie
                analysis = {
                    'data_points': len(ts_data),
                    'time_span_hours': (ts_data['time'].max() - ts_data['time'].min()).total_seconds() / 3600,
                    'value_range': (float(ts_data['value'].min()), float(ts_data['value'].max())),
                    'trend_analysis': self._analyze_trend(ts_data)
                }
                
                # Detectar anomalías temporales simples
                analysis['anomalies'] = self._detect_temporal_anomalies(ts_data)
                
                series_analysis[col] = analysis
                
            except Exception as e:
                logger.error(f"Error analizando serie temporal {col}: {str(e)}")
                continue
        
        return series_analysis
    
    def _analyze_trend(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """Analizar tendencia en serie temporal."""
        if len(ts_data) < 3:
            return {'trend': 'insufficient_data'}
        
        # Convertir tiempo a numérico para regresión
        time_numeric = (ts_data['time'] - ts_data['time'].min()).dt.total_seconds()
        values = ts_data['value']
        
        # Regresión lineal simple
        try:
            correlation, p_value = pearsonr(time_numeric, values)
            
            # Calcular pendiente
            slope = np.polyfit(time_numeric, values, 1)[0]
            
            trend_strength = 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.3 else 'weak'
            trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
            
            return {
                'correlation': float(correlation),
                'p_value': float(p_value),
                'slope': float(slope),
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'significant': p_value < 0.05
            }
            
        except Exception as e:
            return {'trend': 'calculation_error', 'error': str(e)}
    
    def _detect_temporal_anomalies(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """Detectar anomalías temporales simples."""
        values = ts_data['value']
        
        # Detectar valores extremos usando IQR
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ts_data[(values < lower_bound) | (values > upper_bound)]
        
        # Detectar cambios súbitos
        value_diffs = values.diff().abs()
        median_diff = value_diffs.median()
        sudden_changes = ts_data[value_diffs > median_diff * 5]  # 5x el cambio mediano
        
        return {
            'outliers_count': len(outliers),
            'outliers_percentage': (len(outliers) / len(ts_data)) * 100,
            'sudden_changes_count': len(sudden_changes),
            'outlier_timestamps': outliers['time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()[:10],
            'sudden_change_timestamps': sudden_changes['time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()[:10]
        }
    
    def _create_temporal_visualizations(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Crear visualizaciones temporales."""
        temporal_viz_dir = Path(self.config.visualizations_dir) / 'temporal'
        temporal_viz_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Gráficos de patrones temporales
            self._plot_temporal_patterns(data, results, temporal_viz_dir)
            
            # 2. Análisis de gaps
            self._plot_gap_analysis(data, results, temporal_viz_dir)
            
            # 3. Series temporales
            self._plot_time_series(data, results, temporal_viz_dir)
            
        except Exception as e:
            logger.error(f"Error creando visualizaciones temporales: {str(e)}")
    
    def _plot_temporal_patterns(self, data: pd.DataFrame, results: Dict[str, Any], output_dir: Path):
        """Crear gráficos de patrones temporales."""
        patterns = results.get('temporal_patterns', {})
        
        if not patterns:
            return
        
        for time_col, col_results in list(patterns.items())[:1]:  # Solo primera columna temporal
            patterns_data = col_results.get('patterns', {})
            
            if not patterns_data:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Patrón horario
            hourly = patterns_data.get('hourly_pattern', {})
            if 'hourly_distribution' in hourly:
                hours = list(hourly['hourly_distribution'].keys())
                counts = list(hourly['hourly_distribution'].values())
                
                axes[0,0].bar(hours, counts, color=self.config.color_palette['primary'], alpha=0.7)
                axes[0,0].set_title('Patrón por Hora del Día', fontweight='bold')
                axes[0,0].set_xlabel('Hora')
                axes[0,0].set_ylabel('Número de Registros')
                axes[0,0].grid(True, alpha=0.3)
            
            # Patrón diario
            daily = patterns_data.get('daily_pattern', {})
            if 'daily_distribution' in daily:
                days = list(daily['daily_distribution'].keys())
                counts = list(daily['daily_distribution'].values())
                
                axes[0,1].bar(range(len(days)), counts, color=self.config.color_palette['secondary'], alpha=0.7)
                axes[0,1].set_xticks(range(len(days)))
                axes[0,1].set_xticklabels([d[:3] for d in days], rotation=45)
                axes[0,1].set_title('Patrón por Día de la Semana', fontweight='bold')
                axes[0,1].set_ylabel('Número de Registros')
                axes[0,1].grid(True, alpha=0.3)
            
            # Información de gaps
            gaps_info = col_results.get('gaps', {})
            if gaps_info and not gaps_info.get('error'):
                gap_stats = [
                    f"Intervalos totales: {gaps_info.get('total_intervals', 0):,}",
                    f"Intervalo mediano: {gaps_info.get('median_interval_seconds', 0):.1f}s",
                    f"Gaps detectados: {gaps_info.get('number_of_gaps', 0)}",
                    f"Gap más largo: {gaps_info.get('largest_gap_seconds', 0):.1f}s"
                ]
                
                axes[1,0].axis('off')
                axes[1,0].text(0.1, 0.7, '\n'.join(gap_stats), fontsize=12, 
                             transform=axes[1,0].transAxes, verticalalignment='top')
                axes[1,0].set_title('Análisis de Gaps Temporales', fontweight='bold')
            
            # Información de frecuencia
            freq_info = col_results.get('frequency', {})
            if freq_info and not freq_info.get('error'):
                freq_stats = [
                    f"Intervalo común: {freq_info.get('most_common_interval_seconds', 0):.2f}s",
                    f"Frecuencia estimada: {freq_info.get('estimated_frequency_hz', 0):.2f} Hz",
                    f"Regularidad: {freq_info.get('regularity_score', 0):.1f}%",
                    f"Intervalos únicos: {freq_info.get('unique_intervals', 0)}"
                ]
                
                axes[1,1].axis('off')
                axes[1,1].text(0.1, 0.7, '\n'.join(freq_stats), fontsize=12,
                             transform=axes[1,1].transAxes, verticalalignment='top')
                axes[1,1].set_title('Análisis de Frecuencia', fontweight='bold')
            
            plt.suptitle(f'Análisis de Patrones Temporales - {time_col}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / f'temporal_patterns_{time_col}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            break  # Solo crear para la primera columna temporal
    
    def _plot_gap_analysis(self, data: pd.DataFrame, results: Dict[str, Any], output_dir: Path):
        """Crear gráfico de análisis de gaps."""
        # Implementación simplificada
        patterns = results.get('temporal_patterns', {})
        
        if not patterns:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        gap_counts = []
        time_cols = []
        
        for time_col, col_results in patterns.items():
            gaps_info = col_results.get('gaps', {})
            if gaps_info and not gaps_info.get('error'):
                gap_counts.append(gaps_info.get('number_of_gaps', 0))
                time_cols.append(time_col)
        
        if gap_counts:
            ax.bar(range(len(time_cols)), gap_counts, 
                  color=self.config.color_palette['warning'], alpha=0.7)
            ax.set_xticks(range(len(time_cols)))
            ax.set_xticklabels(time_cols, rotation=45, ha='right')
            ax.set_ylabel('Número de Gaps Detectados')
            ax.set_title('Análisis de Gaps por Columna Temporal', fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No se detectaron gaps significativos', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Análisis de Gaps Temporales', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gap_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_series(self, data: pd.DataFrame, results: Dict[str, Any], output_dir: Path):
        """Crear gráficos de series temporales."""
        patterns = results.get('temporal_patterns', {})
        
        if not patterns:
            return
        
        # Tomar la primera columna temporal
        time_col = list(patterns.keys())[0]
        series_analysis = patterns[time_col].get('series_analysis', {})
        
        if not series_analysis:
            return
        
        # Preparar datos temporales
        time_series = self._prepare_time_series(data[time_col])
        
        if time_series is None:
            return
        
        # Crear gráfico para hasta 3 series numéricas
        numeric_series = list(series_analysis.keys())[:3]
        
        if not numeric_series:
            return
        
        fig, axes = plt.subplots(len(numeric_series), 1, figsize=(12, 4 * len(numeric_series)))
        
        if len(numeric_series) == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_series):
            ax = axes[i]
            
            # Datos válidos
            valid_mask = time_series.notna() & data[col].notna()
            
            if valid_mask.sum() < 2:
                continue
            
            plot_time = time_series[valid_mask]
            plot_values = data[col][valid_mask]
            
            # Ordenar por tiempo
            sort_idx = plot_time.argsort()
            plot_time = plot_time.iloc[sort_idx]
            plot_values = plot_values.iloc[sort_idx]
            
            ax.plot(plot_time, plot_values, color=self.config.color_palette['primary'], 
                   linewidth=1, alpha=0.8)
            
            ax.set_title(f'Serie Temporal: {col}', fontweight='bold')
            ax.set_xlabel('Tiempo')
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Series Temporales del Dataset', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'time_series_overview.png', dpi=300, bbox_inches='tight')
        plt.close()