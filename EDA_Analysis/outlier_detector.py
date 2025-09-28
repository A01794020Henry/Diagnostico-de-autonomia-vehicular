"""
Detector de Valores Atípicos
============================

Módulo especializado para detectar y analizar valores atípicos (outliers)
en los datos del sistema vehicular usando múltiples métodos.

Autor: Sistema de diagnóstico de autonomía vehicular
Fecha: Septiembre 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy import stats

logger = logging.getLogger(__name__)


class OutlierDetector:
    """Detector de valores atípicos para datos vehiculares."""
    
    def __init__(self, config):
        """Inicializar detector de outliers."""
        self.config = config
        self.scaler = StandardScaler()
        
    def detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detectar valores atípicos usando múltiples métodos.
        
        Args:
            data: DataFrame a analizar
            
        Returns:
            Diccionario con resultados de detección de outliers
        """
        logger.info("Iniciando detección de valores atípicos")
        
        results = {
            'statistical_outliers': self._detect_statistical_outliers(data),
            'machine_learning_outliers': self._detect_ml_outliers(data),
            'outlier_summary': {},
            'outlier_impact': {}
        }
        
        # Generar resumen combinado
        results['outlier_summary'] = self._generate_outlier_summary(results)
        
        # Analizar impacto de los outliers
        results['outlier_impact'] = self._analyze_outlier_impact(data, results)
        
        # Crear visualizaciones
        self._create_outlier_visualizations(data, results)
        
        logger.info("Detección de outliers completada")
        return results
    
    def _detect_statistical_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detectar outliers usando métodos estadísticos."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {'error': 'No se encontraron columnas numéricas'}
        
        statistical_methods = {
            'z_score': self._z_score_outliers(numeric_data),
            'iqr_method': self._iqr_outliers(numeric_data),
            'modified_z_score': self._modified_z_score_outliers(numeric_data),
            'grubbs_test': self._grubbs_outliers(numeric_data)
        }
        
        return statistical_methods
    
    def _detect_ml_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detectar outliers usando métodos de machine learning."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {'error': 'No se encontraron columnas numéricas'}
        
        # Preparar datos (eliminar valores faltantes y escalar)
        clean_data = numeric_data.dropna()
        
        if len(clean_data) < 10:
            return {'error': 'Insuficientes datos para métodos ML'}
        
        try:
            scaled_data = self.scaler.fit_transform(clean_data)
        except Exception as e:
            logger.error(f"Error escalando datos: {str(e)}")
            return {'error': f'Error en escalado de datos: {str(e)}'}
        
        ml_methods = {
            'isolation_forest': self._isolation_forest_outliers(scaled_data, clean_data),
            'local_outlier_factor': self._lof_outliers(scaled_data, clean_data),
            'one_class_svm': self._one_class_svm_outliers(scaled_data, clean_data)
        }
        
        return ml_methods
    
    def _z_score_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detectar outliers usando Z-score."""
        outliers_by_column = {}
        threshold = 3.0  # Umbral estándar para Z-score
        
        for column in data.columns:
            series = data[column].dropna()
            if len(series) < 3:
                continue
            
            # Calcular Z-scores
            z_scores = np.abs(stats.zscore(series))
            outlier_mask = z_scores > threshold
            outlier_indices = series[outlier_mask].index.tolist()
            
            outliers_by_column[column] = {
                'outlier_indices': outlier_indices,
                'outlier_count': len(outlier_indices),
                'outlier_percentage': (len(outlier_indices) / len(series)) * 100,
                'threshold_used': threshold,
                'max_z_score': float(np.max(z_scores))
            }
        
        return {
            'method': 'Z-Score',
            'outliers_by_column': outliers_by_column,
            'total_outlier_points': sum(len(info['outlier_indices']) for info in outliers_by_column.values())
        }
    
    def _iqr_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detectar outliers usando método IQR."""
        outliers_by_column = {}
        
        for column in data.columns:
            series = data[column].dropna()
            if len(series) < 4:
                continue
            
            # Calcular cuartiles y IQR
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            # Límites para outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Detectar outliers
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_indices = series[outlier_mask].index.tolist()
            
            outliers_by_column[column] = {
                'outlier_indices': outlier_indices,
                'outlier_count': len(outlier_indices),
                'outlier_percentage': (len(outlier_indices) / len(series)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'iqr': IQR,
                'extreme_outliers': len(series[(series < Q1 - 3 * IQR) | (series > Q3 + 3 * IQR)])
            }
        
        return {
            'method': 'IQR',
            'outliers_by_column': outliers_by_column,
            'total_outlier_points': sum(len(info['outlier_indices']) for info in outliers_by_column.values())
        }
    
    def _modified_z_score_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detectar outliers usando Z-score modificado (basado en mediana)."""
        outliers_by_column = {}
        threshold = 3.5  # Umbral para Z-score modificado
        
        for column in data.columns:
            series = data[column].dropna()
            if len(series) < 3:
                continue
            
            # Calcular Z-score modificado
            median = series.median()
            mad = stats.median_abs_deviation(series, scale='normal')
            
            if mad == 0:
                # Si MAD es 0, usar desviación estándar
                mad = series.std()
                if mad == 0:
                    continue
            
            modified_z_scores = 0.6745 * (series - median) / mad
            outlier_mask = np.abs(modified_z_scores) > threshold
            outlier_indices = series[outlier_mask].index.tolist()
            
            outliers_by_column[column] = {
                'outlier_indices': outlier_indices,
                'outlier_count': len(outlier_indices),
                'outlier_percentage': (len(outlier_indices) / len(series)) * 100,
                'threshold_used': threshold,
                'max_modified_z_score': float(np.max(np.abs(modified_z_scores)))
            }
        
        return {
            'method': 'Modified Z-Score',
            'outliers_by_column': outliers_by_column,
            'total_outlier_points': sum(len(info['outlier_indices']) for info in outliers_by_column.values())
        }
    
    def _grubbs_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detectar outliers usando test de Grubbs (para muestras pequeñas)."""
        outliers_by_column = {}
        
        for column in data.columns:
            series = data[column].dropna()
            if len(series) < 7 or len(series) > 100:  # Grubbs es para muestras pequeñas
                continue
            
            # Test de Grubbs para una cola
            mean = series.mean()
            std = series.std()
            
            if std == 0:
                continue
            
            # Calcular estadístico de Grubbs
            grubbs_stats = np.abs(series - mean) / std
            max_grubbs = np.max(grubbs_stats)
            
            # Valor crítico aproximado para alpha=0.05
            n = len(series)
            t_critical = stats.t.ppf(1 - 0.05/(2*n), n-2)
            grubbs_critical = ((n-1) * np.sqrt(t_critical**2)) / np.sqrt(n * (n-2 + t_critical**2))
            
            # Detectar outlier más extremo
            if max_grubbs > grubbs_critical:
                outlier_idx = series.index[np.argmax(grubbs_stats)]
                outlier_indices = [outlier_idx]
            else:
                outlier_indices = []
            
            outliers_by_column[column] = {
                'outlier_indices': outlier_indices,
                'outlier_count': len(outlier_indices),
                'outlier_percentage': (len(outlier_indices) / len(series)) * 100,
                'grubbs_statistic': float(max_grubbs),
                'critical_value': float(grubbs_critical),
                'is_significant': max_grubbs > grubbs_critical
            }
        
        return {
            'method': 'Grubbs Test',
            'outliers_by_column': outliers_by_column,
            'total_outlier_points': sum(len(info['outlier_indices']) for info in outliers_by_column.values())
        }
    
    def _isolation_forest_outliers(self, scaled_data: np.ndarray, original_data: pd.DataFrame) -> Dict[str, Any]:
        """Detectar outliers usando Isolation Forest."""
        try:
            # Configurar Isolation Forest
            isolation_forest = IsolationForest(
                contamination=self.config.outlier_contamination,
                random_state=42,
                n_estimators=100
            )
            
            # Ajustar y predecir
            outlier_labels = isolation_forest.fit_predict(scaled_data)
            outlier_scores = isolation_forest.decision_function(scaled_data)
            
            # Obtener índices de outliers (-1 indica outlier)
            outlier_mask = outlier_labels == -1
            outlier_indices = original_data.index[outlier_mask].tolist()
            
            return {
                'method': 'Isolation Forest',
                'outlier_indices': outlier_indices,
                'outlier_count': len(outlier_indices),
                'outlier_percentage': (len(outlier_indices) / len(original_data)) * 100,
                'contamination_used': self.config.outlier_contamination,
                'outlier_scores': outlier_scores.tolist(),
                'threshold': float(np.percentile(outlier_scores, self.config.outlier_contamination * 100))
            }
            
        except Exception as e:
            logger.error(f"Error en Isolation Forest: {str(e)}")
            return {'error': str(e)}
    
    def _lof_outliers(self, scaled_data: np.ndarray, original_data: pd.DataFrame) -> Dict[str, Any]:
        """Detectar outliers usando Local Outlier Factor."""
        try:
            # Configurar LOF
            lof = LocalOutlierFactor(
                contamination=self.config.outlier_contamination,
                n_neighbors=min(20, len(scaled_data) - 1)
            )
            
            # Ajustar y predecir
            outlier_labels = lof.fit_predict(scaled_data)
            outlier_scores = lof.negative_outlier_factor_
            
            # Obtener índices de outliers (-1 indica outlier)
            outlier_mask = outlier_labels == -1
            outlier_indices = original_data.index[outlier_mask].tolist()
            
            return {
                'method': 'Local Outlier Factor',
                'outlier_indices': outlier_indices,
                'outlier_count': len(outlier_indices),
                'outlier_percentage': (len(outlier_indices) / len(original_data)) * 100,
                'contamination_used': self.config.outlier_contamination,
                'outlier_scores': outlier_scores.tolist(),
                'threshold': float(np.percentile(outlier_scores, self.config.outlier_contamination * 100))
            }
            
        except Exception as e:
            logger.error(f"Error en LOF: {str(e)}")
            return {'error': str(e)}
    
    def _one_class_svm_outliers(self, scaled_data: np.ndarray, original_data: pd.DataFrame) -> Dict[str, Any]:
        """Detectar outliers usando One-Class SVM."""
        try:
            # Configurar One-Class SVM
            svm = OneClassSVM(
                nu=self.config.outlier_contamination,
                kernel='rbf',
                gamma='scale'
            )
            
            # Ajustar y predecir
            outlier_labels = svm.fit_predict(scaled_data)
            outlier_scores = svm.decision_function(scaled_data)
            
            # Obtener índices de outliers (-1 indica outlier)
            outlier_mask = outlier_labels == -1
            outlier_indices = original_data.index[outlier_mask].tolist()
            
            return {
                'method': 'One-Class SVM',
                'outlier_indices': outlier_indices,
                'outlier_count': len(outlier_indices),
                'outlier_percentage': (len(outlier_indices) / len(original_data)) * 100,
                'nu_used': self.config.outlier_contamination,
                'outlier_scores': outlier_scores.tolist(),
                'threshold': float(np.percentile(outlier_scores, self.config.outlier_contamination * 100))
            }
            
        except Exception as e:
            logger.error(f"Error en One-Class SVM: {str(e)}")
            return {'error': str(e)}
    
    def _generate_outlier_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generar resumen de todos los métodos de detección."""
        summary = {
            'methods_used': [],
            'total_outliers_by_method': {},
            'consensus_outliers': [],
            'method_agreement': {}
        }
        
        all_outlier_indices = []
        
        # Procesar métodos estadísticos
        statistical_results = results.get('statistical_outliers', {})
        for method_name, method_results in statistical_results.items():
            if isinstance(method_results, dict) and 'outliers_by_column' in method_results:
                summary['methods_used'].append(method_name)
                
                # Recopilar todos los outliers de este método
                method_outliers = []
                for column_results in method_results['outliers_by_column'].values():
                    method_outliers.extend(column_results.get('outlier_indices', []))
                
                summary['total_outliers_by_method'][method_name] = len(set(method_outliers))
                all_outlier_indices.extend(method_outliers)
        
        # Procesar métodos de ML
        ml_results = results.get('machine_learning_outliers', {})
        for method_name, method_results in ml_results.items():
            if isinstance(method_results, dict) and 'outlier_indices' in method_results:
                summary['methods_used'].append(method_name)
                method_outliers = method_results['outlier_indices']
                summary['total_outliers_by_method'][method_name] = len(method_outliers)
                all_outlier_indices.extend(method_outliers)
        
        # Encontrar consenso (outliers detectados por múltiples métodos)
        if all_outlier_indices:
            from collections import Counter
            outlier_counts = Counter(all_outlier_indices)
            
            # Outliers detectados por al menos 2 métodos
            consensus_threshold = 2
            summary['consensus_outliers'] = [
                idx for idx, count in outlier_counts.items() 
                if count >= consensus_threshold
            ]
            
            summary['method_agreement'] = {
                'total_unique_outliers': len(set(all_outlier_indices)),
                'consensus_outliers_count': len(summary['consensus_outliers']),
                'consensus_threshold': consensus_threshold,
                'outlier_frequency': dict(outlier_counts)
            }
        
        return summary
    
    def _analyze_outlier_impact(self, data: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analizar el impacto de los outliers en las estadísticas."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {'error': 'No hay datos numéricos para analizar impacto'}
        
        # Obtener outliers de consenso
        consensus_outliers = results.get('outlier_summary', {}).get('consensus_outliers', [])
        
        if not consensus_outliers:
            return {'message': 'No hay outliers de consenso para analizar impacto'}
        
        impact_analysis = {}
        
        for column in numeric_data.columns:
            original_series = numeric_data[column].dropna()
            
            if len(original_series) == 0:
                continue
            
            # Crear serie sin outliers
            outlier_indices_in_column = [idx for idx in consensus_outliers if idx in original_series.index]
            clean_series = original_series.drop(outlier_indices_in_column)
            
            if len(clean_series) == 0:
                continue
            
            # Calcular estadísticas con y sin outliers
            original_stats = {
                'mean': original_series.mean(),
                'median': original_series.median(),
                'std': original_series.std(),
                'min': original_series.min(),
                'max': original_series.max()
            }
            
            clean_stats = {
                'mean': clean_series.mean(),
                'median': clean_series.median(),
                'std': clean_series.std(),
                'min': clean_series.min(),
                'max': clean_series.max()
            }
            
            # Calcular cambios porcentuales
            impact_analysis[column] = {
                'outliers_in_column': len(outlier_indices_in_column),
                'outlier_percentage': (len(outlier_indices_in_column) / len(original_series)) * 100,
                'original_stats': original_stats,
                'clean_stats': clean_stats,
                'impact_metrics': {
                    'mean_change_percent': abs(original_stats['mean'] - clean_stats['mean']) / abs(original_stats['mean']) * 100 if original_stats['mean'] != 0 else 0,
                    'std_change_percent': abs(original_stats['std'] - clean_stats['std']) / original_stats['std'] * 100 if original_stats['std'] != 0 else 0,
                    'range_change_percent': abs((original_stats['max'] - original_stats['min']) - (clean_stats['max'] - clean_stats['min'])) / (original_stats['max'] - original_stats['min']) * 100 if (original_stats['max'] - original_stats['min']) != 0 else 0
                }
            }
        
        return impact_analysis
    
    def _create_outlier_visualizations(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Crear visualizaciones de outliers."""
        outlier_viz_dir = Path(self.config.visualizations_dir) / 'outliers'
        outlier_viz_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Boxplots con outliers marcados
            self._plot_outlier_boxplots(data, results, outlier_viz_dir)
            
            # 2. Comparación de métodos
            self._plot_method_comparison(results, outlier_viz_dir)
            
            # 3. Scatter plots con outliers
            self._plot_outlier_scatter(data, results, outlier_viz_dir)
            
        except Exception as e:
            logger.error(f"Error creando visualizaciones de outliers: {str(e)}")
    
    def _plot_outlier_boxplots(self, data: pd.DataFrame, results: Dict[str, Any], output_dir: Path):
        """Crear boxplots con outliers marcados."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return
        
        # Seleccionar hasta 6 columnas
        cols_to_plot = numeric_data.columns[:6]
        
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        
        if n_rows * n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            pass
        else:
            axes = axes.flatten()
        
        # Obtener outliers de consenso
        consensus_outliers = results.get('outlier_summary', {}).get('consensus_outliers', [])
        
        for i, column in enumerate(cols_to_plot):
            ax = axes[i] if len(cols_to_plot) > 1 else axes[0]
            
            series = numeric_data[column].dropna()
            
            if len(series) == 0:
                ax.set_visible(False)
                continue
            
            # Crear boxplot
            box_plot = ax.boxplot(series, patch_artist=True)
            box_plot['boxes'][0].set_facecolor(self.config.color_palette['primary'])
            box_plot['boxes'][0].set_alpha(0.7)
            
            # Marcar outliers de consenso si los hay
            column_consensus_outliers = [idx for idx in consensus_outliers if idx in series.index]
            
            if column_consensus_outliers:
                outlier_values = series.loc[column_consensus_outliers]
                ax.scatter([1] * len(outlier_values), outlier_values, 
                          color=self.config.color_palette['error'], 
                          s=50, alpha=0.8, marker='D', 
                          label=f'Consenso ({len(outlier_values)})')
                ax.legend()
            
            ax.set_title(f'{column}\nOutliers: {len(column_consensus_outliers)}', 
                        fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Ocultar ejes vacíos
        for i in range(len(cols_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Análisis de Outliers - Boxplots', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'outlier_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_method_comparison(self, results: Dict[str, Any], output_dir: Path):
        """Crear comparación visual de métodos."""
        summary = results.get('outlier_summary', {})
        method_counts = summary.get('total_outliers_by_method', {})
        
        if not method_counts:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        methods = list(method_counts.keys())
        counts = list(method_counts.values())
        
        # Crear gráfico de barras
        bars = ax.bar(range(len(methods)), counts, 
                     color=self.config.color_palette['primary'], alpha=0.7)
        
        # Configurar ejes
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('Número de Outliers Detectados')
        ax.set_title('Comparación de Métodos de Detección de Outliers', fontweight='bold')
        
        # Añadir valores en las barras
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts) * 0.01, 
                   str(count), ha='center', va='bottom', fontweight='bold')
        
        # Línea de consenso si existe
        consensus_count = len(summary.get('consensus_outliers', []))
        if consensus_count > 0:
            ax.axhline(y=consensus_count, color=self.config.color_palette['error'], 
                      linestyle='--', linewidth=2, 
                      label=f'Outliers de Consenso: {consensus_count}')
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_outlier_scatter(self, data: pd.DataFrame, results: Dict[str, Any], output_dir: Path):
        """Crear scatter plot con outliers marcados."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return
        
        # Tomar las primeras dos columnas
        col1, col2 = numeric_data.columns[0], numeric_data.columns[1]
        
        # Datos válidos
        valid_mask = numeric_data[col1].notna() & numeric_data[col2].notna()
        plot_data = numeric_data[valid_mask]
        
        if len(plot_data) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Obtener outliers de consenso
        consensus_outliers = results.get('outlier_summary', {}).get('consensus_outliers', [])
        
        # Separar puntos normales y outliers
        normal_mask = ~plot_data.index.isin(consensus_outliers)
        outlier_mask = plot_data.index.isin(consensus_outliers)
        
        # Plotear puntos normales
        if normal_mask.any():
            ax.scatter(plot_data.loc[normal_mask, col1], plot_data.loc[normal_mask, col2],
                      color=self.config.color_palette['primary'], alpha=0.6, 
                      s=30, label='Datos Normales')
        
        # Plotear outliers
        if outlier_mask.any():
            ax.scatter(plot_data.loc[outlier_mask, col1], plot_data.loc[outlier_mask, col2],
                      color=self.config.color_palette['error'], alpha=0.8, 
                      s=80, marker='D', label=f'Outliers ({outlier_mask.sum()})')
        
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f'Distribución de Outliers: {col1} vs {col2}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'outlier_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()