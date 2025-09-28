"""
Analizador de Correlaciones
===========================

Módulo especializado para análisis de correlaciones entre variables
del sistema de diagnóstico vehicular.

Autor: Sistema de diagnóstico de autonomía vehicular  
Fecha: Septiembre 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kendalltau
from itertools import combinations

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Analizador de correlaciones para datos vehiculares."""
    
    def __init__(self, config):
        """Inicializar analizador de correlaciones."""
        self.config = config
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Realizar análisis completo de correlaciones.
        
        Args:
            data: DataFrame a analizar
            
        Returns:
            Diccionario con resultados del análisis de correlaciones
        """
        logger.info("Iniciando análisis de correlaciones")
        
        results = {
            'correlation_matrices': self._calculate_correlation_matrices(data),
            'high_correlations': self._find_high_correlations(data),
            'correlation_summary': self._correlation_summary(data),
            'pairwise_analysis': self._pairwise_analysis(data)
        }
        
        # Generar visualizaciones
        self._create_correlation_visualizations(data, results)
        
        logger.info("Análisis de correlaciones completado")
        return results
    
    def _calculate_correlation_matrices(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calcular matrices de correlación usando diferentes métodos."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return {'error': 'Insuficientes variables numéricas para correlación'}
        
        matrices = {}
        
        # Correlación de Pearson
        try:
            pearson_corr = numeric_data.corr(method='pearson')
            matrices['pearson'] = {
                'matrix': pearson_corr.to_dict(),
                'description': 'Correlación lineal (Pearson)'
            }
        except Exception as e:
            logger.error(f"Error calculando correlación de Pearson: {str(e)}")
        
        # Correlación de Spearman  
        try:
            spearman_corr = numeric_data.corr(method='spearman')
            matrices['spearman'] = {
                'matrix': spearman_corr.to_dict(),
                'description': 'Correlación de rango (Spearman)'
            }
        except Exception as e:
            logger.error(f"Error calculando correlación de Spearman: {str(e)}")
        
        # Correlación de Kendall
        try:
            kendall_corr = numeric_data.corr(method='kendall')
            matrices['kendall'] = {
                'matrix': kendall_corr.to_dict(),
                'description': 'Correlación de Kendall'
            }
        except Exception as e:
            logger.error(f"Error calculando correlación de Kendall: {str(e)}")
        
        return matrices
    
    def _find_high_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Encontrar correlaciones altas."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return {'high_correlations': []}
        
        correlation_matrix = numeric_data.corr()
        high_correlations = []
        
        # Buscar correlaciones por encima del umbral
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                
                if abs(corr_value) >= self.config.correlation_threshold:
                    var1 = correlation_matrix.columns[i]
                    var2 = correlation_matrix.columns[j]
                    
                    high_correlations.append({
                        'variable_1': var1,
                        'variable_2': var2,
                        'correlation': corr_value,
                        'abs_correlation': abs(corr_value),
                        'relationship': 'Positiva' if corr_value > 0 else 'Negativa',
                        'strength': self._interpret_correlation_strength(abs(corr_value))
                    })
        
        # Ordenar por correlación absoluta
        high_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        return {
            'high_correlations': high_correlations,
            'total_pairs': len(high_correlations),
            'threshold_used': self.config.correlation_threshold
        }
    
    def _correlation_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generar resumen estadístico de correlaciones."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return {'error': 'Insuficientes variables para análisis'}
        
        correlation_matrix = numeric_data.corr()
        
        # Extraer triángulo superior (sin diagonal)
        upper_triangle = np.triu(correlation_matrix, k=1)
        correlations = upper_triangle[upper_triangle != 0]
        
        if len(correlations) == 0:
            return {'error': 'No se calcularon correlaciones válidas'}
        
        summary = {
            'total_pairs': len(correlations),
            'mean_correlation': float(np.mean(np.abs(correlations))),
            'median_correlation': float(np.median(np.abs(correlations))),
            'max_correlation': float(np.max(np.abs(correlations))),
            'min_correlation': float(np.min(np.abs(correlations))),
            'std_correlation': float(np.std(np.abs(correlations))),
            'positive_correlations': int(np.sum(correlations > 0)),
            'negative_correlations': int(np.sum(correlations < 0)),
            'strong_correlations': int(np.sum(np.abs(correlations) >= 0.7)),
            'moderate_correlations': int(np.sum((np.abs(correlations) >= 0.3) & (np.abs(correlations) < 0.7))),
            'weak_correlations': int(np.sum(np.abs(correlations) < 0.3))
        }
        
        return summary
    
    def _pairwise_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Análisis detallado por pares de variables."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return {'pairs_analyzed': []}
        
        pairs_analysis = []
        
        # Analizar hasta 10 pares más correlacionados
        correlation_matrix = numeric_data.corr()
        
        # Encontrar los pares más correlacionados
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if not np.isnan(corr_value):
                    correlations.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # Ordenar por correlación absoluta
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Analizar top 10 pares
        for pair_info in correlations[:10]:
            var1, var2 = pair_info['var1'], pair_info['var2']
            
            try:
                # Estadísticas del par
                series1 = data[var1].dropna()
                series2 = data[var2].dropna()
                
                # Encontrar índices comunes (sin valores faltantes en ambas)
                common_indices = series1.index.intersection(series2.index)
                common_data1 = series1.loc[common_indices]
                common_data2 = series2.loc[common_indices]
                
                if len(common_data1) < 3:
                    continue
                
                # Calcular correlaciones con p-values
                pearson_corr, pearson_p = pearsonr(common_data1, common_data2)
                spearman_corr, spearman_p = spearmanr(common_data1, common_data2)
                
                pair_analysis = {
                    'variable_1': var1,
                    'variable_2': var2,
                    'sample_size': len(common_data1),
                    'pearson': {
                        'correlation': pearson_corr,
                        'p_value': pearson_p,
                        'significant': pearson_p < 0.05
                    },
                    'spearman': {
                        'correlation': spearman_corr,
                        'p_value': spearman_p,
                        'significant': spearman_p < 0.05
                    },
                    'relationship_strength': self._interpret_correlation_strength(abs(pearson_corr)),
                    'relationship_direction': 'Positiva' if pearson_corr > 0 else 'Negativa'
                }
                
                pairs_analysis.append(pair_analysis)
                
            except Exception as e:
                logger.error(f"Error analizando par {var1}-{var2}: {str(e)}")
                continue
        
        return {'pairs_analyzed': pairs_analysis}
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpretar la fuerza de una correlación."""
        correlation = abs(correlation)
        
        if correlation >= 0.9:
            return "Muy fuerte"
        elif correlation >= 0.7:
            return "Fuerte" 
        elif correlation >= 0.5:
            return "Moderada"
        elif correlation >= 0.3:
            return "Débil"
        else:
            return "Muy débil"
    
    def _create_correlation_visualizations(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Crear visualizaciones de correlación."""
        corr_viz_dir = Path(self.config.visualizations_dir) / 'correlations'
        corr_viz_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Matriz de correlación principal
            self._plot_correlation_heatmap(data, corr_viz_dir)
            
            # 2. Gráfico de correlaciones altas
            self._plot_high_correlations(results.get('high_correlations', {}), corr_viz_dir)
            
            # 3. Distribución de correlaciones
            self._plot_correlation_distribution(data, corr_viz_dir)
            
        except Exception as e:
            logger.error(f"Error creando visualizaciones de correlación: {str(e)}")
    
    def _plot_correlation_heatmap(self, data: pd.DataFrame, output_dir: Path):
        """Crear mapa de calor de correlaciones."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return
        
        correlation_matrix = numeric_data.corr()
        
        # Configurar el gráfico
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Crear máscara para el triángulo superior
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Crear el heatmap
        im = ax.imshow(correlation_matrix.values, cmap='RdBu_r', aspect='auto', 
                      vmin=-1, vmax=1)
        
        # Configurar ejes
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(correlation_matrix.columns)
        
        # Añadir valores de correlación
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                if not mask[i, j]:  # Solo mostrar triángulo inferior
                    text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlación', rotation=270, labelpad=20)
        
        ax.set_title('Matriz de Correlación', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_high_correlations(self, high_corr_results: Dict[str, Any], output_dir: Path):
        """Crear gráfico de correlaciones altas."""
        high_correlations = high_corr_results.get('high_correlations', [])
        
        if not high_correlations:
            # Crear gráfico indicando que no hay correlaciones altas
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'No se encontraron correlaciones\npor encima del umbral ({self.config.correlation_threshold})', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=14, fontweight='bold')
            ax.set_title('Correlaciones Altas', fontsize=16, fontweight='bold')
            ax.axis('off')
            plt.savefig(output_dir / 'high_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # Tomar top 15 correlaciones
        top_correlations = high_correlations[:15]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Preparar datos
        pair_labels = [f"{item['variable_1']} - {item['variable_2']}" for item in top_correlations]
        correlations = [item['correlation'] for item in top_correlations]
        
        # Colores basados en signo
        colors = [self.config.color_palette['success'] if corr > 0 else self.config.color_palette['error'] 
                 for corr in correlations]
        
        # Crear gráfico de barras horizontal
        bars = ax.barh(range(len(correlations)), correlations, color=colors, alpha=0.7)
        
        # Configurar ejes
        ax.set_yticks(range(len(correlations)))
        ax.set_yticklabels(pair_labels, fontsize=8)
        ax.set_xlabel('Correlación')
        ax.set_title('Top Correlaciones Altas', fontsize=14, fontweight='bold')
        
        # Línea vertical en cero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Añadir valores en las barras
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            ax.text(bar.get_width() + (0.01 if corr > 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                   f'{corr:.3f}', ha='left' if corr > 0 else 'right', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'high_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_distribution(self, data: pd.DataFrame, output_dir: Path):
        """Crear histograma de distribución de correlaciones."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return
        
        correlation_matrix = numeric_data.corr()
        
        # Extraer triángulo superior (sin diagonal)
        upper_triangle = np.triu(correlation_matrix, k=1)
        correlations = upper_triangle[upper_triangle != 0]
        
        if len(correlations) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histograma de todas las correlaciones
        ax1.hist(correlations, bins=30, color=self.config.color_palette['primary'], 
                alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(correlations), color=self.config.color_palette['error'], 
                   linestyle='--', linewidth=2, label=f'Media: {np.mean(correlations):.3f}')
        ax1.axvline(np.median(correlations), color=self.config.color_palette['warning'], 
                   linestyle='-', linewidth=2, label=f'Mediana: {np.median(correlations):.3f}')
        ax1.set_xlabel('Correlación')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución de Correlaciones', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histograma de correlaciones absolutas
        abs_correlations = np.abs(correlations)
        ax2.hist(abs_correlations, bins=30, color=self.config.color_palette['secondary'], 
                alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(abs_correlations), color=self.config.color_palette['error'], 
                   linestyle='--', linewidth=2, label=f'Media: {np.mean(abs_correlations):.3f}')
        ax2.axvline(self.config.correlation_threshold, color=self.config.color_palette['accent'], 
                   linestyle=':', linewidth=2, label=f'Umbral: {self.config.correlation_threshold}')
        ax2.set_xlabel('Correlación Absoluta')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Correlaciones Absolutas', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()