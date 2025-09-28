"""
Analizador Estadístico Avanzado
==============================

Módulo para realizar análisis estadísticos detallados de los datos vehiculares,
incluyendo análisis descriptivo, tests de normalidad y análisis inferencial.

Autor: Sistema de diagnóstico de autonomía vehicular
Fecha: Septiembre 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Analizador estadístico para datos vehiculares."""
    
    def __init__(self, config):
        """Inicializar analizador estadístico."""
        self.config = config
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Realizar análisis estadístico completo.
        
        Args:
            data: DataFrame a analizar
            
        Returns:
            Diccionario con resultados estadísticos
        """
        logger.info("Iniciando análisis estadístico completo")
        
        results = {
            'descriptive_stats': self._descriptive_analysis(data),
            'distribution_analysis': self._distribution_analysis(data),
            'normality_tests': self._normality_tests(data),
            'homogeneity_tests': self._homogeneity_tests(data),
            'central_tendency': self._central_tendency_analysis(data),
            'variability_analysis': self._variability_analysis(data),
            'skewness_kurtosis': self._skewness_kurtosis_analysis(data)
        }
        
        # Generar visualizaciones estadísticas
        self._create_statistical_visualizations(data, results)
        
        logger.info("Análisis estadístico completado")
        return results
    
    def _descriptive_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Análisis descriptivo básico."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {'error': 'No se encontraron columnas numéricas'}
        
        descriptive = {
            'numeric_columns_count': len(numeric_data.columns),
            'basic_stats': numeric_data.describe().to_dict(),
            'additional_stats': {}
        }
        
        # Estadísticas adicionales por columna
        for col in numeric_data.columns:
            series = numeric_data[col].dropna()
            if len(series) > 0:
                descriptive['additional_stats'][col] = {
                    'variance': series.var(),
                    'coefficient_of_variation': series.std() / series.mean() if series.mean() != 0 else np.inf,
                    'range': series.max() - series.min(),
                    'iqr': series.quantile(0.75) - series.quantile(0.25),
                    'mad': stats.median_abs_deviation(series),
                    'geometric_mean': stats.gmean(series[series > 0]) if (series > 0).any() else None,
                    'harmonic_mean': stats.hmean(series[series > 0]) if (series > 0).any() else None
                }
        
        return descriptive
    
    def _distribution_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Análisis de distribuciones."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        distribution_results = {
            'distribution_fits': {},
            'distribution_summaries': {}
        }
        
        # Distribuciones a probar
        distributions_to_test = [
            stats.norm,      # Normal
            stats.lognorm,   # Log-normal
            stats.expon,     # Exponencial
            stats.gamma,     # Gamma
            stats.beta,      # Beta
            stats.uniform,   # Uniforme
        ]
        
        for col in numeric_data.columns:
            series = numeric_data[col].dropna()
            
            if len(series) < 10:  # Muy pocos datos
                continue
            
            # Resumen de la distribución
            distribution_results['distribution_summaries'][col] = {
                'min': series.min(),
                'max': series.max(),
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'skewness': stats.skew(series),
                'kurtosis': stats.kurtosis(series)
            }
            
            # Prueba de ajuste de distribuciones
            best_distributions = []
            
            for distribution in distributions_to_test:
                try:
                    # Ajustar distribución
                    params = distribution.fit(series)
                    
                    # Test de Kolmogorov-Smirnov
                    D, p_value = stats.kstest(series, lambda x: distribution.cdf(x, *params))
                    
                    best_distributions.append({
                        'distribution': distribution.name,
                        'parameters': params,
                        'ks_statistic': D,
                        'p_value': p_value,
                        'aic': self._calculate_aic(series, distribution, params),
                        'bic': self._calculate_bic(series, distribution, params)
                    })
                    
                except Exception as e:
                    logger.debug(f"Error ajustando {distribution.name} para {col}: {str(e)}")
                    continue
            
            # Ordenar por AIC (mejor ajuste)
            best_distributions.sort(key=lambda x: x['aic'])
            distribution_results['distribution_fits'][col] = best_distributions[:3]  # Top 3
        
        return distribution_results
    
    def _normality_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Tests de normalidad."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        normality_results = {
            'shapiro_wilk': {},
            'anderson_darling': {},
            'jarque_bera': {},
            'normalcy_summary': {}
        }
        
        for col in numeric_data.columns:
            series = numeric_data[col].dropna()
            
            if len(series) < 3:
                continue
            
            try:
                # Test de Shapiro-Wilk (para muestras pequeñas < 5000)
                if len(series) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(series)
                    normality_results['shapiro_wilk'][col] = {
                        'statistic': shapiro_stat,
                        'p_value': shapiro_p,
                        'is_normal': shapiro_p > 0.05
                    }
                
                # Test de Anderson-Darling
                anderson_result = stats.anderson(series, dist='norm')
                normality_results['anderson_darling'][col] = {
                    'statistic': anderson_result.statistic,
                    'critical_values': anderson_result.critical_values.tolist(),
                    'significance_levels': anderson_result.significance_level.tolist(),
                    'is_normal': anderson_result.statistic < anderson_result.critical_values[2]  # 5% nivel
                }
                
                # Test de Jarque-Bera
                if len(series) > 7:  # Mínimo requerido
                    jb_stat, jb_p = stats.jarque_bera(series)
                    normality_results['jarque_bera'][col] = {
                        'statistic': jb_stat,
                        'p_value': jb_p,
                        'is_normal': jb_p > 0.05
                    }
                
                # Resumen de normalidad
                tests_passed = 0
                total_tests = 0
                
                if col in normality_results['shapiro_wilk']:
                    total_tests += 1
                    if normality_results['shapiro_wilk'][col]['is_normal']:
                        tests_passed += 1
                
                if col in normality_results['anderson_darling']:
                    total_tests += 1
                    if normality_results['anderson_darling'][col]['is_normal']:
                        tests_passed += 1
                
                if col in normality_results['jarque_bera']:
                    total_tests += 1
                    if normality_results['jarque_bera'][col]['is_normal']:
                        tests_passed += 1
                
                normality_results['normalcy_summary'][col] = {
                    'tests_passed': tests_passed,
                    'total_tests': total_tests,
                    'normalcy_score': tests_passed / total_tests if total_tests > 0 else 0,
                    'recommendation': self._get_normality_recommendation(tests_passed, total_tests)
                }
                
            except Exception as e:
                logger.error(f"Error en tests de normalidad para {col}: {str(e)}")
        
        return normality_results
    
    def _homogeneity_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Tests de homogeneidad de varianzas."""
        numeric_data = data.select_dtypes(include=[np.number])
        categorical_data = data.select_dtypes(include=['object', 'category'])
        
        if categorical_data.empty:
            return {'error': 'No se encontraron variables categóricas para agrupar'}
        
        homogeneity_results = {
            'levene_tests': {},
            'bartlett_tests': {},
            'fligner_tests': {}
        }
        
        for cat_col in categorical_data.columns:
            # Tomar solo categorías más frecuentes para evitar grupos muy pequeños
            top_categories = data[cat_col].value_counts().head(5).index.tolist()
            
            for num_col in numeric_data.columns:
                try:
                    # Crear grupos por categoría
                    groups = []
                    group_names = []
                    
                    for category in top_categories:
                        group_data = data[data[cat_col] == category][num_col].dropna()
                        if len(group_data) >= 3:  # Mínimo para el test
                            groups.append(group_data)
                            group_names.append(str(category))
                    
                    if len(groups) < 2:
                        continue
                    
                    test_key = f"{num_col}_by_{cat_col}"
                    
                    # Test de Levene (robusto)
                    levene_stat, levene_p = stats.levene(*groups)
                    homogeneity_results['levene_tests'][test_key] = {
                        'statistic': levene_stat,
                        'p_value': levene_p,
                        'homogeneous': levene_p > 0.05,
                        'groups': group_names,
                        'group_sizes': [len(g) for g in groups]
                    }
                    
                    # Test de Bartlett (asume normalidad)
                    bartlett_stat, bartlett_p = stats.bartlett(*groups)
                    homogeneity_results['bartlett_tests'][test_key] = {
                        'statistic': bartlett_stat,
                        'p_value': bartlett_p,
                        'homogeneous': bartlett_p > 0.05,
                        'groups': group_names
                    }
                    
                    # Test de Fligner-Killeen (no paramétrico)
                    fligner_stat, fligner_p = stats.fligner(*groups)
                    homogeneity_results['fligner_tests'][test_key] = {
                        'statistic': fligner_stat,
                        'p_value': fligner_p,
                        'homogeneous': fligner_p > 0.05,
                        'groups': group_names
                    }
                    
                except Exception as e:
                    logger.error(f"Error en test de homogeneidad para {num_col} by {cat_col}: {str(e)}")
        
        return homogeneity_results
    
    def _central_tendency_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Análisis de tendencia central."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        central_tendency = {
            'measures_comparison': {},
            'robust_statistics': {}
        }
        
        for col in numeric_data.columns:
            series = numeric_data[col].dropna()
            
            if len(series) == 0:
                continue
            
            # Medidas de tendencia central
            mean_val = series.mean()
            median_val = series.median()
            
            try:
                mode_val = stats.mode(series, keepdims=False)[0]
            except:
                mode_val = None
            
            # Medias robustas
            trimmed_mean_5 = stats.trim_mean(series, 0.05)  # 5% trimmed
            trimmed_mean_10 = stats.trim_mean(series, 0.10)  # 10% trimmed
            
            central_tendency['measures_comparison'][col] = {
                'mean': mean_val,
                'median': median_val,
                'mode': mode_val,
                'trimmed_mean_5pct': trimmed_mean_5,
                'trimmed_mean_10pct': trimmed_mean_10,
                'mean_median_diff': abs(mean_val - median_val),
                'symmetry_indicator': self._assess_symmetry(mean_val, median_val)
            }
            
            # Estadísticas robustas
            central_tendency['robust_statistics'][col] = {
                'median_abs_deviation': stats.median_abs_deviation(series),
                'iqr': series.quantile(0.75) - series.quantile(0.25),
                'midhinge': (series.quantile(0.75) + series.quantile(0.25)) / 2,
                'midrange': (series.min() + series.max()) / 2,
                'tukey_biweight_location': self._tukey_biweight_location(series)
            }
        
        return central_tendency
    
    def _variability_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Análisis de variabilidad."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        variability = {
            'dispersion_measures': {},
            'relative_variability': {},
            'quartile_analysis': {}
        }
        
        for col in numeric_data.columns:
            series = numeric_data[col].dropna()
            
            if len(series) == 0:
                continue
            
            mean_val = series.mean()
            std_val = series.std()
            
            # Medidas de dispersión absoluta
            variability['dispersion_measures'][col] = {
                'variance': series.var(),
                'std_deviation': std_val,
                'mean_abs_deviation': np.mean(np.abs(series - mean_val)),
                'median_abs_deviation': stats.median_abs_deviation(series),
                'range': series.max() - series.min(),
                'iqr': series.quantile(0.75) - series.quantile(0.25)
            }
            
            # Medidas de variabilidad relativa
            variability['relative_variability'][col] = {
                'coefficient_of_variation': std_val / mean_val if mean_val != 0 else np.inf,
                'quartile_coefficient': ((series.quantile(0.75) - series.quantile(0.25)) / 
                                       (series.quantile(0.75) + series.quantile(0.25))) 
                                      if (series.quantile(0.75) + series.quantile(0.25)) != 0 else np.inf,
                'relative_range': (series.max() - series.min()) / mean_val if mean_val != 0 else np.inf
            }
            
            # Análisis de cuartiles
            q1 = series.quantile(0.25)
            q2 = series.quantile(0.50)  # mediana
            q3 = series.quantile(0.75)
            
            variability['quartile_analysis'][col] = {
                'q1': q1,
                'q2': q2,
                'q3': q3,
                'iqr': q3 - q1,
                'lower_fence': q1 - 1.5 * (q3 - q1),
                'upper_fence': q3 + 1.5 * (q3 - q1),
                'outliers_count': ((series < (q1 - 1.5 * (q3 - q1))) | 
                                 (series > (q3 + 1.5 * (q3 - q1)))).sum()
            }
        
        return variability
    
    def _skewness_kurtosis_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Análisis de asimetría y curtosis."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        shape_analysis = {
            'skewness_analysis': {},
            'kurtosis_analysis': {},
            'shape_summary': {}
        }
        
        for col in numeric_data.columns:
            series = numeric_data[col].dropna()
            
            if len(series) < 3:
                continue
            
            # Asimetría
            skewness = stats.skew(series)
            skewness_test = stats.skewtest(series) if len(series) >= 8 else (None, None)
            
            shape_analysis['skewness_analysis'][col] = {
                'skewness': skewness,
                'skewness_test_statistic': skewness_test[0] if skewness_test[0] else None,
                'skewness_test_pvalue': skewness_test[1] if skewness_test[1] else None,
                'skewness_interpretation': self._interpret_skewness(skewness)
            }
            
            # Curtosis
            kurt = stats.kurtosis(series, fisher=True)  # Excess kurtosis
            kurtosis_test = stats.kurtosistest(series) if len(series) >= 20 else (None, None)
            
            shape_analysis['kurtosis_analysis'][col] = {
                'kurtosis': kurt,
                'kurtosis_test_statistic': kurtosis_test[0] if kurtosis_test[0] else None,
                'kurtosis_test_pvalue': kurtosis_test[1] if kurtosis_test[1] else None,
                'kurtosis_interpretation': self._interpret_kurtosis(kurt)
            }
            
            # Resumen de forma
            shape_analysis['shape_summary'][col] = {
                'distribution_shape': self._classify_distribution_shape(skewness, kurt),
                'symmetry_level': self._classify_symmetry(skewness),
                'tail_behavior': self._classify_tails(kurt)
            }
        
        return shape_analysis
    
    def _calculate_aic(self, data, distribution, params):
        """Calcular Criterio de Información de Akaike."""
        try:
            log_likelihood = np.sum(distribution.logpdf(data, *params))
            k = len(params)
            n = len(data)
            return 2 * k - 2 * log_likelihood
        except:
            return np.inf
    
    def _calculate_bic(self, data, distribution, params):
        """Calcular Criterio de Información Bayesiano."""
        try:
            log_likelihood = np.sum(distribution.logpdf(data, *params))
            k = len(params)
            n = len(data)
            return k * np.log(n) - 2 * log_likelihood
        except:
            return np.inf
    
    def _get_normality_recommendation(self, tests_passed: int, total_tests: int) -> str:
        """Generar recomendación basada en tests de normalidad."""
        score = tests_passed / total_tests if total_tests > 0 else 0
        
        if score >= 0.8:
            return "Distribución normal - Usar métodos paramétricos"
        elif score >= 0.5:
            return "Normalidad cuestionable - Considerar transformación o métodos robustos"
        else:
            return "Distribución no normal - Usar métodos no paramétricos"
    
    def _assess_symmetry(self, mean: float, median: float) -> str:
        """Evaluar simetría basada en media y mediana."""
        diff = abs(mean - median)
        ratio = diff / abs(median) if median != 0 else np.inf
        
        if ratio < 0.05:
            return "Simétrica"
        elif mean > median:
            return "Asimetría positiva (cola derecha)"
        else:
            return "Asimetría negativa (cola izquierda)"
    
    def _tukey_biweight_location(self, data) -> float:
        """Calcular ubicación Tukey biweight (estimador robusto)."""
        try:
            median = np.median(data)
            mad = stats.median_abs_deviation(data)
            
            if mad == 0:
                return median
            
            u = (data - median) / (6 * mad)
            weights = np.where(np.abs(u) < 1, (1 - u**2)**2, 0)
            
            return np.sum(weights * data) / np.sum(weights) if np.sum(weights) > 0 else median
        except:
            return np.median(data)
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpretar valor de asimetría."""
        if abs(skewness) < 0.5:
            return "Aproximadamente simétrica"
        elif abs(skewness) < 1:
            return "Moderadamente asimétrica"
        else:
            return "Altamente asimétrica"
    
    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpretar valor de curtosis."""
        if abs(kurtosis) < 0.5:
            return "Mesocúrtica (normal)"
        elif kurtosis > 0.5:
            return "Leptocúrtica (colas pesadas)"
        else:
            return "Platocúrtica (colas ligeras)"
    
    def _classify_distribution_shape(self, skewness: float, kurtosis: float) -> str:
        """Clasificar forma de distribución."""
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "Aproximadamente normal"
        elif abs(skewness) < 0.5:
            return "Simétrica no normal"
        elif skewness > 0:
            return "Cola derecha dominante"
        else:
            return "Cola izquierda dominante"
    
    def _classify_symmetry(self, skewness: float) -> str:
        """Clasificar nivel de simetría."""
        abs_skew = abs(skewness)
        if abs_skew < 0.25:
            return "Muy simétrica"
        elif abs_skew < 0.5:
            return "Moderadamente simétrica"
        elif abs_skew < 1:
            return "Asimétrica"
        else:
            return "Muy asimétrica"
    
    def _classify_tails(self, kurtosis: float) -> str:
        """Clasificar comportamiento de colas."""
        if kurtosis > 2:
            return "Colas muy pesadas"
        elif kurtosis > 0.5:
            return "Colas pesadas"
        elif kurtosis > -0.5:
            return "Colas normales"
        else:
            return "Colas ligeras"
    
    def _create_statistical_visualizations(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Crear visualizaciones estadísticas."""
        stats_viz_dir = Path(self.config.visualizations_dir) / 'distributions'
        stats_viz_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Distribuciones de variables numéricas
            self._plot_distributions(data, stats_viz_dir)
            
            # 2. Q-Q plots para normalidad
            self._plot_qq_plots(data, stats_viz_dir)
            
            # 3. Análisis de asimetría y curtosis
            self._plot_skewness_kurtosis(results['skewness_kurtosis'], stats_viz_dir)
            
            # 4. Comparación de medidas de tendencia central
            self._plot_central_tendency_comparison(results['central_tendency'], stats_viz_dir)
            
        except Exception as e:
            logger.error(f"Error creando visualizaciones estadísticas: {str(e)}")
    
    def _plot_distributions(self, data: pd.DataFrame, output_dir: Path):
        """Crear gráficos de distribución."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return
        
        # Seleccionar hasta 12 columnas para visualizar
        cols_to_plot = numeric_data.columns[:12]
        
        n_cols = min(4, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if n_rows * n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(cols_to_plot):
            ax = axes[i] if len(cols_to_plot) > 1 else axes[0]
            
            series = numeric_data[col].dropna()
            if len(series) == 0:
                continue
                
            # Histograma con curva de densidad
            ax.hist(series, bins=30, density=True, alpha=0.7, 
                   color=self.config.color_palette['primary'])
            
            # Agregar curva de densidad estimada
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(series)
                x_range = np.linspace(series.min(), series.max(), 100)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            except:
                pass
            
            ax.set_title(f'Distribución de {col}', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Densidad')
            ax.legend()
        
        # Ocultar ejes vacíos
        for i in range(len(cols_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'distributions_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_qq_plots(self, data: pd.DataFrame, output_dir: Path):
        """Crear Q-Q plots para evaluar normalidad."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return
        
        cols_to_plot = numeric_data.columns[:9]  # 3x3 grid
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(cols_to_plot):
            if i >= 9:
                break
                
            ax = axes[i]
            series = numeric_data[col].dropna()
            
            if len(series) < 3:
                ax.set_visible(False)
                continue
            
            # Q-Q plot
            stats.probplot(series, dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot: {col}', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Ocultar ejes vacíos
        for i in range(len(cols_to_plot), 9):
            axes[i].set_visible(False)
        
        plt.suptitle('Q-Q Plots para Evaluación de Normalidad', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'qq_plots_normality.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_skewness_kurtosis(self, skew_kurt_results: Dict[str, Any], output_dir: Path):
        """Crear gráfico de asimetría vs curtosis."""
        if not skew_kurt_results:
            return
        
        skewness_data = skew_kurt_results.get('skewness_analysis', {})
        kurtosis_data = skew_kurt_results.get('kurtosis_analysis', {})
        
        if not skewness_data or not kurtosis_data:
            return
        
        # Extraer valores
        columns = []
        skewness_values = []
        kurtosis_values = []
        
        for col in skewness_data.keys():
            if col in kurtosis_data:
                columns.append(col)
                skewness_values.append(skewness_data[col]['skewness'])
                kurtosis_values.append(kurtosis_data[col]['kurtosis'])
        
        if not columns:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        scatter = ax.scatter(skewness_values, kurtosis_values, 
                           s=100, alpha=0.7, c=self.config.color_palette['primary'])
        
        # Añadir líneas de referencia
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Curtosis normal')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Simetría')
        
        # Etiquetas
        for i, col in enumerate(columns):
            ax.annotate(col, (skewness_values[i], kurtosis_values[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Asimetría (Skewness)')
        ax.set_ylabel('Curtosis (Kurtosis)')
        ax.set_title('Análisis de Forma: Asimetría vs Curtosis', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'skewness_kurtosis_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_central_tendency_comparison(self, central_tendency_results: Dict[str, Any], output_dir: Path):
        """Crear comparación de medidas de tendencia central."""
        measures_comparison = central_tendency_results.get('measures_comparison', {})
        
        if not measures_comparison:
            return
        
        # Preparar datos
        columns = list(measures_comparison.keys())[:10]  # Limitar a 10 columnas
        means = []
        medians = []
        trimmed_means = []
        
        for col in columns:
            data = measures_comparison[col]
            means.append(data.get('mean', 0))
            medians.append(data.get('median', 0))
            trimmed_means.append(data.get('trimmed_mean_5pct', 0))
        
        if not columns:
            return
        
        # Crear gráfico
        x = np.arange(len(columns))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, means, width, label='Media', 
                      color=self.config.color_palette['primary'], alpha=0.8)
        bars2 = ax.bar(x, medians, width, label='Mediana',
                      color=self.config.color_palette['secondary'], alpha=0.8)
        bars3 = ax.bar(x + width, trimmed_means, width, label='Media Recortada 5%',
                      color=self.config.color_palette['accent'], alpha=0.8)
        
        ax.set_xlabel('Variables')
        ax.set_ylabel('Valores')
        ax.set_title('Comparación de Medidas de Tendencia Central', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(columns, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'central_tendency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()