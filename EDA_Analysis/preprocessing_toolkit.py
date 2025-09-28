"""
Herramientas de Preprocesamiento
================================

Módulo con herramientas para preprocesamiento de datos basado en
los resultados del análisis exploratorio.

Autor: Sistema de diagnóstico de autonomía vehicular
Fecha: Septiembre 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import boxcox
from scipy.special import inv_boxcox

logger = logging.getLogger(__name__)


class PreprocessingToolkit:
    """Herramientas de preprocesamiento para datos vehiculares."""
    
    def __init__(self, config):
        """Inicializar toolkit de preprocesamiento."""
        self.config = config
        self.preprocessing_history = []
        self.fitted_transformers = {}
    
    def generate_preprocessing_recommendations(self, eda_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generar recomendaciones de preprocesamiento basadas en resultados EDA.
        
        Args:
            eda_results: Resultados del análisis EDA
            
        Returns:
            Diccionario con recomendaciones de preprocesamiento
        """
        logger.info("Generando recomendaciones de preprocesamiento")
        
        recommendations = {
            'missing_values': self._recommend_missing_value_handling(eda_results),
            'outliers': self._recommend_outlier_handling(eda_results),
            'scaling': self._recommend_scaling_methods(eda_results),
            'encoding': self._recommend_encoding_methods(eda_results),
            'transformations': self._recommend_transformations(eda_results),
            'feature_selection': self._recommend_feature_selection(eda_results)
        }
        
        return recommendations
    
    def apply_preprocessing(self, data: pd.DataFrame, 
                          preprocessing_plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Aplicar plan de preprocesamiento a los datos.
        
        Args:
            data: DataFrame original
            preprocessing_plan: Plan de preprocesamiento a aplicar
            
        Returns:
            Tupla con (datos_procesados, reporte_procesamiento)
        """
        logger.info("Aplicando plan de preprocesamiento")
        
        processed_data = data.copy()
        processing_report = {
            'steps_applied': [],
            'transformations': {},
            'warnings': [],
            'original_shape': data.shape,
            'final_shape': None
        }
        
        try:
            # 1. Manejo de valores faltantes
            if 'missing_values' in preprocessing_plan:
                processed_data, step_report = self._handle_missing_values(
                    processed_data, preprocessing_plan['missing_values']
                )
                processing_report['steps_applied'].append('missing_values')
                processing_report['transformations']['missing_values'] = step_report
            
            # 2. Manejo de valores atípicos
            if 'outliers' in preprocessing_plan:
                processed_data, step_report = self._handle_outliers(
                    processed_data, preprocessing_plan['outliers']
                )
                processing_report['steps_applied'].append('outliers')
                processing_report['transformations']['outliers'] = step_report
            
            # 3. Transformaciones
            if 'transformations' in preprocessing_plan:
                processed_data, step_report = self._apply_transformations(
                    processed_data, preprocessing_plan['transformations']
                )
                processing_report['steps_applied'].append('transformations')
                processing_report['transformations']['transformations'] = step_report
            
            # 4. Codificación de variables categóricas
            if 'encoding' in preprocessing_plan:
                processed_data, step_report = self._encode_categorical_variables(
                    processed_data, preprocessing_plan['encoding']
                )
                processing_report['steps_applied'].append('encoding')
                processing_report['transformations']['encoding'] = step_report
            
            # 5. Escalado
            if 'scaling' in preprocessing_plan:
                processed_data, step_report = self._scale_features(
                    processed_data, preprocessing_plan['scaling']
                )
                processing_report['steps_applied'].append('scaling')
                processing_report['transformations']['scaling'] = step_report
            
            processing_report['final_shape'] = processed_data.shape
            logger.info(f"Preprocesamiento completado: {data.shape} → {processed_data.shape}")
            
        except Exception as e:
            logger.error(f"Error durante preprocesamiento: {str(e)}")
            processing_report['error'] = str(e)
        
        return processed_data, processing_report
    
    def _recommend_missing_value_handling(self, eda_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recomendar estrategias para manejo de valores faltantes."""
        quality_results = eda_results.get('data_quality', {})
        missing_info = quality_results.get('missing_values', {})
        
        if not missing_info or missing_info.get('missing_percentage', 0) == 0:
            return {'strategy': 'no_action', 'reason': 'No hay valores faltantes'}
        
        columns_with_missing = missing_info.get('columns_with_missing', {})
        recommendations = {}
        
        for column, col_info in columns_with_missing.items():
            missing_pct = col_info['percentage']
            
            if missing_pct > 50:
                recommendations[column] = {
                    'strategy': 'drop_column',
                    'reason': f'Más del 50% de valores faltantes ({missing_pct:.1f}%)'
                }
            elif missing_pct > 20:
                recommendations[column] = {
                    'strategy': 'advanced_imputation',
                    'method': 'knn',
                    'reason': f'Alto porcentaje de faltantes ({missing_pct:.1f}%), usar KNN'
                }
            elif missing_pct > 5:
                recommendations[column] = {
                    'strategy': 'statistical_imputation',
                    'method': 'median',  # Para datos vehiculares, mediana es más robusta
                    'reason': f'Porcentaje moderado ({missing_pct:.1f}%), usar mediana'
                }
            else:
                recommendations[column] = {
                    'strategy': 'simple_imputation',
                    'method': 'mean',
                    'reason': f'Bajo porcentaje ({missing_pct:.1f}%), usar media'
                }
        
        return {
            'overall_strategy': 'column_specific',
            'column_recommendations': recommendations,
            'total_missing_percentage': missing_info.get('missing_percentage', 0)
        }
    
    def _recommend_outlier_handling(self, eda_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recomendar estrategias para manejo de valores atípicos."""
        outlier_results = eda_results.get('outliers', {})
        
        if 'error' in outlier_results:
            return {'strategy': 'no_action', 'reason': 'No se pudo analizar outliers'}
        
        outlier_summary = outlier_results.get('outlier_summary', {})
        consensus_outliers = outlier_summary.get('consensus_outliers', [])
        
        if not consensus_outliers:
            return {'strategy': 'no_action', 'reason': 'No se detectaron outliers de consenso'}
        
        total_outliers = len(consensus_outliers)
        outlier_percentage = outlier_summary.get('method_agreement', {}).get('consensus_outliers_count', 0)
        
        if outlier_percentage > 10:
            return {
                'strategy': 'cap_outliers',
                'method': 'iqr_capping',
                'reason': f'Alto porcentaje de outliers ({outlier_percentage}%), usar capping'
            }
        elif outlier_percentage > 5:
            return {
                'strategy': 'transform_outliers',
                'method': 'log_transform',
                'reason': f'Porcentaje moderado de outliers ({outlier_percentage}%), usar transformación'
            }
        else:
            return {
                'strategy': 'remove_outliers',
                'outlier_indices': consensus_outliers,
                'reason': f'Bajo porcentaje de outliers ({outlier_percentage}%), remover'
            }
    
    def _recommend_scaling_methods(self, eda_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recomendar métodos de escalado."""
        stats_results = eda_results.get('statistics', {})
        
        if 'error' in stats_results:
            return {'strategy': 'standard_scaling', 'reason': 'Método por defecto'}
        
        # Analizar distribuciones para recomendar escalado apropiado
        normality_results = stats_results.get('normality_tests', {})
        skewness_results = stats_results.get('skewness_kurtosis', {}).get('skewness_analysis', {})
        
        recommendations = {}
        
        # Si hay muchas variables con distribuciones normales, usar StandardScaler
        normal_variables = 0
        skewed_variables = 0
        total_variables = 0
        
        for var, normality_info in normality_results.get('normalcy_summary', {}).items():
            total_variables += 1
            if normality_info.get('normalcy_score', 0) > 0.5:
                normal_variables += 1
        
        for var, skew_info in skewness_results.items():
            if abs(skew_info.get('skewness', 0)) > 1:
                skewed_variables += 1
        
        if total_variables == 0:
            return {'strategy': 'standard_scaling', 'reason': 'Método por defecto'}
        
        normal_ratio = normal_variables / total_variables
        skewed_ratio = skewed_variables / total_variables
        
        if normal_ratio > 0.7:
            return {
                'strategy': 'standard_scaling',
                'method': 'StandardScaler',
                'reason': f'{normal_ratio:.1%} de variables con distribución normal'
            }
        elif skewed_ratio > 0.5:
            return {
                'strategy': 'robust_scaling',
                'method': 'RobustScaler',
                'reason': f'{skewed_ratio:.1%} de variables con alta asimetría'
            }
        else:
            return {
                'strategy': 'minmax_scaling',
                'method': 'MinMaxScaler',
                'reason': 'Distribuciones mixtas, usar escalado min-max'
            }
    
    def _recommend_encoding_methods(self, eda_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recomendar métodos de codificación para variables categóricas."""
        quality_results = eda_results.get('data_quality', {})
        cardinality_info = quality_results.get('cardinality', {})
        
        high_cardinality = cardinality_info.get('high_cardinality_columns', {})
        low_cardinality = cardinality_info.get('low_cardinality_columns', {})
        
        recommendations = {}
        
        # Variables de baja cardinalidad -> One-Hot Encoding
        for column, info in low_cardinality.items():
            unique_count = info['unique_count']
            if unique_count <= 5:
                recommendations[column] = {
                    'method': 'onehot_encoding',
                    'reason': f'Baja cardinalidad ({unique_count} valores únicos)'
                }
            elif unique_count <= 10:
                recommendations[column] = {
                    'method': 'label_encoding',
                    'reason': f'Cardinalidad moderada ({unique_count} valores únicos)'
                }
        
        # Variables de alta cardinalidad -> Estrategias especiales
        for column, info in high_cardinality.items():
            unique_count = info['unique_count']
            if unique_count > 100:
                recommendations[column] = {
                    'method': 'frequency_encoding',
                    'reason': f'Muy alta cardinalidad ({unique_count} valores únicos)'
                }
            else:
                recommendations[column] = {
                    'method': 'target_encoding',
                    'reason': f'Alta cardinalidad ({unique_count} valores únicos)'
                }
        
        return {
            'column_recommendations': recommendations,
            'default_strategy': 'label_encoding'
        }
    
    def _recommend_transformations(self, eda_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recomendar transformaciones de variables."""
        stats_results = eda_results.get('statistics', {})
        skewness_results = stats_results.get('skewness_kurtosis', {}).get('skewness_analysis', {})
        
        recommendations = {}
        
        for variable, skew_info in skewness_results.items():
            skewness = skew_info.get('skewness', 0)
            
            if abs(skewness) > 2:
                if skewness > 0:
                    recommendations[variable] = {
                        'transformation': 'log_transform',
                        'reason': f'Muy asimétrica positiva (skewness: {skewness:.2f})'
                    }
                else:
                    recommendations[variable] = {
                        'transformation': 'square_transform',
                        'reason': f'Muy asimétrica negativa (skewness: {skewness:.2f})'
                    }
            elif abs(skewness) > 1:
                recommendations[variable] = {
                    'transformation': 'sqrt_transform',
                    'reason': f'Moderadamente asimétrica (skewness: {skewness:.2f})'
                }
        
        return {
            'column_recommendations': recommendations,
            'general_strategy': 'normalize_distributions'
        }
    
    def _recommend_feature_selection(self, eda_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recomendar estrategias de selección de características."""
        correlation_results = eda_results.get('correlations', {})
        high_correlations = correlation_results.get('high_correlations', {}).get('high_correlations', [])
        
        recommendations = {
            'multicollinearity_removal': [],
            'low_variance_removal': [],
            'feature_importance_analysis': True
        }
        
        # Identificar variables altamente correlacionadas para remoción
        seen_pairs = set()
        for corr_pair in high_correlations:
            var1 = corr_pair['variable_1']
            var2 = corr_pair['variable_2']
            correlation = corr_pair['correlation']
            
            pair_key = tuple(sorted([var1, var2]))
            if pair_key not in seen_pairs and abs(correlation) > 0.9:
                recommendations['multicollinearity_removal'].append({
                    'variables': [var1, var2],
                    'correlation': correlation,
                    'recommendation': f'Remover una de las variables (correlación: {correlation:.3f})'
                })
                seen_pairs.add(pair_key)
        
        return recommendations
    
    def _handle_missing_values(self, data: pd.DataFrame, 
                              missing_strategy: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Manejar valores faltantes según estrategia."""
        processed_data = data.copy()
        report = {'columns_processed': [], 'methods_used': {}, 'rows_affected': 0}
        
        column_recommendations = missing_strategy.get('column_recommendations', {})
        
        for column, strategy_info in column_recommendations.items():
            if column not in processed_data.columns:
                continue
                
            strategy = strategy_info['strategy']
            
            if strategy == 'drop_column':
                processed_data = processed_data.drop(columns=[column])
                report['columns_processed'].append(column)
                report['methods_used'][column] = 'column_dropped'
                
            elif strategy == 'simple_imputation':
                method = strategy_info.get('method', 'mean')
                if pd.api.types.is_numeric_dtype(processed_data[column]):
                    if method == 'mean':
                        fill_value = processed_data[column].mean()
                    elif method == 'median':
                        fill_value = processed_data[column].median()
                    else:
                        fill_value = processed_data[column].mode()[0] if not processed_data[column].mode().empty else 0
                    
                    missing_count = processed_data[column].isnull().sum()
                    processed_data[column] = processed_data[column].fillna(fill_value)
                    
                    report['columns_processed'].append(column)
                    report['methods_used'][column] = f'simple_imputation_{method}'
                    report['rows_affected'] += missing_count
                    
            elif strategy == 'advanced_imputation':
                method = strategy_info.get('method', 'knn')
                
                if method == 'knn':
                    # Usar solo columnas numéricas para KNN
                    numeric_data = processed_data.select_dtypes(include=[np.number])
                    
                    if column in numeric_data.columns and len(numeric_data.columns) > 1:
                        imputer = KNNImputer(n_neighbors=5)
                        
                        # Ajustar y transformar
                        imputed_values = imputer.fit_transform(numeric_data)
                        
                        # Reemplazar solo la columna específica
                        col_index = numeric_data.columns.get_loc(column)
                        missing_count = processed_data[column].isnull().sum()
                        processed_data[column] = imputed_values[:, col_index]
                        
                        report['columns_processed'].append(column)
                        report['methods_used'][column] = 'knn_imputation'
                        report['rows_affected'] += missing_count
        
        return processed_data, report
    
    def _handle_outliers(self, data: pd.DataFrame, 
                        outlier_strategy: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Manejar valores atípicos según estrategia."""
        processed_data = data.copy()
        report = {'method_used': None, 'outliers_processed': 0}
        
        strategy = outlier_strategy.get('strategy', 'no_action')
        
        if strategy == 'remove_outliers':
            outlier_indices = outlier_strategy.get('outlier_indices', [])
            if outlier_indices:
                initial_rows = len(processed_data)
                processed_data = processed_data.drop(index=outlier_indices, errors='ignore')
                final_rows = len(processed_data)
                
                report['method_used'] = 'outlier_removal'
                report['outliers_processed'] = initial_rows - final_rows
                
        elif strategy == 'cap_outliers':
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                Q1 = processed_data[column].quantile(0.25)
                Q3 = processed_data[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Aplicar capping
                outliers_capped = ((processed_data[column] < lower_bound) | 
                                 (processed_data[column] > upper_bound)).sum()
                
                processed_data[column] = processed_data[column].clip(lower=lower_bound, upper=upper_bound)
                report['outliers_processed'] += outliers_capped
            
            report['method_used'] = 'iqr_capping'
        
        return processed_data, report
    
    def _apply_transformations(self, data: pd.DataFrame, 
                              transformation_strategy: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Aplicar transformaciones a variables."""
        processed_data = data.copy()
        report = {'transformations_applied': {}, 'columns_transformed': []}
        
        column_recommendations = transformation_strategy.get('column_recommendations', {})
        
        for column, transform_info in column_recommendations.items():
            if column not in processed_data.columns:
                continue
                
            if not pd.api.types.is_numeric_dtype(processed_data[column]):
                continue
                
            transformation = transform_info['transformation']
            
            try:
                if transformation == 'log_transform':
                    # Asegurar valores positivos
                    min_val = processed_data[column].min()
                    if min_val <= 0:
                        shift = abs(min_val) + 1
                        processed_data[column] = processed_data[column] + shift
                    
                    processed_data[column] = np.log1p(processed_data[column])
                    
                elif transformation == 'sqrt_transform':
                    # Asegurar valores no negativos
                    min_val = processed_data[column].min()
                    if min_val < 0:
                        processed_data[column] = processed_data[column] - min_val
                    
                    processed_data[column] = np.sqrt(processed_data[column])
                    
                elif transformation == 'square_transform':
                    processed_data[column] = np.square(processed_data[column])
                
                report['transformations_applied'][column] = transformation
                report['columns_transformed'].append(column)
                
            except Exception as e:
                logger.warning(f"Error aplicando transformación {transformation} a {column}: {str(e)}")
        
        return processed_data, report
    
    def _encode_categorical_variables(self, data: pd.DataFrame, 
                                     encoding_strategy: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Codificar variables categóricas."""
        processed_data = data.copy()
        report = {'encoding_methods': {}, 'new_columns': [], 'dropped_columns': []}
        
        column_recommendations = encoding_strategy.get('column_recommendations', {})
        
        for column, encoding_info in column_recommendations.items():
            if column not in processed_data.columns:
                continue
            
            method = encoding_info['method']
            
            try:
                if method == 'onehot_encoding':
                    # One-hot encoding
                    dummies = pd.get_dummies(processed_data[column], prefix=column)
                    processed_data = pd.concat([processed_data, dummies], axis=1)
                    processed_data = processed_data.drop(columns=[column])
                    
                    report['new_columns'].extend(dummies.columns.tolist())
                    report['dropped_columns'].append(column)
                    
                elif method == 'label_encoding':
                    # Label encoding
                    le = LabelEncoder()
                    processed_data[f'{column}_encoded'] = le.fit_transform(processed_data[column].astype(str))
                    processed_data = processed_data.drop(columns=[column])
                    
                    self.fitted_transformers[f'{column}_label_encoder'] = le
                    report['new_columns'].append(f'{column}_encoded')
                    report['dropped_columns'].append(column)
                    
                elif method == 'frequency_encoding':
                    # Frequency encoding
                    freq_map = processed_data[column].value_counts().to_dict()
                    processed_data[f'{column}_freq'] = processed_data[column].map(freq_map)
                    processed_data = processed_data.drop(columns=[column])
                    
                    report['new_columns'].append(f'{column}_freq')
                    report['dropped_columns'].append(column)
                
                report['encoding_methods'][column] = method
                
            except Exception as e:
                logger.warning(f"Error codificando {column} con {method}: {str(e)}")
        
        return processed_data, report
    
    def _scale_features(self, data: pd.DataFrame, 
                       scaling_strategy: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Escalar características numéricas."""
        processed_data = data.copy()
        report = {'scaling_method': None, 'columns_scaled': []}
        
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return processed_data, report
        
        method = scaling_strategy.get('method', 'StandardScaler')
        
        try:
            if method == 'StandardScaler':
                scaler = StandardScaler()
            elif method == 'MinMaxScaler':
                scaler = MinMaxScaler()
            elif method == 'RobustScaler':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()  # Default
            
            # Ajustar y transformar
            scaled_values = scaler.fit_transform(processed_data[numeric_columns])
            processed_data[numeric_columns] = scaled_values
            
            # Guardar el scaler ajustado
            self.fitted_transformers['scaler'] = scaler
            
            report['scaling_method'] = method
            report['columns_scaled'] = numeric_columns.tolist()
            
        except Exception as e:
            logger.error(f"Error aplicando escalado {method}: {str(e)}")
            report['error'] = str(e)
        
        return processed_data, report