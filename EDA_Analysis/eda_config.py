"""
Configuración para el Análisis Exploratorio de Datos (EDA)
=====================================================

Definiciones de configuración, estilos y parámetros para el análisis.

Autor: Sistema de diagnóstico de autonomía vehicular
Fecha: Septiembre 2025
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class EDAConfig:
    """Configuración principal para el análisis EDA."""
    
    def __init__(self, output_dir="./EDA_output"):
        """Inicializar configuración."""
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "reports"
        self.visualizations_dir = self.output_dir / "visualizations"
        
        # Configuración de visualización
        self.setup_visualization_style()
        
        # Parámetros de análisis
        self.outlier_methods = [
            'isolation_forest',
            'local_outlier_factor', 
            'one_class_svm',
            'statistical_z_score'
        ]
        
        self.correlation_methods = ['pearson', 'spearman', 'kendall']
        
        # Umbrales
        self.missing_threshold = 0.5  # 50% de valores faltantes
        self.high_cardinality_threshold = 50  # Variables categóricas
        self.correlation_threshold = 0.7  # Correlación alta
        self.outlier_contamination = 0.1  # 10% de outliers esperados
        
        # Configuración temporal
        self.temporal_frequency_detection = True
        self.seasonal_decomposition = True
        
        # Variables específicas del dominio vehicular
        self.vehicle_signal_categories = {
            'battery': ['voltage', 'current', 'soc', 'temperature', 'battery'],
            'motor': ['speed', 'torque', 'temperature', 'motor', 'rpm'],
            'position': ['latitude', 'longitude', 'altitude', 'gps'],
            'can': ['arbitration_id', 'message_name', 'signal_name'],
            'temporal': ['timestamp', 'datetime', 'time']
        }
    
    def setup_visualization_style(self):
        """Configurar estilo visual para gráficos."""
        # Estilo base
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Paleta de colores para vehículos eléctricos
        self.color_palette = {
            'primary': '#2E8B57',      # Verde mar (sustentabilidad)
            'secondary': '#1E90FF',    # Azul dodger (tecnología)
            'accent': '#FF6B35',       # Naranja (energía)
            'warning': '#FFD700',      # Dorado (advertencia)
            'error': '#DC143C',        # Rojo carmesí (error)
            'success': '#228B22',      # Verde bosque (éxito)
            'neutral': '#708090'       # Gris pizarra (neutral)
        }
        
        # Lista de colores vehiculares para gráficos
        self.vehicle_colors = [
            self.color_palette['primary'],    # Verde mar
            self.color_palette['secondary'],  # Azul dodger
            self.color_palette['accent'],     # Naranja
            self.color_palette['success'],    # Verde bosque
            self.color_palette['warning'],    # Dorado
            self.color_palette['error'],      # Rojo carmesí
            self.color_palette['neutral']     # Gris pizarra
        ]
        
        # Configuración de matplotlib
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        # Configuración de seaborn
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style("whitegrid")
    
    def get_signal_category(self, signal_name):
        """Determinar categoría de una señal vehicular."""
        signal_lower = signal_name.lower()
        
        for category, keywords in self.vehicle_signal_categories.items():
            if any(keyword in signal_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def get_category_color(self, category):
        """Obtener color para una categoría específica."""
        category_colors = {
            'battery': self.color_palette['success'],
            'motor': self.color_palette['primary'],
            'position': self.color_palette['secondary'],
            'can': self.color_palette['neutral'],
            'temporal': self.color_palette['accent'],
            'other': self.color_palette['neutral']
        }
        
        return category_colors.get(category, self.color_palette['neutral'])


class AnalysisParameters:
    """Parámetros específicos para diferentes tipos de análisis."""
    
    # Parámetros para detección de anomalías
    OUTLIER_PARAMS = {
        'isolation_forest': {
            'contamination': 0.1,
            'random_state': 42,
            'n_estimators': 100
        },
        'local_outlier_factor': {
            'contamination': 0.1,
            'n_neighbors': 20
        },
        'one_class_svm': {
            'contamination': 0.1,
            'kernel': 'rbf',
            'gamma': 'scale'
        }
    }
    
    # Parámetros para análisis estadístico
    STATISTICAL_PARAMS = {
        'confidence_level': 0.95,
        'normality_test': 'shapiro',
        'homoscedasticity_test': 'levene',
        'max_categories_for_chi2': 10
    }
    
    # Parámetros para análisis temporal
    TEMPORAL_PARAMS = {
        'seasonal_periods': [24, 168, 8760],  # Horas, semana, año
        'trend_detection': True,
        'seasonality_detection': True,
        'anomaly_detection': True
    }
    
    # Parámetros para visualización
    VISUALIZATION_PARAMS = {
        'max_categories_in_plot': 20,
        'histogram_bins': 50,
        'scatter_sample_size': 10000,
        'heatmap_annot': True
    }


class ReportTemplates:
    """Templates para generación de reportes."""
    
    @staticmethod
    def get_html_header():
        """Header HTML para reportes."""
        return """
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Análisis EDA - Sistema Diagnóstico Vehicular</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }
                .header {
                    text-align: center;
                    border-bottom: 3px solid #2E8B57;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }
                .section {
                    margin-bottom: 40px;
                }
                .metric-card {
                    display: inline-block;
                    margin: 10px;
                    padding: 20px;
                    background: #f8f9fa;
                    border: 2px solid #e9ecef;
                    border-radius: 8px;
                    min-width: 200px;
                    text-align: center;
                }
                .metric-value {
                    font-size: 2em;
                    font-weight: bold;
                    color: #2E8B57;
                }
                .metric-label {
                    color: #6c757d;
                    font-size: 0.9em;
                }
                .alert {
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                .alert-warning {
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                    color: #856404;
                }
                .alert-error {
                    background-color: #f8d7da;
                    border: 1px solid #f5c6cb;
                    color: #721c24;
                }
                .alert-success {
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    color: #155724;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }
                th {
                    background-color: #2E8B57;
                    color: white;
                }
                .code {
                    background-color: #f8f9fa;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }
            </style>
        </head>
        <body>
            <div class="container">
        """
    
    @staticmethod
    def get_html_footer():
        """Footer HTML para reportes."""
        return """
            </div>
        </body>
        </html>
        """
