"""
Script de prueba para el sistema EDA
===================================

Prueba rápida del sistema completo de análisis exploratorio.

Autor: Sistema de diagnóstico de autonomía vehicular
Fecha: Septiembre 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Agregar el directorio EDA_Analysis al path
eda_path = Path(__file__).parent / 'EDA_Analysis'
sys.path.append(str(eda_path))

try:
    # Importar módulos del sistema EDA
    from eda_config import EDAConfig
    from eda_main import EDAMainAnalyzer
    
    print("✅ Módulos EDA importados correctamente")
    
    # Crear datos de prueba
    np.random.seed(42)
    n_samples = 1000
    
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1S'),
        'vehicle_speed': np.random.normal(50, 15, n_samples).clip(0, 120),
        'engine_rpm': np.random.normal(2000, 500, n_samples).clip(600, 6000),
        'fuel_consumption': np.random.exponential(8, n_samples),
        'engine_temp': np.random.normal(90, 10, n_samples).clip(60, 120),
        'battery_voltage': np.random.normal(12.5, 0.5, n_samples).clip(11, 14)
    })
    
    # Introducir algunos valores faltantes
    test_data.loc[np.random.choice(test_data.index, 50), 'fuel_consumption'] = np.nan
    
    print(f"📊 Datos de prueba generados: {test_data.shape}")
    
    # Configurar sistema EDA
    config = EDAConfig()
    analyzer = EDAMainAnalyzer(config)
    
    print("🔧 Sistema EDA configurado")
    
    # Ejecutar análisis completo
    print("🚀 Iniciando análisis exploratorio...")
    results = analyzer.run_complete_analysis(test_data)
    
    print("\n📋 RESULTADOS DEL ANÁLISIS:")
    print("=" * 50)
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key}: {len(value)} elementos")
        elif isinstance(value, str):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {type(value).__name__}")
    
    print(f"\n✅ Análisis completado exitosamente")
    print(f"📁 Resultados guardados en: {config.output_dir}")
    print(f"📈 Visualizaciones en: {config.visualizations_dir}")
    
except Exception as e:
    print(f"❌ Error durante la prueba: {str(e)}")
    import traceback
    traceback.print_exc()