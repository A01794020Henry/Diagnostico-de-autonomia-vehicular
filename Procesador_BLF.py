"""
ARCHIVO ACTUALIZADO - Usar main_blf_processor.py
===============================================

Este archivo ha sido reemplazado por una versión más completa y robusta.

Para usar el nuevo sistema:
1. Interfaz gráfica: python main_blf_processor.py
2. Línea de comandos: python main_blf_processor.py --cli --help
3. Ejemplo de uso: python ejemplo_uso.py

Nuevas características:
- Procesamiento de múltiples archivos BLF
- Unificación y ordenamiento cronológico automático  
- Decodificación robusta con archivos DBC
- Interfaz gráfica interactiva con filtros
- Exportación de datos y gráficos
- Manejo robusto de errores

Ver README.md para documentación completa.
"""

print("⚠️  Este archivo ha sido actualizado.")
print("🚀 Usa 'python main_blf_processor.py' para el nuevo sistema completo.")
print("📖 Ver README.md para documentación detallada.")

# Código original conservado para referencia:
import cantools
import can
import pandas as pd
import matplotlib
import pyqtgraph

# Carga del DBC disponible 
# db = cantools.database.load_file(r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Desarrollo de Software EV\Datos\DBC\IP_JZ - CAN EV.DBC")

# Usar el nuevo sistema en su lugar:
if __name__ == "__main__":
    import os
    print("\n" + "="*60)
    print("EJECUTANDO NUEVO SISTEMA...")
    print("="*60)
    
    # Intentar ejecutar el nuevo sistema
    try:
        from main_blf_processor import main
        main()
    except ImportError:
        print("Para usar el nuevo sistema, ejecuta:")
        print("python main_blf_processor.py")
    except Exception as e:
        print(f"Error: {e}")
        print("Ejecuta manualmente: python main_blf_processor.py")



    





