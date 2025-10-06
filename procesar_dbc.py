"""
Script simple para procesar archivos DBC y generar Excel
Uso: python procesar_dbc.py [archivo_dbc] [archivo_excel_salida]

Autor: GitHub Copilot
Fecha: 24 de septiembre de 2025
"""

import sys
import os
from dbc_processor_xlsx import process_dbc_to_excel

def main():
    """
    Función principal para procesamiento simple de DBC a Excel
    """
    print("🔧 PROCESADOR SIMPLE DBC -> EXCEL")
    print("=" * 50)
    
    # Verificar argumentos
    if len(sys.argv) < 2:
        print("📋 USO:")
        print(f"   python {os.path.basename(__file__)} archivo.dbc [salida.xlsx]")
        print("\n📝 EJEMPLOS:")
        print(f"   python {os.path.basename(__file__)} mi_archivo.dbc")
        print(f"   python {os.path.basename(__file__)} mi_archivo.dbc resultado.xlsx")
        print("\nSi no especificas archivo de salida, se generará automáticamente.")
        return
    
    # Obtener rutas de archivos
    dbc_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        excel_file = sys.argv[2]
    else:
        # Generar nombre automáticamente
        base_name = os.path.splitext(os.path.basename(dbc_file))[0]
        excel_file = f"{base_name}_procesado.xlsx"
    
    # Verificar si el archivo DBC existe
    if not os.path.exists(dbc_file):
        print(f"❌ Error: Archivo no encontrado: {dbc_file}")
        return
    
    print(f"📄 Archivo DBC: {dbc_file}")
    print(f"📊 Archivo Excel: {excel_file}")
    print()
    
    # Procesar
    print("⚙️ Procesando archivo DBC...")
    
    if process_dbc_to_excel(dbc_file, excel_file):
        print(f"✅ ¡Procesamiento completado exitosamente!")
        print(f"📊 Archivo generado: {excel_file}")
        
        # Mostrar información del archivo generado
        if os.path.exists(excel_file):
            file_size = os.path.getsize(excel_file) / 1024  # KB
            print(f"📏 Tamaño del archivo: {file_size:.1f} KB")
    else:
        print("❌ Error durante el procesamiento")


if __name__ == "__main__":
    main()