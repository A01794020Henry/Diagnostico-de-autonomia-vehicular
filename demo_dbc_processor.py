"""
Script de demostración para el procesador DBC
Muestra cómo usar la clase DBCProcessor para extraer información de archivos DBC
y exportarla a Excel

Autor: GitHub Copilot
Fecha: 24 de septiembre de 2025
"""

import os
import sys
from dbc_processor_xlsx import process_dbc_to_excel, DBCProcessor
import logging

def main():
    """
    Función principal del script de demostración
    """
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dbc_processing.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    print("=" * 60)
    print("PROCESADOR DE ARCHIVOS DBC A EXCEL")
    print("=" * 60)
    
    # Ejemplo 1: Procesamiento automático
    print("\n1. EJEMPLO DE USO BÁSICO")
    print("-" * 40)
    
    # Definir rutas de archivos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dbc_file = os.path.join(current_dir, "Procesamiento_H2", "Procesamiento.DBC")
    excel_output = os.path.join(current_dir, "resultado_procesamiento_dbc.xlsx")
    
    print(f"Archivo DBC: {dbc_file}")
    print(f"Archivo Excel de salida: {excel_output}")
    
    # Verificar si el archivo DBC existe
    if not os.path.exists(dbc_file):
        print(f"\n⚠️  Archivo DBC no encontrado: {dbc_file}")
        
        # Crear archivo DBC de ejemplo si no existe
        print("Creando archivo DBC de ejemplo...")
        create_sample_dbc_file(dbc_file)
    
    # Procesar archivo DBC
    if process_dbc_to_excel(dbc_file, excel_output):
        print(f"\n✅ Procesamiento completado exitosamente!")
        print(f"📊 Archivo Excel generado: {excel_output}")
    else:
        print(f"\n❌ Error durante el procesamiento")
        return False
    
    # Ejemplo 2: Uso detallado de la clase DBCProcessor
    print("\n\n2. EJEMPLO DE USO DETALLADO")
    print("-" * 40)
    
    processor = DBCProcessor(dbc_file)
    
    if processor.process_dbc():
        print("\nInformación extraída del archivo DBC:")
        print(f"📄 Archivo: {os.path.basename(dbc_file)}")
        print(f"🔹 Nodos encontrados: {len(processor.nodes)}")
        print(f"📨 Mensajes encontrados: {len(processor.messages)}")
        print(f"📡 Señales encontradas: {len(processor.signals)}")
        print(f"📋 Tablas de valores: {len(processor.value_tables)}")
        print(f"⚙️ Atributos: {len(processor.attributes)}")
        print(f"💬 Comentarios: {len(processor.comments)}")
        
        # Mostrar algunos ejemplos de datos
        if processor.messages:
            print("\n📨 MENSAJES (primeros 5):")
            for i, (msg_id, msg_info) in enumerate(processor.messages.items()):
                if i >= 5:
                    break
                print(f"  • ID: {msg_id} (0x{msg_id:X}) - {msg_info['name']} - {len(msg_info['signals'])} señales")
        
        if processor.signals:
            print("\n📡 SEÑALES (primeras 5):")
            for i, (signal_key, signal_info) in enumerate(processor.signals.items()):
                if i >= 5:
                    break
                print(f"  • {signal_info['name']} - Msg: {signal_info['message_name']} - Unidad: {signal_info['unit']}")
        
        # Obtener DataFrames
        messages_df = processor.get_messages_dataframe()
        signals_df = processor.get_signals_dataframe()
        
        print(f"\n📊 DataFrame de mensajes: {len(messages_df)} filas")
        print(f"📊 DataFrame de señales: {len(signals_df)} filas")
        
    else:
        print("❌ Error al procesar el archivo DBC")
        return False
    
    print("\n" + "=" * 60)
    print("PROCESAMIENTO COMPLETADO")
    print("=" * 60)
    
    return True


def create_sample_dbc_file(dbc_file_path: str):
    """
    Crea un archivo DBC de ejemplo para demostración
    
    Args:
        dbc_file_path (str): Ruta donde crear el archivo DBC
    """
    sample_dbc_content = '''VERSION ""


NS_ :
	NS_DESC_
	CM_
	BA_DEF_
	BA_
	VAL_
	CAT_DEF_
	CAT_
	FILTER
	BA_DEF_DEF_
	EV_DATA_
	ENVVAR_DATA_
	SGTYPE_
	SGTYPE_VAL_
	BA_DEF_SGTYPE_
	BA_SGTYPE_
	SIG_VALTYPE_
	SIGTYPE_VALTYPE_
	BO_TX_BU_
	BA_DEF_REL_
	BA_REL_
	BA_DEF_DEF_REL_
	BU_SG_REL_
	BU_EV_REL_
	BU_BO_REL_
	SG_MUL_VAL_

BS_: Gateway ECU_Motor ECU_Frenos ECU_Transmision

BU_: Gateway ECU_Motor ECU_Frenos ECU_Transmision


BO_ 512 Motor_Status: 8 ECU_Motor
 SG_ RPM : 0|16@1+ (1,0) [0|8000] "rpm" Gateway,ECU_Frenos
 SG_ Temperatura_Motor : 16|8@1+ (1,-40) [-40|150] "°C" Gateway
 SG_ Presion_Aceite : 24|12@1+ (0.1,0) [0|10] "bar" Gateway
 SG_ Estado_Motor : 36|2@1+ (1,0) [0|3] "" Gateway

BO_ 513 Velocidad_Vehiculo: 4 ECU_Transmision
 SG_ Velocidad : 0|16@1+ (0.1,0) [0|300] "km/h" Gateway,ECU_Frenos
 SG_ Marcha_Actual : 16|4@1+ (1,0) [0|8] "" Gateway

BO_ 514 Sistema_Frenos: 6 ECU_Frenos
 SG_ Presion_Freno_Delantero : 0|12@1+ (0.1,0) [0|200] "bar" Gateway
 SG_ Presion_Freno_Trasero : 12|12@1+ (0.1,0) [0|200] "bar" Gateway
 SG_ Estado_ABS : 24|1@1+ (1,0) [0|1] "" Gateway
 SG_ Estado_ESP : 25|1@1+ (1,0) [0|1] "" Gateway

BO_ 768 Sensores_Ambiente: 8 Gateway
 SG_ Temperatura_Exterior : 0|8@1+ (1,-40) [-40|80] "°C" ECU_Motor,ECU_Frenos
 SG_ Humedad : 8|8@1+ (0.5,0) [0|100] "%" ECU_Motor
 SG_ Presion_Atmosferica : 16|16@1+ (0.1,0) [800|1200] "hPa" ECU_Motor


BA_DEF_ "BusType" STRING ;
BA_DEF_ "VFrameFormat" ENUM  "StandardCAN","ExtendedCAN","reserved","J1939PG","reserved","reserved","reserved","reserved","reserved","reserved","reserved","reserved","reserved","reserved","StandardCAN_FD","ExtendedCAN_FD";
BA_DEF_ "GenMsgCycleTime" INT 0 3600000;
BA_DEF_ BO_ "GenMsgDelayTime" INT 0 1000;

BA_DEF_DEF_ "BusType" "CAN";
BA_DEF_DEF_ "VFrameFormat" "StandardCAN";
BA_DEF_DEF_ "GenMsgCycleTime" 0;
BA_DEF_DEF_ "GenMsgDelayTime" 0;

BA_ "GenMsgCycleTime" BO_ 512 100;
BA_ "GenMsgCycleTime" BO_ 513 50;
BA_ "GenMsgCycleTime" BO_ 514 20;
BA_ "GenMsgCycleTime" BO_ 768 1000;

VAL_ 512 Estado_Motor 0 "Apagado" 1 "Ralenti" 2 "Normal" 3 "Alta_Carga" ;
VAL_ 513 Marcha_Actual 0 "Parking" 1 "Reversa" 2 "Neutral" 3 "Drive_1" 4 "Drive_2" 5 "Drive_3" 6 "Drive_4" 7 "Sport" ;
VAL_ 514 Estado_ABS 0 "Inactivo" 1 "Activo" ;
VAL_ 514 Estado_ESP 0 "Inactivo" 1 "Activo" ;

CM_ BO_ 512 "Mensaje con información del estado del motor";
CM_ BO_ 513 "Información de velocidad y transmisión";
CM_ BO_ 514 "Estado del sistema de frenos y seguridad";
CM_ BO_ 768 "Sensores ambientales del vehículo";
CM_ SG_ 512 RPM "Revoluciones por minuto del motor";
CM_ SG_ 512 Temperatura_Motor "Temperatura del refrigerante del motor";
CM_ SG_ 513 Velocidad "Velocidad actual del vehículo";
'''
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(dbc_file_path), exist_ok=True)
    
    # Escribir archivo DBC de ejemplo
    with open(dbc_file_path, 'w', encoding='utf-8') as f:
        f.write(sample_dbc_content)
    
    print(f"✅ Archivo DBC de ejemplo creado: {dbc_file_path}")


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 ¡Demostración completada exitosamente!")
        input("\nPresiona Enter para continuar...")
    else:
        print("\n⚠️ La demostración terminó con errores")
        input("\nPresiona Enter para continuar...")
        sys.exit(1)