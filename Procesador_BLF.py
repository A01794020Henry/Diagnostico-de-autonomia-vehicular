import cantools
import can
import pandas as pd
import matplotlib
import pyqtgraph

# Carga del DBC disponible 
db = cantools.database.load_file(r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Desarrollo de Software EV\Datos\DBC\IP_JZ - CAN EV.DBC")


#Lectura del archvio DBC
for mesaje in db.messages:
    print(mesaje.name)
    print(mesaje.frame_id)
    print([s.name] for s in mesaje.signals)
    
""" # Lectura del archivo BLF
for msg in can.BLFReader("ruta_al_blf"):
    print(msg.arbitration_id)  # El ID del mensaje
    print(msg.data) # Los datos crudos (bytes)
    print(msg.timestamp) # El tiempo del mensaje
    break # Solo se requiere el primer mensaje
    

# Decodificación de mensajes con DBC
valores = db.decode_message(msg.arbitration_id, mensaje.data)
print (valores) """

# Se debe calcular y sobreponer el consumo con respecto al avance en el tiempo



    





