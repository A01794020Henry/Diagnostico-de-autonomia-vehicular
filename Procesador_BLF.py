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




