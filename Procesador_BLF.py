import cantools
import can
import pandas as pd
import matplotlib
import pyqtgraph

from pprint import pprint

# Carga del DBC disponible 
db = cantools.database.load_file(r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Desarrollo de Software EV\Datos\DBC\IP_JZ - CAN EV.DBC")

#Lectura del archvio DBC


