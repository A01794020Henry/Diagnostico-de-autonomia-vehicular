# Script para procesamiento de archivos .GPX para rutas de validación de autonomía

# Importación de librerias

import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import numpy as np
import folium
import pandas as pd


# Ruta del archivo GPX
gpx_file_path = "C:\\haranzales\\OneDrive - Superpolo S.A.S\\Ingenieria\\Gestion\\Logs\\17_sept_2025_7_17_52.gpx"

# Lectura de archivo .GPX
with open(gpx_file_path, 'r', encoding='utf-8') as gpx_file:
    gpx = gpxpy.parse(gpx_file)
    
#Extracción de datos de la ruta
latitudes= []
longitudes= []
altitudes= []
times= []

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000 # Radio de la Tierra en metros
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

distances = [0]

for track in gpx.tracks:
    for segment in track.segments:
        prev_point = None
        for point in segment.points:
            latitudes.append(point.latitude)
            longitudes.append(point.longitude)
            altitudes.append(point.elevation)
            times.append(point.time)
            if prev_point:
                d = haversine(prev_point.latitude, prev_point.longitude, point.latitude, point.longitude)
                distances.append(distances[-1] + d)
            prev_point = point
            
# Calcular el tiempo en segundos desde el inicio del recorrido
time_seconds = [(t - times[0]).total_seconds() for t in times]

# Inicio de la sección de graficas 
matplotlib.rcParams["font.family"] = "Century Gothic"

# Graficación de Altitud vs Distancia
plt.figure(figsize=(10,5))
plt.plot(np.array(distances)/1000, altitudes, label="Altitud")
# Calcular altitud promedio
altitud_promedio = np.mean(altitudes)
plt.axhline(altitud_promedio, color='red', linestyle='--', label=f'Altitud promedio: {altitud_promedio:.2f} m')
plt.xlabel("Distancia recorrida (km)")
plt.ylabel("Altitud del recorrido (m)")
plt.title("Comportamiento de la altitud con respecto a la distancia recorrida")
plt.grid()
plt.legend()
plt.savefig(r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Repositorios\Prueba de Autonomia EV_Prototipo\Distancia_vs_Altura.png")
plt.show()

# Gráfica de la Altiud y el Tiempo de desplazamiento
plt.figure(figsize=(10,5))
plt.plot(np.array(time_seconds)/60, altitudes, label="Altitud")
altitud_promedio = np.mean(altitudes)
plt.axhline(altitud_promedio, color="red", linestyle="--", label=f"Altitud promedio: {altitud_promedio:.2f} m")
plt.xlabel('Tiempo del recorrido(minutos)')
plt.ylabel('Altitud del recorrido (m)')
plt.title('Comportamiento de la altitud con respecto a la distancia recorrida')
plt.grid()
plt.legend()
plt.savefig(r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Repositorios\Prueba de Autonomia EV_Prototipo\Tiempo_vs_Altura.png")
plt.show()

# Grafica de la ruta realizada (2D) Sin interactividad
plt.figure(figsize=(10,5))
plt.plot(longitudes, latitudes, marker='o', markersize=2)
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Representación gráfica de la Ruta GPS")
plt.grid()
plt.savefig(r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Repositorios\Prueba de Autonomia EV_Prototipo\Recorrido_GPS.png")
plt.show()

# Mapa interactivo de la ruta recorrida
# Creación de DataFrame

df = pd.DataFrame({
    'lat': latitudes,
    'lon': longitudes,
    'alt': altitudes,
    'time': times,
    'dist': distances
})

df['time'] = pd.to_datetime(df['time'])
df['dt'] = df['time'].diff().dt.total_seconds().fillna(0)
df['ddist'] = df['dist'].diff().fillna(0)
df['vel_kmh'] = df['ddist'] / df['dt'] * 3.6
df['vel_kmh'] = df['vel_kmh'].replace([np.inf, -np.inf], np.nan).fillna(0)
df['dalt'] = df['alt'].diff().fillna(0)
df['pendiente_%'] = (df['dalt'] / df['ddist']).replace([np.inf, -np.inf], 0) * 100
df['pendiente_%'] = df['pendiente_%'].fillna(0)
df['aceleracion'] = df['vel_kmh'].diff().fillna(0) / df['dt'].replace(0, np.nan)
df['aceleracion'] = df['aceleracion'].replace([np.inf, -np.inf], 0).fillna(0)

# Paradas: velocidad < 1 km/h por más de 30 s
parada_mask = (df['vel_kmh'] < 1)
paradas = []
en_parada = False
for i, row in df.iterrows():
    if parada_mask[i] and not en_parada:
        inicio = i
        en_parada = True
    elif not parada_mask[i] and en_parada:
        fin = i
        if (df.loc[fin, 'time'] - df.loc[inicio, 'time']).total_seconds() > 30:
            paradas.append((inicio, fin))
        en_parada = False

# Segmentación de tramos (cada 2 km)
tramos = []
tramo_inicio = 0
for i, d in enumerate(df['dist']):
    if d - df['dist'][tramo_inicio] >= 2000:
        tramos.append((tramo_inicio, i))
        tramo_inicio = i
if tramo_inicio < len(df)-1:
    tramos.append((tramo_inicio, len(df)-1))

# Consumo energético estimado
consumo_kwh_km = 1.2
df['consumo_kwh'] = df['ddist']/1000 * consumo_kwh_km
df['consumo_acum_kwh'] = df['consumo_kwh'].cumsum()

# Estadísticas generales
vel_media = df['vel_kmh'].mean()
vel_max = df['vel_kmh'].max()
vel_min = df['vel_kmh'][df['vel_kmh']>0].min()
alt_min = df['alt'].min()
alt_max = df['alt'].max()
alt_media = df['alt'].mean()
desnivel = df['dalt'][df['dalt']>0].sum()
pend_media = df['pendiente_%'].mean()
pend_max = df['pendiente_%'].max()
pend_min = df['pendiente_%'].min()
acel_max = df['aceleracion'].max()
acel_min = df['aceleracion'].min()
num_paradas = len(paradas)
tiempo_total = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()/60
dist_total = df['dist'].iloc[-1]/1000
consumo_total = df['consumo_acum_kwh'].iloc[-1]

print(f"Distancia total: {dist_total:.2f} km")
print(f"Tiempo total: {tiempo_total:.1f} min")
print(f"Velocidad media: {vel_media:.2f} km/h | Máx: {vel_max:.2f} | Mín: {vel_min:.2f}")
print(f"Altitud media: {alt_media:.1f} m | Máx: {alt_max:.1f} | Mín: {alt_min:.1f}")
print(f"Desnivel acumulado: {desnivel:.1f} m")
print(f"Pendiente media: {pend_media:.2f}% | Máx: {pend_max:.2f}% | Mín: {pend_min:.2f}%")
print(f"Aceleración máx: {acel_max:.2f} m/s² | mín: {acel_min:.2f} m/s²")
print(f"Número de paradas: {num_paradas}")
print(f"Consumo energético estimado: {consumo_total:.2f} kWh")

# Guardar resumen en CSV
df.to_csv(r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Repositorios\Prueba de Autonomia EV_Prototipo\resumen_ruta.csv", index=False)

# =====================
# Gráficas avanzadas
# =====================
plt.figure(figsize=(10,5))
plt.plot(df['dist']/1000, df['vel_kmh'])
plt.axhline(vel_media, color='red', linestyle='--', label=f'Velocidad promedio: {vel_media:.2f} km/h')
plt.xlabel('Distancia (km)')
plt.ylabel('Velocidad (km/h)')
plt.title('Velocidad vs Distancia')
plt.grid()
plt.legend()
plt.savefig(r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Repositorios\Prueba de Autonomia EV_Prototipo\Velocidad_vs_Distancia.png")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df['dist']/1000, df['pendiente_%'])
plt.xlabel('Distancia (km)')
plt.ylabel('Pendiente (%)')
plt.title('Pendiente vs Distancia')
plt.grid()
umbral_pend = 10  # %
extremos = df[(df['pendiente_%'] > umbral_pend) | (df['pendiente_%'] < -umbral_pend)]
plt.scatter(extremos['dist']/1000, extremos['pendiente_%'], color='red', label='Pendiente extrema')
plt.legend()
plt.savefig(r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Repositorios\Prueba de Autonomia EV_Prototipo\pendiente_vs_distancia.png")
plt.show()

plt.figure(figsize=(10,5))
plt.hist(df['vel_kmh'], bins=30, color='skyblue', edgecolor='k')
plt.xlabel('Velocidad (km/h)')
plt.ylabel('Frecuencia')
plt.title('Histograma de velocidades')
plt.grid()
plt.savefig(r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Repositorios\Prueba de Autonomia EV_Prototipo\Histograma_Velocidades.png")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df['dist']/1000, df['aceleracion'])
plt.xlabel('Distancia (km)')
plt.ylabel('Aceleración (m/s²)')
plt.title('Aceleración vs Distancia')
plt.grid()
plt.savefig(r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Repositorios\Prueba de Autonomia EV_Prototipo\Aceleracion_vs_Distancia.png")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df['dist']/1000, df['consumo_acum_kwh'])
plt.xlabel('Distancia (km)')
plt.ylabel('Consumo acumulado (kWh)')
plt.title('Consumo energético estimado vs Distancia (Grafica de Prueba)')
plt.grid()
plt.savefig(r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Repositorios\Prueba de Autonomia EV_Prototipo\Consumo_vs_Distancia.png")
plt.show()

# Mapa interactivo avanzado con paradas y tramos
if latitudes and longitudes:
    center = [latitudes[0], longitudes[0]]
    m = folium.Map(location=center, zoom_start=15)
    folium.PolyLine(list(zip(latitudes, longitudes)), color='blue', weight=4, opacity=0.7, tooltip='Ruta').add_to(m)
    folium.Marker([latitudes[0], longitudes[0]], popup='Inicio', icon=folium.Icon(color='green')).add_to(m)
    folium.Marker([latitudes[-1], longitudes[-1]], popup='Fin', icon=folium.Icon(color='red')).add_to(m)
    # Paradas
    for inicio, fin in paradas:
        folium.Marker([df.loc[inicio, 'lat'], df.loc[inicio, 'lon']], popup=f'Parada {inicio}', icon=folium.Icon(color='orange', icon='pause')).add_to(m)
    # Tramos
    colors = ['purple','cadetblue','darkred','darkgreen','darkblue','orange','black']
    for idx, (i0, i1) in enumerate(tramos):
        folium.PolyLine(df.loc[i0:i1, ['lat','lon']].values, color=colors[idx%len(colors)], weight=3, opacity=0.5, tooltip=f'Tramo {idx+1}').add_to(m)
        # Puntos con pendientes extremas
        umbral_pend = 10  # %
        extremos = df[(df['pendiente_%'] > umbral_pend) | (df['pendiente_%'] < -umbral_pend)]
        for _, row in extremos.iterrows():
            color = 'red' if row['pendiente_%'] > umbral_pend else 'blue'
            folium.CircleMarker([row['lat'], row['lon']], radius=6, color=color, fill=True, fill_opacity=0.7,
                popup=f"Pendiente: {row['pendiente_%']:.1f}%").add_to(m)
    m.save(r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Repositorios\Prueba de Autonomia EV_Prototipo\mapa_ruta_avanzado.html")
    print('Mapa interactivo avanzado guardado como mapa_ruta_avanzado.html')

# Centro del mapa: primer punto de la ruta
if latitudes and longitudes:
    center = [latitudes[0], longitudes[0]]
    m = folium.Map(location=center, zoom_start=15)

    # Dibujar la ruta
    folium.PolyLine(list(zip(latitudes, longitudes)), color='blue', weight=4, opacity=0.7, tooltip='Ruta').add_to(m)

    # Marcar inicio y fin
    folium.Marker([latitudes[0], longitudes[0]], popup='Inicio', icon=folium.Icon(color='green')).add_to(m)
    folium.Marker([latitudes[-1], longitudes[-1]], popup='Fin', icon=folium.Icon(color='red')).add_to(m)

    # Agregar waypoints si existen
    if hasattr(gpx, 'waypoints') and gpx.waypoints:
        for wp in gpx.waypoints:
            folium.Marker([wp.latitude, wp.longitude], popup=wp.name or 'Parada', icon=folium.Icon(color='orange', icon='flag')).add_to(m)

    # Guardar el mapa
    m.save(r"C:\haranzales\OneDrive - Superpolo S.A.S\Ingenieria\Repositorios\Prueba de Autonomia EV_Prototipo\Mapa_Ruta_Paradas.html")
    print('Mapa interactivo guardado como mapa_ruta.html')
else:
    print("No se encontraron purnto para recontruir la ruta")