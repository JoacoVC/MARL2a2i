import os
import glob
import pandas as pd

# 1. Configura la ruta de la carpeta donde están tus 20 archivos CSV
# (Puedes usar '.' si el script está en la misma carpeta que los archivos)
ruta_carpeta = "output\csv\entrenamiento\presion\SARSA_run_1" 
    
# 2. Obtener la lista de todos los archivos CSV en esa carpeta
archivos_csv = glob.glob(os.path.join(ruta_carpeta, "*.csv"))

print(f"Se han encontrado {len(archivos_csv)} archivos CSV para procesar.")

# 3. Leer todos los archivos y guardarlos en una lista
lista_dataframes = []
for archivo in archivos_csv:
    df = pd.read_csv(archivo)
    lista_dataframes.append(df)

# 4. Concatenar todos los datos en un único DataFrame gigante
data_combinada = pd.concat(lista_dataframes, ignore_index=True)

# 5. Agrupar por la columna 'step' y calcular la media de todas las columnas
df_media = data_combinada.groupby('step', as_index=False).mean()

# 6. Guardar el resultado en un nuevo archivo CSV idéntico en estructura
archivo_salida = "output\csv\entrenamiento\presion\SARSA_run_1\media_presion.csv"
df_media.to_csv(archivo_salida, index=False)

print(f"¡Proceso completado con éxito! El archivo agrupado se guardó como: {archivo_salida}")