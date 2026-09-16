import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text  # <--- Nueva importación

# =====================================================================
# CONFIGURACIÓN: Cambia aquí el nombre de la columna que quieres graficar
# =====================================================================
metrica_a_graficar = "system_total_waiting_time"
# =====================================================================

# 1. Cargar los 3 archivos de promedios
try:
    df_combinada = pd.read_csv(r"resultados un agente\pruebas\combinada\SARSA_run_1\SARSA_run_1_conn0_ep0.csv")
    df_dif_waiting = pd.read_csv(r"resultados un agente\pruebas\dif\SARSA_run_1\SARSA_run_1_conn0_ep0.csv")
    df_presion = pd.read_csv(r"resultados un agente\pruebas\presion\SARSA_run_1\SARSA_run_1_conn0_ep0.csv")
except FileNotFoundError as e:
    print(f"Error: Asegúrate de que los archivos estén en la misma carpeta. Detalle: {e}")
    exit()

# Verificar si la métrica escrita existe
if metrica_a_graficar not in df_combinada.columns:
    print(f"Error: La métrica '{metrica_a_graficar}' no existe en los archivos CSV.")
    exit()

# 2. Calcular los promedios globales
media_global_comb = df_combinada[metrica_a_graficar].mean()
media_global_dif = df_dif_waiting[metrica_a_graficar].mean()
media_global_pres = df_presion[metrica_a_graficar].mean()

# 3. Configurar el diseño de la gráfica
fig, ax = plt.subplots(figsize=(12, 7))
plt.grid(True, linestyle="--", alpha=0.5)

# Colores fijos
color_comb = "#1f77b4"
color_dif = "#ff7f0e"
color_pres = "#2ca02c"

# 4. Graficar las líneas de evolución paso a paso
ax.plot(df_combinada["step"], df_combinada[metrica_a_graficar], 
        label="Media Combinada (Evolución)", color=color_comb, linewidth=1.2, alpha=0.85)

ax.plot(df_dif_waiting["step"], df_dif_waiting[metrica_a_graficar], 
        label="Media Dif Waiting Time (Evolución)", color=color_dif, linewidth=1.2, alpha=0.85)

ax.plot(df_presion["step"], df_presion[metrica_a_graficar], 
        label="Media Presión (Evolución)", color=color_pres, linewidth=1.2, alpha=0.85)

# 5. Añadir las líneas horizontales de las medias globales
ax.axhline(y=media_global_comb, color=color_comb, linestyle="--", alpha=0.9, linewidth=1.5,
           label=f"Promedio Combinada ({media_global_comb:.2f})")

ax.axhline(y=media_global_dif, color=color_dif, linestyle="--", alpha=0.9, linewidth=1.5,
           label=f"Promedio Dif Waiting ({media_global_dif:.2f})")

ax.axhline(y=media_global_pres, color=color_pres, linestyle="--", alpha=0.9, linewidth=1.5,
           label=f"Promedio Presión ({media_global_pres:.2f})")

# 6. Añadir las etiquetas de texto flotantes (Guardándolas en una lista)
x_pos_texto = df_combinada["step"].max() * 0.93 
texts = []

texts.append(ax.text(x_pos_texto, media_global_comb, f" {media_global_comb:.2f} ", color="white", weight="bold",
                     fontsize=9, va="center", ha="center", bbox=dict(facecolor=color_comb, alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2')))

texts.append(ax.text(x_pos_texto, media_global_dif, f" {media_global_dif:.2f} ", color="white", weight="bold",
                     fontsize=9, va="center", ha="center", bbox=dict(facecolor=color_dif, alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2')))

texts.append(ax.text(x_pos_texto, media_global_pres, f" {media_global_pres:.2f} ", color="white", weight="bold",
                     fontsize=9, va="center", ha="center", bbox=dict(facecolor=color_pres, alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2')))

# AJUSTE AUTOMÁTICO: Evita que los textos se pisen entre sí en el eje Y
adjust_text(texts, autoalign='y', only_move={'text': 'y'})

# 7. Títulos y etiquetas de los ejes
plt.title(f"Promedio entrenamiento", fontsize=13, pad=15, weight='bold')
plt.xlabel("Pasos", fontsize=11)
plt.ylabel("Tiempo de Espera", fontsize=11)

# 8. Mover la leyenda ABAJO y organizarla en 2 columnas
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10, frameon=True)

# 9. Guardar y mostrar
nombre_imagen = f"graficaprueba_{metrica_a_graficar}.png"
plt.savefig(nombre_imagen, dpi=300, bbox_inches="tight")
print(f"Gráfica guardada exitosamente como '{nombre_imagen}'")

plt.show()