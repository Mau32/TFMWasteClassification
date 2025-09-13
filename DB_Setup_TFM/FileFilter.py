import os
import shutil

# Ruta original
origen_labels = "DB/labels"
origen_images = "DB/images"

# Ruta destino
destino_labels = "DB_DETECTION/labels"
destino_images = "DB_DETECTION/images"

# Crear carpetas de destino
os.makedirs(destino_labels, exist_ok=True)
os.makedirs(destino_images, exist_ok=True)


# Función para verificar si una línea es de detección (YOLO: 5 valores)
def es_linea_deteccion(linea):
    partes = linea.strip().split()
    if len(partes) != 5:
        return False
    try:
        float_vals = [float(x) for x in partes[1:]]
        return all(0.0 <= val <= 1.0 for val in float_vals)
    except ValueError:
        return False


# Contadores
aceptados = 0
descartados = 0
sin_imagen = 0

# Recorrer los archivos .txt de labels
for archivo in os.listdir(origen_labels):
    if not archivo.endswith(".txt"):
        continue

    ruta_label = os.path.join(origen_labels, archivo)

    with open(ruta_label, "r") as f:
        lineas = [l.strip() for l in f if l.strip()]

    # Aceptar solo si tiene líneas y todas son de detección
    if len(lineas) > 0 and all(es_linea_deteccion(l) for l in lineas):
        # Copiar label
        shutil.copy2(ruta_label, os.path.join(destino_labels, archivo))

        # Copiar imagen correspondiente
        nombre_base = os.path.splitext(archivo)[0]
        imagen_copiada = False
        for ext in ['.jpg', '.jpeg', '.png']:
            ruta_img = os.path.join(origen_images, nombre_base + ext)
            if os.path.exists(ruta_img):
                shutil.copy2(ruta_img, os.path.join(destino_images, nombre_base + ext))
                imagen_copiada = True
                break

        if imagen_copiada:
            aceptados += 1
        else:
            sin_imagen += 1
            print(f"[!] Imagen no encontrada para: {archivo}")

    else:
        descartados += 1

# Mostrar resumen
print("\n=== RESUMEN ===")
print(f"✅ Archivos aceptados       : {aceptados}")
print(f"❌ Archivos descartados     : {descartados}")
print(f"⚠  Etiquetas sin imagen     : {sin_imagen}")
print("====================")
