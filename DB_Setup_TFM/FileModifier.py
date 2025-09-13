import os

# Configuración
input_folder = "DB/labels"  # Carpeta con los archivos originales
output_folder = "DB/labels_modified"  # Carpeta donde se guardarán los archivos modificados
current_index = 0
new_index = 3  # Número con el que se quiere reemplazar el índice "i" de la clase actual en la primera posición
ln_counter = 0  # Contador de líneas modificadas
fl_counter = 0  # Contador de archivos modificados

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Procesar cada archivo en la carpeta de entrada
for filename in os.listdir(input_folder):
    fl_name = 'Nombre de un archivo distinto'

    if filename.endswith(".txt"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, "r") as infile:
            lines = infile.readlines()

        modified_lines = []
        for line in lines:
            parts = line.split()

            if parts and parts[0] == str(current_index):  # Si la primera posición es la buscada, cambiarla
                parts[0] = str(new_index)
                ln_counter += 1  # Contando líneas modificadas

                if fl_name != filename:
                    fl_counter += 1  # Contando el archivo como modificado
                    fl_name = filename
                    print(filename)

            modified_lines.append(" ".join(parts) + "\n")

        with open(output_path, "w") as outfile:
            outfile.writelines(modified_lines)

print(f"Proceso completado. {ln_counter} Líneas de {fl_counter} archivos modificados guardados en '{output_folder}'.")
