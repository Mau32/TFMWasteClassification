# Desarrollo de un sistema de clasificación de residuos domésticos mediante un brazo robótico

---

El proyecto implementa un sistema automatizado de **clasificación de residuos domésticos** que integra visión por computador, aprendizaje profundo y robótica colaborativa mediante el uso de:  
- **YOLOv8** para la detección y clasificación de objetos.  
- **Cámara estéreo OAK-D Lite (Luxonis)** para estimación 3D.  
- **Brazo robótico UR3e (Universal Robots)** para la manipulación física de los residuos.  

---

## 📂 Estructura del repositorio

```
TFMWasteClassification/
│── main.py                     # Script principal de ejecución
│── ApriltagDetector.py         # Módulo de detección de Apriltags
│── DataCollection.py           # Módulo opcional para la creación de una base de datos
│── RobotControl.py             # Módulo de gestión de la comunicación y movimiento del brazo robótico con RTDE
│── Transformations.py          # Módulo de operaciones matemáticas de matrices
│── YOLOv8ObjDetector.py        # Módulo de implementación del modelo de deteccón YOLOv8 y de estimación espacial con la cámara OAK-D Lite
│── requirements.txt            # Librerías necesarias para el correcto funcionamiento del programa
│── Yolo-Weights/               # Modelos de YOLOv8 entrenados (versiones 1E y 2D presentes en este repositorio)
│── RobotScripts/               # Archivos .urp y programas de instalación para la Polyscope del robot UR3e
│── DB_Setup_TFM/               # Scripts de manejo de datasets (reordenar, eliminar, cambiar índices de clases)
│── README.md                   # Este archivo
│── data.yaml                   # Archivo que acompaña la base de datos con información general de su distribución y clases
│── .gitignore                  # Exclusiones (venv, __pycache__, .idea, etc.)
```

---

## ⚙️ Requisitos

- **Python 3.10**  
- Librerías: instalar con  
  ```bash
  pip install -r requirements.txt
  ```  
- Entorno: Ubuntu 24.04 (probado con PyCharm y cámara OAK-D Lite).  
- **Robot UR3e** con Polyscope.  

---

## 🧠 Modelos YOLOv8 disponibles

El repositorio incluye dos modelos entrenados finales:  

- **Propuesta I (1E)**: `Carton`, `Envases`, `Vidrio`  
- **Propuesta II (2D)**: `Carton`, `Latas`, `Plastico`, `Vidrio`  

Los pesos están en la carpeta `weights/`.  

👉 En caso de querer entrenar nuevos modelos, los enlaces a las bases de datos y notebooks de entrenamiento en Google Colab se encuentran en los **Anexos de la memoria**.  

---

## 🚀 Puesta en marcha / Instalación

1. **Configurar el entorno Python**  
   - Instalar dependencias:  
     ```bash
     pip install -r requirements.txt
     ```  

2. **Configurar el robot UR3e**  
   - Copiar a la Polyscope los programas disponibles en la carpeta `RobotScripts/` (archivos `.urp` y programas de instalación).  
   - En `main.py`, configurar la dirección del robot:  
     ```python
     robot_ip = "xxx.xxx.xxx.xxx"
     port = xxxxx
     ```  

3. **Elegir la propuesta de clasificación**  
   - En `main.py`, seleccionar el modelo YOLO correspondiente:  
     ```python
     # Para Propuesta I:
     model = YOLO("weights/yolov8l_PROPUESTA_I.pt")
     # objects = ["Carton", "Envases", "Vidrio"]

     # Para Propuesta II:
     model = YOLO("weights/yolov8l_PROPUESTA_II.pt")
     # objects = ["Carton", "Latas", "Plastico", "Vidrio"]
     ```
   - Asegúrate de comentar o descomentar la lista de clases adecuada según la propuesta elegida.  

4. **Ejecutar el sistema**  
   ```bash
   python main.py
   ```  

5. **Opcional – Manejo de bases de datos**  
   - Los programas de `DB_Setup_TFM/` permiten reorganizar datasets, cambiar índices de clases, y limpiar registros. Útiles si se desea reentrenar nuevos modelos en YOLOv8.
   - Existen dos funciones más a las propias de detectar y clasificar objetos en main.py:
     - Activar la cámara para tomar imágenes.
     - Crear una base de datos con el orden de carpetas usual para el entrenamiento de YOLO.

---

## 🧩 Enlaces relevantes

- **Bases de datos**: utilizadas para este TFM, disponibles en https://mega.nz/folder/xl5zHIxQ#cZ7AY5HGoctJ8lNEyEN3RA.  
- **Videos del sistema en funcionamiento**: accesibles en la lista de reproducción de YouTube https://www.youtube.com/playlist?list=PL7M1fQJCSgQcr_DgHD3ZIVviH09riklLt.
- **Scripts de entrenamiento Google Colab Notebooks**: utilizados para el reentranemiento de los modelos:
  - https://colab.research.google.com/drive/1LJx3o_qCiY7XvwKirQhIxRT5SHodkWq7?usp=drive_link
  - https://colab.research.google.com/drive/13PWNDt4WnBbizawfPwXNcWRtxjsbD3Kg?usp=drive_link
 
---

## 👨‍💻 Autor

**Maurizio Rocco D’Alvano Teran**  
Máster Universitario en Ingeniería Electromecánica  
Universidad Politécnica de Madrid (ETSIDI-UPM)
