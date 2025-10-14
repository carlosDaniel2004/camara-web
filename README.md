# Proyecto de Reconocimiento Facial

Este es un proyecto de reconocimiento facial basado en la web utilizando Flask, OpenCV y face_recognition. Está diseñado para ser desplegado utilizando Docker.

## Requisitos

- Docker

## Cómo ejecutar el proyecto

1.  **Construir la imagen de Docker:**

    ```bash
    docker build -t reconocimiento-facial .
    ```

2.  **Ejecutar el contenedor de Docker:**

    ```bash
    docker run -p 8000:8000 reconocimiento-facial
    ```

3.  **Acceder a la aplicación:**

    Abre tu navegador web y ve a `http://localhost:8000`.

## Cómo añadir personas para reconocer

1.  Añade las imágenes de las personas que quieres reconocer en la carpeta `app/models`.
2.  En el archivo `app/app.py`, descomenta y modifica las siguientes líneas para cargar las imágenes y los nombres de las personas:

    ```python
    # obama_image = face_recognition.load_image_file("app/models/obama.jpg")
    # obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    # known_face_encodings.append(obama_face_encoding)
    # known_face_names.append("Barack Obama")
    ```

    Por cada persona que quieras añadir, copia y pega este bloque de código y cambia el nombre del archivo de la imagen y el nombre de la persona.
