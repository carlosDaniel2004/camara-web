from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from deepface import DeepFace
import os
import base64

app = Flask(__name__)
# Es necesario para mostrar mensajes (flash)
app.secret_key = "tu_clave_secreta_aqui"

# --- Configuración ---
MODELS_PATH = 'app/models'
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)
    print(f"Directorio '{MODELS_PATH}' creado. Por favor, añade imágenes en esta carpeta.")

# --- Rutas de Flask ---
@app.route('/')
def index():
    """Sirve la página principal."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Gestiona la subida de nuevas fotos para el reconocimiento."""
    if 'photo' not in request.files or 'name' not in request.form:
        flash('Error: Faltan partes del formulario.')
        return redirect(url_for('index'))
    
    file = request.files['photo']
    name = request.form.get('name', '').strip()

    if file.filename == '':
        flash('Error: No se seleccionó ningún archivo.')
        return redirect(url_for('index'))

    if not name:
        flash('Error: El campo de nombre no puede estar vacío.')
        return redirect(url_for('index'))

    if file and name:
        # Crea un nombre de archivo seguro a partir del nombre de la persona
        filename = secure_filename(name) + os.path.splitext(file.filename)[1]
        file.save(os.path.join(MODELS_PATH, filename))
        flash(f'¡Foto de "{name}" subida exitosamente!')

        # Borra el archivo de caché de deepface para forzar una re-indexación
        representations_path = os.path.join(MODELS_PATH, "representations_vgg_face.pkl")
        if os.path.exists(representations_path):
            try:
                os.remove(representations_path)
                flash('Índice de rostros actualizado.')
            except OSError as e:
                flash(f'Error al actualizar el índice: {e}')
    
    return redirect(url_for('index'))

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Recibe un fotograma de video, lo procesa y devuelve los resultados."""
    try:
        data = request.json['image']
        header, encoded = data.split(",", 1)
        image_data = base64.b64decode(encoded)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Frame inválido'}), 400

        all_faces_coords = []
        recognized_faces_data = []

        # 1. Detectar todos los rostros
        try:
            all_faces_list = DeepFace.extract_faces(
                img_path=frame,
                detector_backend='opencv',
                enforce_detection=False
            )
            if all_faces_list:
                all_faces_coords = [face['facial_area'] for face in all_faces_list]
        except Exception:
            pass

        # 2. Reconocer rostros conocidos
        try:
            if any(fname.endswith(('.jpg', '.jpeg', '.png')) for fname in os.listdir(MODELS_PATH)):
                dfs = DeepFace.find(
                    img_path=frame,
                    db_path=MODELS_PATH,
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                if dfs and not dfs[0].empty:
                    for _, row in dfs[0].iterrows():
                        if 'identity' in row and 'source_x' in row:
                            identity = row['identity']
                            x, y, w, h = row['source_x'], row['source_y'], row['source_w'], row['source_h']
                            name = os.path.splitext(os.path.basename(identity))[0].replace("_", " ").title()
                            recognized_faces_data.append({'name': name, 'coords': {'x': x, 'y': y, 'w': w, 'h': h}})
        except Exception:
            pass

        return jsonify({
            'all_faces': all_faces_coords,
            'recognized_faces': recognized_faces_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
