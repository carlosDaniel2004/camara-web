from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import os
import base64

app = Flask(__name__)

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

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Recibe un fotograma de video, lo procesa y devuelve los resultados."""
    try:
        # Recibir la imagen en formato base64
        data = request.json['image']
        # Decodificar la imagen
        header, encoded = data.split(",", 1)
        image_data = base64.b64decode(encoded)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Frame inválido'}), 400

        # --- Detección y Reconocimiento ---
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
            pass # Ignorar si no se detectan caras

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
            pass # Ignorar si no hay coincidencias

        return jsonify({
            'all_faces': all_faces_coords,
            'recognized_faces': recognized_faces_data
        })

    except Exception as e:
        print(f"Error en process_frame: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
