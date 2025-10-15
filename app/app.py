from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from deepface import DeepFace
import os
import base64

app = Flask(__name__)
app.secret_key = "tu_clave_secreta_aqui"

# --- Configuración ---
MODELS_PATH = 'app/models'
UPLOAD_FOLDER = 'app/static'
PROCESSED_FOLDER = 'app/static/processed_videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# --- Funciones de Detección de Geometría ---

def get_color_name(hsv):
    """Identifica el nombre de un color basado en su valor HSV."""
    h, s, v = hsv
    if s < 25 and v > 180: return "Blanco"
    if v < 50: return "Negro"
    if s < 100 and v < 100: return "Gris"

    # Rangos de H (Hue) para colores
    if (h < 10 or h > 170): return "Rojo"
    if (h < 25): return "Naranja"
    if (h < 35): return "Amarillo"
    if (h < 85): return "Verde"
    if (h < 130): return "Azul"
    if (h < 160): return "Violeta"
    return "Desconocido"

def detect_shapes(frame):
    """Detecta triángulos, cuadrados, rectángulos y círculos en un frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Usar Canny edge detection para encontrar los bordes de las figuras
    edged = cv2.Canny(blurred, 50, 150)
    
    # Encontrar contornos en el mapa de bordes
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        
        # Obtener el color promedio del contorno
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val = cv2.mean(frame, mask=mask)
        hsv_color = cv2.cvtColor(np.uint8([[mean_val[:3]]]), cv2.COLOR_BGR2HSV)[0][0]
        color_name = get_color_name(hsv_color)

        x, y, w, h = cv2.boundingRect(approx)
        shape_name = ""
        
        if len(approx) == 3:
            shape_name = "Triangulo"
        elif len(approx) == 4:
            aspect_ratio = float(w) / h
            shape_name = "Cuadrado" if 0.95 < aspect_ratio < 1.05 else "Rectangulo"
        else:
            # Detección de círculos
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            area = cv2.contourArea(cnt)
            circle_area = np.pi * (radius ** 2)
            if abs(1 - (area / circle_area)) < 0.2: # Si el área del contorno es similar al área del círculo
                shape_name = "Circulo"

        if shape_name:
            label = f"{color_name} {shape_name}"
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return frame

# --- Rutas de Reconocimiento Facial ---

@app.route('/')
def index():
    """Sirve la página principal de reconocimiento facial."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # ... (código de subida de fotos existente)
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
        filename = secure_filename(name) + os.path.splitext(file.filename)[1]
        file.save(os.path.join(MODELS_PATH, filename))
        flash(f'¡Foto de "{name}" subida exitosamente!')

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
    # ... (código de procesamiento de frame existente)
    try:
        data = request.json['image']
        header, encoded = data.split(",", 1)
        image_data = base64.b64decode(encoded)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None: return jsonify({'error': 'Frame inválido'}), 400

        all_faces_coords = []
        recognized_faces_data = []

        try:
            all_faces_list = DeepFace.extract_faces(img_path=frame, detector_backend='opencv', enforce_detection=False)
            if all_faces_list:
                all_faces_coords = [face['facial_area'] for face in all_faces_list]
        except Exception: pass

        try:
            if any(fname.endswith(('.jpg', '.jpeg', '.png')) for fname in os.listdir(MODELS_PATH)):
                dfs = DeepFace.find(img_path=frame, db_path=MODELS_PATH, enforce_detection=False, detector_backend='opencv')
                if dfs and not dfs[0].empty:
                    for _, row in dfs[0].iterrows():
                        if 'identity' in row and 'source_x' in row:
                            identity, x, y, w, h = row['identity'], row['source_x'], row['source_y'], row['source_w'], row['source_h']
                            name = os.path.splitext(os.path.basename(identity))[0].replace("_", " ").title()
                            recognized_faces_data.append({'name': name, 'coords': {'x': x, 'y': y, 'w': w, 'h': h}})
        except Exception: pass

        return jsonify({'all_faces': all_faces_coords, 'recognized_faces': recognized_faces_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Rutas de Detección de Geometría ---

@app.route('/geometry', methods=['GET', 'POST'])
def geometry_page():
    """Muestra la página de subida de video y procesa el video."""
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('Error: No se encontró el archivo de video.', 'error')
            return redirect(request.url)
        
        file = request.files['video']
        if file.filename == '':
            flash('Error: No se seleccionó ningún archivo.', 'error')
            return redirect(request.url)

        if file:
            try:
                filename = secure_filename(file.filename)
                input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(input_path)

                # Solución Definitiva: Usar el códec VP9 en un contenedor WebM para máxima compatibilidad.
                base_filename = os.path.splitext(filename)[0]
                output_filename = f"processed_{base_filename}.webm"
                output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
                
                cap = cv2.VideoCapture(input_path)
                fourcc = cv2.VideoWriter_fourcc(*'VP90')
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                if not out.isOpened():
                    raise IOError("No se pudo abrir el VideoWriter. Verifica que el códec 'avc1' esté disponible en el entorno del servidor.")

                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    processed_frame = detect_shapes(frame)
                    out.write(processed_frame)

                cap.release()
                out.release()
                os.remove(input_path) # Elimina el video original

                flash('¡Video procesado exitosamente!', 'success')
                return render_template('geometry.html', processed_video=output_filename)
            except Exception as e:
                # Log detallado del error en la consola del servidor
                import traceback
                print("--- OCURRIÓ UN ERROR AL PROCESAR EL VIDEO ---")
                traceback.print_exc()
                print("---------------------------------------------")
                flash(f'Error interno al procesar el video: {e}', 'error')
                return redirect(request.url)

    return render_template('geometry.html')

@app.route('/delete/<string:filename>')
def delete_video(filename):
    """Elimina un video procesado de la carpeta static."""
    try:
        # Medida de seguridad: solo permitir borrar archivos que empiecen con 'processed_'
        if not filename.startswith('processed_'):
            flash('Error: Intento de eliminación de archivo no válido.', 'error')
            return redirect(url_for('geometry_page'))

        # Apuntar a la nueva carpeta de videos procesados
        filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            flash(f'Video "{filename}" eliminado correctamente.', 'success')
        else:
            flash(f'Error: El video "{filename}" no fue encontrado.', 'error')
    except Exception as e:
        flash(f'Ocurrió un error al eliminar el video: {e}', 'error')
    
    return redirect(url_for('geometry_page'))

# --- Rutas de Detección de Geometría en Vivo ---

@app.route('/realtime_geometry')
def realtime_geometry_page():
    """Muestra la página de detección de geometría en tiempo real."""
    return render_template('realtime_geometry.html')

@app.route('/process_geometry_frame', methods=['POST'])
def process_geometry_frame():
    """Recibe un fotograma, detecta figuras y devuelve los resultados."""
    try:
        data = request.json['image']
        header, encoded = data.split(",", 1)
        image_data = base64.b64decode(encoded)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Frame inválido'}), 400

        # Reutilizamos la lógica de detección de formas
        shapes_data = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue

            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_val = cv2.mean(frame, mask=mask)
            hsv_color = cv2.cvtColor(np.uint8([[mean_val[:3]]]), cv2.COLOR_BGR2HSV)[0][0]
            color_name = get_color_name(hsv_color)

            x, y, w, h = cv2.boundingRect(approx)
            shape_name = ""
            
            if len(approx) == 3: shape_name = "Triangulo"
            elif len(approx) == 4:
                aspect_ratio = float(w) / h
                shape_name = "Cuadrado" if 0.95 < aspect_ratio < 1.05 else "Rectangulo"
            else:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                area = cv2.contourArea(cnt)
                circle_area = np.pi * (radius ** 2)
                if abs(1 - (area / circle_area)) < 0.2: shape_name = "Circulo"

            if shape_name:
                shapes_data.append({
                    'label': f"{color_name} {shape_name}",
                    'x': x, 'y': y, 'w': w, 'h': h
                })
        
        return jsonify({'shapes': shapes_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
