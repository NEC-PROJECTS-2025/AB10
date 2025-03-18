import shutil
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Paths
dataset_path = "rgb_blur"
model_path = "models/model.h5"
class_names = ['scroll_down', 'scroll_left', 'scroll_right', 'scroll_up', 'zoom_in', 'zoom_out']
static_path = os.path.join(app.root_path, 'static')
# Model Loading
model = load_model(model_path)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(120, 160, 3))
feature_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

# Helper Functions
def preprocess_and_extract_features(sequence_path):
    num_frames_per_sequence = 10
    frame_height, frame_width, color_channels = 120, 160, 3
    selected_frames_indices = list(range(0, 40, 4))[:num_frames_per_sequence]
    sequence_frames = []

    for frame_index in selected_frames_indices:
        frame_file = f'{frame_index}.png'
        frame_path = os.path.join(sequence_path, frame_file)
        if os.path.exists(frame_path):
            img = tf.keras.preprocessing.image.load_img(frame_path, target_size=(frame_height, frame_width))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            sequence_frames.append(img_array)

    if len(sequence_frames) == num_frames_per_sequence:
        sequence_frames = preprocess_input(np.array(sequence_frames))
        features = [feature_extractor.predict(np.expand_dims(img, axis=0)).flatten() for img in sequence_frames]
        return np.expand_dims(np.array(features), axis=0)
    return None


# Set the UPLOAD_FOLDER in Flask configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['STATIC_FOLDER'] = os.path.join(app.root_path, 'static')
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/project', methods=['GET', 'POST'])
def project():
    gestures = sorted(os.listdir(dataset_path))
    selected_gesture = request.form.get('gesture')
    selected_sequence = request.form.get('sequence')

    # Handle Dataset Prediction
    if selected_gesture and selected_sequence:
        sequence_path = os.path.join(dataset_path, selected_gesture, selected_sequence)
        return redirect(url_for('result', gesture=selected_gesture, sequence=selected_sequence))

    sequences = os.listdir(os.path.join(dataset_path, selected_gesture)) if selected_gesture else []
    sequence_images = [
        os.path.join(dataset_path, selected_gesture, sequence, f"{i}.png")
        for i in range(40)
    ] if selected_gesture and selected_sequence else []

    # Handle Realtime Prediction
    if 'files' in request.files:
        files = request.files.getlist('files')

        # Create a folder for real-time uploads
        upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'realtime_upload')
        if os.path.exists(upload_folder):
            shutil.rmtree(upload_folder)
        os.makedirs(upload_folder, exist_ok=True)

        # Save uploaded files
        for file in files:
            filename = secure_filename(file.filename)
            file.save(os.path.join(upload_folder, filename))

        # Validate and predict
        uploaded_files = sorted(os.listdir(upload_folder), key=lambda x: int(x.split('.')[0]))
        if len(uploaded_files) != 40:
            return render_template(
                'project.html',
                gestures=gestures,
                sequences=sequences,
                selected_gesture=selected_gesture,
                sequence_images=sequence_images,
                error="Please upload exactly 40 images for real-time prediction."
            )

        features = preprocess_and_extract_features(upload_folder)
        if features is not None:
            predicted_probs = model.predict(features)
            predicted_label = np.argmax(predicted_probs, axis=1)[0]
            prediction = class_names[predicted_label]
        else:
            prediction = "Error processing the images. Check file format."

        return render_template('result.html', gesture="Uploaded -Unknown", sequence="Uploaded Images", prediction=prediction)

    return render_template(
        'project.html',
        gestures=gestures,
        sequences=sequences,
        selected_gesture=selected_gesture,
        sequence_images=sequence_images
    )


@app.route('/result')
def result():
    gesture = request.args.get('gesture')
    sequence = request.args.get('sequence')

    if gesture == "Realtime Prediction":
        prediction = request.args.get('prediction')
    else:
        sequence_path = os.path.join(dataset_path, gesture, sequence)
        features = preprocess_and_extract_features(sequence_path)
        if features is not None:
            predicted_probs = model.predict(features)
            predicted_label = np.argmax(predicted_probs, axis=1)[0]
            prediction = class_names[predicted_label]
        else:
            prediction = "Error processing sequence. Try again."

    return render_template('result.html', gesture=gesture, sequence=sequence, prediction=prediction)



@app.route('/about')
def aboutus():
    return render_template('about.html', title="About Project")


@app.route('/metrics')
def conference_paper():
    return render_template('metrics.html', title="Metrics")


@app.route('/flowchart')
def working():
    return render_template('flowchart.html', title="How It Works")


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(static_path, filename)

@app.route('/realtime', methods=['GET', 'POST'])
def realtime():
    if request.method == 'POST':
        # Capturing video through form submission
        video_file = request.files.get('video')
        if not video_file:
            return render_template('realtime.html', error="No video file uploaded.")
        
        # Save video to the upload folder
        video_filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video_file.save(video_path)

        # Extract frames from video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        # Choose 10 equally spaced frames
        num_frames_to_capture = 10
        frame_indices = np.linspace(0, total_frames - 1, num_frames_to_capture, dtype=int)
        captured_frames = []
        frame_save_folder = os.path.join(static_path, 'realtime_frames')

        if os.path.exists(frame_save_folder):
            shutil.rmtree(frame_save_folder)
        os.makedirs(frame_save_folder, exist_ok=True)

        for i, frame_index in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (160, 120))  # Resize to match model requirements
                frame_filename = f'frame_{i}.png'
                frame_path = os.path.join(frame_save_folder, frame_filename)
                cv2.imwrite(frame_path, frame_resized)
                captured_frames.append(frame_resized)

        cap.release()

        # Preprocess and extract features from captured frames
        features = None
        if len(captured_frames) == num_frames_to_capture:
            captured_frames = preprocess_input(np.array(captured_frames))
            features = [feature_extractor.predict(np.expand_dims(img, axis=0)).flatten() for img in captured_frames]
            features = np.expand_dims(np.array(features), axis=0)

        # Perform prediction if features are extracted successfully
        prediction = "Error: Unable to process captured frames for prediction."
        if features is not None:
            predicted_probs = model.predict(features)
            predicted_label = np.argmax(predicted_probs, axis=1)[0]
            prediction = class_names[predicted_label]

        return render_template(
            'realtime_result.html',
            video_properties={
                'Total Frames': total_frames,
                'FPS': round(fps, 2),
                'Width': width,
                'Height': height,
                'Duration': round(duration, 2)
            },
            frames=[f'static/realtime_frames/frame_{i}.png' for i in range(num_frames_to_capture)],
            prediction=prediction
        )

    return render_template('realtime.html')


if __name__ == '__main__':
    app.run(debug=True)
