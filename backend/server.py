from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from keras.initializers import VarianceScaling
from keras.optimizers import SGD
from sklearn.metrics import silhouette_score, davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer
from autoencoder import autoencoder
from method2 import Lab_preprocessing
from method3 import log_preprocessing
from method4 import hsv_preprocessing
from method5 import lab_log_preprocessing
from method6 import no_local_means_filter_preprocessing
from dec import DEC
from skull_stripping import skull_stripping
from sklearn.cluster import KMeans
from keras.callbacks import EarlyStopping
import matplotlib
from flask_socketio import SocketIO, emit
import os
import traceback

matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, async_mode='gevent', cors_allowed_origins="*")

def choose_method(method, image):
    if method == "1":
        x = image.reshape((-1, 3))
        x = x / 255
        return x
    elif method == "2":
        x = Lab_preprocessing(image=image)
        return x
    elif method == "3":
        x = log_preprocessing(image=image)
        return x
    elif method == "4":
        x = hsv_preprocessing(image=image)
        return x
    elif method == "5":
        x = lab_log_preprocessing(image=image)
        return x
    elif method == "6":
        x = no_local_means_filter_preprocessing(image=image)
        return x
    else:
        print("Invalid method choice")
        return None

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        domain = request.form.get('domain')
        method = request.form.get('method')
        print(method)
        print(file)
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Read image file
            nparr = np.frombuffer(file.read(), np.uint8)
            print("fille",nparr)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Ensure img is read correctly
            if img is None:
                return jsonify({'error': 'Invalid image format'}), 400

            pretrain_epochs = 1
            batch_size = 256
            maxiter = 2000
            update_interval = 140
            verbose = 1
            pretrain_optimizer = 'adam'
            print(domain)
            print(method)

            if domain == "brain tumor detection":
                image = skull_stripping(img)
                print("ok",image)
            else:
                image = img

            x = choose_method(method=method, image=image)

            if x is None:
                return jsonify({'error': 'Invalid preprocessing method'}), 400

            dims = [x.shape[-1], 500, 500, 2000, 3]
            loss = 'mse'
            optimizer = SGD(0.001, 0.9)
            init = VarianceScaling(scale=1./3., mode='fan_in', distribution='uniform')
            _autoencoder, _encoder = autoencoder(dims=dims, activation='relu', initializer='glorot_uniform')

            socketio.emit('training', {'data': 'Autoencoder Training Phase Started', 'progress': 10})
            socketio.sleep(0)
            _autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
            early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1, restore_best_weights=True)
            socketio.emit('training', {'data': 'Autoencoder Training Phase in Progress', 'progress': 15})
            socketio.sleep(1)
            _autoencoder.fit(x, x, batch_size=256, epochs=pretrain_epochs, verbose=1, callbacks=[early_stop])
            socketio.emit('training', {'data': 'Autoencoder Training Phase Completed', 'progress': 35})
            socketio.sleep(2)

            km = KMeans(random_state=42, init="k-means++")
            visualizer = KElbowVisualizer(km, k=(2, 11))
            socketio.emit('training', {'data': 'Optimal K Selection in Progress', 'progress': 40})
            socketio.sleep(3)
            visualizer.fit(_encoder.predict(x))
            optimal_k = visualizer.elbow_value_
            socketio.emit('training', {'data': 'Optimal K Selected', 'progress': 50})
            socketio.sleep(4)
            print("Optimal number of clusters (k):", optimal_k)

            socketio.emit('training', {'data': 'DEC Training Phase Started', 'progress': 55})
            socketio.sleep(5)
            dec = DEC(dims=dims, n_clusters=optimal_k, initializer=init)
            dec.compile(optimizer=optimizer, loss=loss)
            socketio.emit('training', {'data': 'DEC Training Phase in Progress', 'progress': 70})
            socketio.sleep(6)
            dec.fit(x, batch_size=batch_size, maxiter=maxiter, update_interval=update_interval, verbose=verbose)
            y_pred = dec.predict(x)
            socketio.emit('training', {'data': 'DEC Training Phase Completed', 'progress': 100})
            socketio.sleep(7)

            db_index = davies_bouldin_score(x, y_pred)
            print(f'Davies-Bouldin index for {optimal_k} clusters : ', db_index)

            segmented_image = y_pred.reshape(img.shape[:2])

            # Define colors for each label
            label_colors = {
                0: (60, 16, 152),  # Deep Purple
                1: (132, 41, 246),  # Lavender
                2: (110, 193, 228),  # Sky Blue
                3: (254, 221, 58),  # Sunflower Yellow
                4: (226, 169, 41),  # Amber
                5: (155, 155, 155),  # Dim Gray
                6: (32, 178, 170),  # Turquoise
                7: (255, 105, 180),  # Hot Pink
                8: (128, 0, 128),  # Purple
                9: (255, 140, 0),  # Orange
                10: (0, 128, 0)  # Green
            }

            # Create a mask for each unique value and apply the color
            colored_mask = np.zeros(segmented_image.shape + (3,), dtype=np.uint8)
            for label, rgb_color in label_colors.items():
                colored_mask[segmented_image == label] = rgb_color

            # Convert LAB image to base64 string
            _, encoded_img = cv2.imencode('.png', colored_mask)
            base64_image = base64.b64encode(encoded_img).decode('utf-8')

            return jsonify({'image': base64_image})

        else:
            return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        print('Error:', str(e))
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    socketio.run(app)
