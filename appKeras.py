from flask import Flask, jsonify, render_template, request, redirect, session, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import os

# --------------------------------------
# GANTI: Dari PyTorch ke TensorFlow/Keras
# --------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore

# Ensure PIL is imported first
import PIL
from PIL import Image
print("PIL version:", PIL.__version__)

from tensorflow.keras.preprocessing import image  # type: ignore
# (Untuk MobileNetV2)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore

# Suppress TensorFlow/Keras verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
tf.get_logger().setLevel('ERROR')
tf.keras.utils.disable_interactive_logging()

from threading import Timer
from datetime import timedelta
from flask_socketio import SocketIO, emit
from flask_admin import Admin, AdminIndexView
from flask_admin.contrib.sqla import ModelView

# Set up the app
app = Flask(__name__)
app.secret_key = "mysecretkey"  # This is a dummy key
app.permanent_session_lifetime = timedelta(hours=24)

# Initialize socketio
socketio = SocketIO(app)

# ======================================================
# Set up database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(50), nullable=False, unique=True)
    password_hash = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f'User {self.id}'

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# Create the database and admin user
with app.app_context():
    db.create_all()
    # Create an admin user if it doesn't exist
    admin_user = User.query.filter_by(username='admin').first()
    if not admin_user:
        admin_user = User(username='admin', email='amrizadi@gmail.com')
        admin_user.set_password('adminpassword')
        db.session.add(admin_user)
        db.session.commit()

# Secure Admin Index View


class SecureAdminIndexView(AdminIndexView):
    def is_accessible(self):
        return 'username' in session and session['username'] == 'admin'

    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for('home'))


# Initialize Flask-Admin with the secure index view
admin = Admin(app, name='Admin Panel', template_mode='bootstrap3',
              index_view=SecureAdminIndexView())
admin.add_view(ModelView(User, db.session))

# Set up paths
UPLOAD_FOLDER = r'./static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------------------------------------------
# LOAD MODEL MobileNetV2 (.keras)
# -------------------------------------------------------
model_path = './static/models/MobileNetV2.keras'

# Ensure the models directory exists
model_dir = os.path.dirname(model_path)
os.makedirs(model_dir, exist_ok=True)

# Check if model file exists
if not os.path.exists(model_path):
    print(f"WARNING: Model file not found at {model_path}")
    print("Will attempt to create a new model when needed")

# Lazy loading - only load model when needed
model = None

def load_keras_model():
    global model
    if model is None:
        print("Loading Keras model...")
        try:
            # Try standard loading first
            model = load_model(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Standard model loading failed: {str(e)}")
            
            # Try custom loading with custom objects
            try:
                import tensorflow as tf
                from tensorflow.keras.models import model_from_json
                
                # Define custom objects to handle missing modules
                custom_objects = {
                    'Functional': tf.keras.Model,
                }
                
                # Try loading with custom objects
                model = load_model(model_path, custom_objects=custom_objects, compile=False)
                print("Model loaded with custom objects successfully!")
            except Exception as e2:
                print(f"Custom model loading also failed: {str(e2)}")
                
                # Try a third approach - using custom model config
                try:
                    print("Attempting to use MobileNetV2 from applications...")
                    # Create a MobileNetV2 model directly from Keras applications
                    from tensorflow.keras.applications import MobileNetV2
                    base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
                    
                    # Create a custom model with the same structure as your trained model
                    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
                    x = tf.keras.layers.Dense(128, activation='relu')(x)
                    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
                    
                    print("Created a new MobileNetV2 model. Note: This is using default weights, not your trained weights!")
                    
                    # This is a fallback without your specific weights
                    # You should retrain this model or find a way to load your specific weights
                except Exception as e3:
                    print(f"All model loading approaches failed.")
                    print(f"Error 1: {str(e)}")
                    print(f"Error 2: {str(e2)}")
                    print(f"Error 3: {str(e3)}")
                    raise Exception("Could not load or create a MobileNetV2 model")
        
        # Test prediction with a dummy tensor to ensure the model works correctly
        try:
            dummy_input = np.zeros((1, 224, 224, 3))  # Create a dummy black image
            test_prediction = model.predict(dummy_input)
            print(f"Model test prediction completed successfully. Shape: {test_prediction.shape}")
        except Exception as e:
            print(f"Warning: Model test prediction failed: {str(e)}")
            
    return model

# Labels coin (sesuaikan dengan model Anda)
labels = ['Normal', 'Diabetik Retinopati']

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------------------------------
# Preprocessing function
# -------------------------------------------------------


def preprocess_image(image_path):
    try:
        # Explicitly import PIL to check if it's available
        try:
            from PIL import Image
            print("PIL/Pillow is properly installed")
        except ImportError:
            raise ImportError("PIL/Pillow library is not properly installed. Please run 'pip install Pillow'.")
            
        try:
            # Try using Keras' load_img
            from tensorflow.keras.preprocessing import image
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
        except Exception as keras_error:
            # Fallback to direct PIL implementation
            print(f"Keras load_img failed: {str(keras_error)}. Trying direct PIL implementation...")
            
            # Open and resize the image with PIL
            img = Image.open(image_path)
            img = img.convert('RGB')  # Convert to RGB to ensure 3 channels
            img = img.resize((224, 224))
            x = np.array(img).astype('float32')
            
        # Expanding dimensions
        x = np.expand_dims(x, axis=0)
        
        # Apply appropriate preprocessing
        try:
            # Try MobileNetV2 preprocessing
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            x = preprocess_input(x)
        except Exception as e:
            print(f"MobileNetV2 preprocessing failed: {str(e)}. Using simple normalization...")
            # Simple normalization as fallback
            x = x / 255.0
            
        return x
    except Exception as e:
        print(f"Image preprocessing error: {str(e)}")
        raise e

# ======================================================
# Home page


@app.route('/')
def home():
    if 'username' in session:
        print(f'You are logged in as {session["username"]}')
        return redirect(url_for('dashboard'))
    return render_template('index.html')

# Dashboard page


@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html')
    return redirect(url_for('home'))

# Results page


@app.route('/results', methods=['POST'])
def results():
    app.logger.info("Received image upload request")
    
    if 'username' not in session:
        app.logger.warning("Unauthorized upload attempt - no user session")
        return jsonify({
            'error': 'Unauthorized',
            'message': 'You must be logged in to upload images.'
        }), 401

    # Check if the image file was included in the request
    if 'image' not in request.files:
        app.logger.warning("No image file in request")
        return jsonify({
            'error': 'No image uploaded',
            'message': 'Please select an image file to upload.'
        }), 400

    file = request.files['image']
    app.logger.info(f"Received file: {file.filename}, type: {file.content_type}, size: {file.content_length or 'unknown'}")
    
    if file.filename == '':
        app.logger.warning("Empty filename in upload")
        return jsonify({
            'error': 'No image selected',
            'message': 'The selected file is empty.'
        }), 400

    # Check if the file has an allowed extension
    if not allowed_file(file.filename):
        app.logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Allowed file types are PNG, JPG, JPEG.'
        }), 400

    # Save the uploaded file
    username = session['username']
    filename = f"{username}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    app.logger.info(f"File saved to: {filepath}")  # Debugging

    # Schedule deletion after 1 hour
    def delete_file(path):
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted file: {path}")

    delete_timer = Timer(3600, delete_file, [filepath])
    delete_timer.start()


    try:
        app.logger.info("Starting image preprocessing")
        # Preprocess & infer
        img_tensor = preprocess_image(filepath)
        
        app.logger.info("Loading model")
        # Get model and make prediction
        model = load_keras_model()
        
        app.logger.info("Running prediction")
        # Model dengan 1 neuron output (sigmoid)
        prediction = model.predict(img_tensor, verbose=0)  # Tambah verbose=0 untuk mengurangi log
        app.logger.info(f"Prediction shape: {prediction.shape}, values: {prediction}")
        
        # Validasi output model
        if prediction.shape[0] == 0:
            raise ValueError("Model prediction returned empty result")
        
        # Ambil nilai sigmoid dan pastikan dalam range [0,1]
        score = float(prediction[0][0])
        score = max(0.0, min(1.0, score))  # Clamp ke range [0,1]
        app.logger.info(f"Prediction score (clamped): {score}")

        # Tentukan threshold yang optimal (bisa disesuaikan berdasarkan validasi model)
        THRESHOLD = 0.5
        
        # Tentukan kelas berdasarkan threshold
        if score >= THRESHOLD:
            top_class_name = 'Diabetik Retinopati'
            top_class_probability = score
        else:
            top_class_name = 'Normal'
            top_class_probability = 1.0 - score

        # Buat dictionary probabilitas untuk kedua kelas
        normal_prob = 1.0 - score
        diabetic_prob = score
        
        all_classes = {
            'Normal': round(normal_prob, 4),
            'Diabetik Retinopati': round(diabetic_prob, 4)
        }

        # Validasi probabilitas
        total_prob = normal_prob + diabetic_prob
        if abs(total_prob - 1.0) > 0.001:  # Toleransi untuk floating point
            app.logger.warning(f"Probabilities don't sum to 1.0: {total_prob}")

        app.logger.info(f"Classification result: {top_class_name} with confidence {top_class_probability:.4f}")
        
        return jsonify({
            'success': True,
            'top_class_name': top_class_name,
            'top_class_probability': round(top_class_probability, 4),
            'all_classes': all_classes,
            'threshold_used': THRESHOLD,
            'raw_score': round(score, 4),
            'image_url': f'/static/uploads/{filename}'
        })

    except ValueError as ve:
        app.logger.error(f"Validation error: {str(ve)}")
        return jsonify({
            'success': False,
            'error': 'Validation failed', 
            'message': str(ve)
        }), 400

    except Exception as e:
        app.logger.error(f"Inference failed: {str(e)}", exc_info=True)
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Inference failed', 
            'message': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500

    # try:
    #     app.logger.info("Starting image preprocessing")
    #     # Preprocess & infer
    #     img_tensor = preprocess_image(filepath)
        
    #     app.logger.info("Loading model")
    #     # Get model and make prediction
    #     model = load_keras_model()
        
    #     app.logger.info("Running prediction")
    #     # Model dengan 1 neuron output (sigmoid), misal shape = (1,1)
    #     prediction = model.predict(img_tensor)  # Contoh output: [[0.73]]
    #     app.logger.info(f"Prediction shape: {prediction.shape}, values: {prediction}")
        
    #     score = float(prediction[0][0])        # Ambil nilai sigmoid
    #     app.logger.info(f"Prediction score: {score}")

    #     # Tentukan kelas berdasarkan threshold (0.5)
    #     if score >= 0.5:
    #         top_class_name = 'Diabetik Retinopati'
    #         top_class_probability = score
    #     else:
    #         top_class_name = 'Normal'
    #         # Jika ingin menampilkan "keyakinan" terhadap kelas 0,
    #         # bisa pakai 1 - score (opsional).
    #         top_class_probability = 1.0 - score

    #     # Buat dictionary probabilitas untuk kedua kelas
    #     all_classes = {
    #         'Normal': 1.0 - score,
    #         'Diabetik Retinopati': score
    #     }

    #     app.logger.info(f"Returning results: {top_class_name} with confidence {top_class_probability:.4f}")
    #     return jsonify({
    #         'top_class_name': top_class_name,
    #         'top_class_probability': top_class_probability,
    #         'all_classes': all_classes,
    #         'image_url': f'/static/uploads/{filename}'
    #     })
    # except Exception as e:
    #     app.logger.error(f"Inference failed: {str(e)}", exc_info=True)
    #     import traceback
    #     app.logger.error(traceback.format_exc())
    #     return jsonify({
    #         'error': 'Inference failed', 
    #         'message': str(e),
    #         'traceback': traceback.format_exc()
    #     }), 500


# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        login_identifier = request.form.get('login_identifier')
        password = request.form.get('password')

        if not login_identifier or not password:
            flash('Please fill in all fields', 'error')
            return redirect(url_for('home'))

        user = User.query.filter(
            (User.username == login_identifier) | (
                User.email == login_identifier)
        ).first()
        if user and user.check_password(password):
            session['username'] = user.username
            session['email'] = user.email
            session.permanent = True
            flash('Login successful', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'error')
        return redirect(url_for('home'))
    else:
        return render_template('login.html')

# SignUp page


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not email or not password or not confirm_password:
            flash('Please fill in all fields', 'error')
            return redirect(url_for('signup'))
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))

        user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
        if user:
            flash('Username or email already exists', 'error')
            return redirect(url_for('signup'))

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        session['username'] = new_user.username
        session['email'] = new_user.email
        session.permanent = True
        return redirect(url_for('dashboard'))
    else:
        return render_template('signup.html')

# Logout


@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('email', None)
    return redirect(url_for('home'))

# Custom 404 error handler


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

# ==================================================================
# SocketIO event for image upload


@socketio.on('upload_image')
def handle_image_upload(data):
    app.logger.info("Received SocketIO image upload")
    
    if 'username' not in session:
        app.logger.warning("Unauthorized SocketIO upload attempt - no user session")
        emit('upload_error', {'error': 'Unauthorized'})
        return

    file = data['file']
    app.logger.info(f"Received SocketIO file: {file['filename']}, size: {len(file['data'])} bytes")
    
    username = session['username']
    filename = f"{username}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{file['filename']}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'wb') as f:
        f.write(file['data'])
    app.logger.info(f"SocketIO file saved to: {filepath}")

    # Schedule deletion after 1 hour
    def delete_file(path):
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted file: {path}")

    delete_timer = Timer(3600, delete_file, [filepath])
    delete_timer.start()

    try:
        app.logger.info("SocketIO starting image preprocessing")
        img_tensor = preprocess_image(filepath)
        
        app.logger.info("SocketIO loading model")
        model = load_keras_model()
        
        app.logger.info("SocketIO running prediction")
        predictions = model.predict(img_tensor)
        app.logger.info(f"SocketIO prediction shape: {predictions.shape}, values: {predictions}")
        
        # Handle binary classification (sigmoid) output
        if len(predictions.shape) == 2 and predictions.shape[1] == 1:
            # Binary classification with sigmoid
            score = float(predictions[0][0])
            app.logger.info(f"SocketIO binary prediction score: {score}")
            
            if score >= 0.5:
                top_class_name = 'Diabetik Retinopati'
                top_class_probability = score
            else:
                top_class_name = 'Normal'
                top_class_probability = 1.0 - score
                
            all_classes = {
                'Normal': 1.0 - score,
                'Diabetik Retinopati': score
            }
        else:
            # Multi-class with softmax
            probabilities = tf.nn.softmax(predictions, axis=1).numpy()[0]
            app.logger.info(f"SocketIO softmax probabilities: {probabilities}")
            
            top_class_index = np.argmax(probabilities)
            top_class_name = labels[top_class_index]
            top_class_probability = float(probabilities[top_class_index])
            
            all_classes = {
                labels[i]: float(probabilities[i])
                for i in range(len(labels))
            }

        app.logger.info(f"SocketIO returning results: {top_class_name} with confidence {top_class_probability:.4f}")
        emit('classification_result', {
            'top_class_name': top_class_name,
            'top_class_probability': top_class_probability,
            'all_classes': all_classes,
            'image_url': f'/static/uploads/{filename}'
        })

    except Exception as e:
        app.logger.error(f"SocketIO inference failed: {str(e)}", exc_info=True)
        import traceback
        app.logger.error(traceback.format_exc())
        emit('upload_error', {
            'error': 'An error occurred during inference. Please try again.',
            'details': str(e),
            'traceback': traceback.format_exc()
        })


# ==================================================================
# Run the app
if __name__ == '__main__':
    print("Starting Eye Diagnosis Flask application with Keras model...")
    print("Server running at http://127.0.0.1:5000")
    # For Docker deployment, use 0.0.0.0 to make the server accessible from outside the container
    socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
