from flask import Flask, jsonify, render_template, request, redirect, session, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import os
from ultralytics import YOLO
from threading import Timer
import time
from datetime import timedelta
from flask_socketio import SocketIO, emit
from flask_admin import Admin, AdminIndexView
from flask_admin.contrib.sqla import ModelView

# set up the app
app = Flask(__name__)
app.secret_key = "mysecretkey"  # this is dummy key
app.permanent_session_lifetime = timedelta(
    hours=24)  # Set session lifetime to 24 hours

# initialize socketio
socketio = SocketIO(app)


# ======================================================
# set up database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# user model


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(50), nullable=False, unique=True)
    password_hash = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f'User {self.id}'

    def set_password(self, password):  # hash the password
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):  # check the password
        return check_password_hash(self.password_hash, password)


# Create the database and admin user
with app.app_context():
    db.create_all()
    # Create an admin user if it doesn't exist
    admin_user = User.query.filter_by(username='admin').first()
    if not admin_user:
        admin_user = User(username='admin', email='amrizadi@gmail.com')
        admin_user.set_password('adminpassword')  # Set a secure password
        db.session.add(admin_user)
        db.session.commit()


# Secure Admin Index View
class SecureAdminIndexView(AdminIndexView):
    def is_accessible(self):
        # Only allow access if the user is logged in and is an admin
        return 'username' in session and session['username'] == 'admin'

    def inaccessible_callback(self, name, **kwargs):
        # Redirect to the home page if the user is not allowed
        return redirect(url_for('home'))


# Initialize Flask-Admin with the secure index view
admin = Admin(app, name='Admin Panel', template_mode='bootstrap3',
              index_view=SecureAdminIndexView())

# Add the User model to the admin interface
admin.add_view(ModelView(User, db.session))

# set up paths
UPLOAD_FOLDER = r'./static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO classification model
model = YOLO(r'./static/models/coin224_150e-cls.pt')
# model = YOLO(r'./static/models/SIBI640_100e-cls.pt')

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ======================================================
# home page


@app.route('/')
def home():
    if 'username' in session:
        print(f'You are logged in as {session["username"]}')
        return redirect(url_for('dashboard'))
    return render_template('index.html')

# dashboard page


@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html')
    return redirect(url_for('home'))


@app.route('/results', methods=['POST'])
def results():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized', 'message': 'You must be logged in to upload images.'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded', 'message': 'Please select an image file to upload.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected', 'message': 'The selected file is empty.'}), 400

    # Check if the file has an allowed extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type', 'message': 'Allowed file types are PNG, JPG, JPEG.'}), 400

    # Save the uploaded file with a unique name
    username = session['username']
    filename = f"{username}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print(f"File saved to: {filepath}")  # Debugging

    # Schedule deletion of the file after 1 hour
    def delete_file(path):
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted file: {path}")

    delete_timer = Timer(3600, delete_file, [filepath])
    delete_timer.start()

    try:
        # Run inference using YOLO
        results = model.predict(source=filepath)
        probs = results[0].probs

        # Get top-1 class and confidence
        top_class_index = probs.top1
        top_class_name = results[0].names[top_class_index]
        top_class_probability = float(probs.top1conf)

        # Get all class probabilities
        all_classes = {
            results[0].names[index]: float(prob)
            for index, prob in enumerate(probs.data.cpu().numpy())
        }

        # Return the results
        return jsonify({
            'top_class_name': top_class_name,
            'top_class_probability': top_class_probability,
            'all_classes': all_classes,
            'image_url': f'/static/uploads/{filename}'
        })
    except Exception as e:
        app.logger.error(f"Inference failed: {e}")
        return jsonify({'error': 'Inference failed', 'message': str(e)}), 500


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

        user = User.query.filter((User.username == login_identifier) | (
            User.email == login_identifier)).first()
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
        # Handle form submission
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Validate inputs
        if not username or not email or not password or not confirm_password:
            flash('Please fill in all fields', 'error')
            return redirect(url_for('signup'))
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))

        # Check if user already exists
        user = User.query.filter(
            (User.username == username) | (User.email == email)).first()
        if user:
            flash('Username or email already exists', 'error')
            return redirect(url_for('signup'))

        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        # Log in the new user
        session['username'] = new_user.username
        session['email'] = new_user.email
        session.permanent = True
        return redirect(url_for('dashboard'))
    else:
        return render_template('signup.html')

# logout


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
    if 'username' not in session:
        emit('upload_error', {'error': 'Unauthorized'})
        return

    # Save the uploaded file
    file = data['file']
    username = session['username']
    filename = f"{username}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{file['filename']}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'wb') as f:
        f.write(file['data'])

    # Schedule deletion of the file after 1 min
    def delete_file(path):
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted file: {path}")

    delete_timer = Timer(3600, delete_file, [filepath])
    delete_timer.start()

    try:
        # Run inference using YOLO
        results = model.predict(source=filepath)
        probs = results[0].probs

        # Get top-1 class and confidence
        top_class_index = probs.top1
        top_class_name = results[0].names[top_class_index]
        top_class_probability = float(probs.top1conf)

        # Get all class probabilities
        all_classes = {
            results[0].names[index]: float(prob)
            for index, prob in enumerate(probs.data.cpu().numpy())
        }

        # Emit the results to the client
        emit('classification_result', {
            'top_class_name': top_class_name,
            'top_class_probability': top_class_probability,
            'all_classes': all_classes,
            'image_url': f'/static/uploads/{filename}'
        })

    except Exception as e:
        app.logger.error(f"Inference failed: {e}")
        emit('upload_error', {
             'error': 'An error occurred during inference. Please try again.'})


# ==================================================================
# run the app
if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
