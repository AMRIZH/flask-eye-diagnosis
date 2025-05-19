# **Eye Diagnosis Flask Application**

A sophisticated Flask web application for diagnosing eye conditions using machine learning. This application supports multiple model backends (Keras, ResNet, and YOLO) and provides a clean, user-friendly interface for uploading and analyzing eye images.

---

## **Table of Contents**
1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Model Selection](#model-selection)
5. [Running the Application](#running-the-application)
6. [Docker Deployment](#docker-deployment)
7. [Project Structure](#project-structure)
8. [User Guide](#user-guide)
9. [Troubleshooting](#troubleshooting)

---

## **Features**

- **Multiple Model Backends**: Choose between Keras (MobileNetV2), ResNet, or YOLO for image classification
- **User Authentication**: Secure login/signup system with admin capabilities
- **Modern UI**: Clean, responsive interface with medical theming using Tailwind CSS
- **Real-time Analysis**: Upload images and receive instant diagnostic results
- **Admin Panel**: Manage users and system settings (accessible to admin users)
- **Docker Support**: Easy deployment with Docker and docker-compose

---

## **Prerequisites**

Before you begin, ensure you have the following installed:

1. **Python 3.8+**:
   - Download Python from the [official website](https://www.python.org/downloads/).
   - Follow the installation instructions for your operating system.

2. **Pip (Python Package Installer)**:
   - Usually included with Python installation.
   - Verify installation with `pip --version`.

3. **Git (Optional)**:
   - For cloning the repository.
   - [Download Git](https://git-scm.com/downloads).

4. **Docker & Docker Compose (Optional)**:
   - For containerized deployment.
   - [Install Docker](https://docs.docker.com/get-docker/).
   - [Install Docker Compose](https://docs.docker.com/compose/install/).

---

## **Installation**

### **1. Clone or Download the Repository**
```bash
git clone <repository-url>
cd flask_pakhelmi
```

### **2. Create and Activate Virtual Environment**
It's recommended to use a virtual environment to avoid conflicts with other projects:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### **3. Install Dependencies**

The application has three different model backends, each with its own requirements file:

- For Keras (MobileNetV2) backend:
  ```bash
  pip install -r requirements_keras.txt
  ```

- For ResNet backend:
  ```bash
  pip install -r requirements_resnet.txt
  ```

- For YOLO backend:
  ```bash
  pip install -r requirements_yolo.txt
  ```

---

## **Model Selection**

This application supports three different machine learning backends for eye diagnosis:

### **1. Keras (MobileNetV2)**
- Lightweight and fast model optimized for mobile and embedded applications
- Binary classification: Normal vs. Diabetic Retinopathy
- Use `appKeras.py` to run this version

### **2. ResNet**
- Deep residual network with high accuracy
- Binary classification: Normal vs. Diabetic Retinopathy
- Use `appResnet.py` to run this version

### **3. YOLO (You Only Look Once)**
- Object detection model capable of identifying multiple eye conditions
- Can detect and localize features in eye images
- Use `appyolo.py` to run this version

### **Model Files**
Place your model files in the appropriate location:
- Keras: `./static/models/MobileNetV2.keras`
- ResNet: `./static/models/resnet.pth`
- YOLO: `./static/models/yolov5.pt`

---

## **Running the Application**

### **Running the Keras Version**
```bash
python appKeras.py
```

### **Running the ResNet Version**
```bash
python appResnet.py
```

### **Running the YOLO Version**
```bash
python appyolo.py
```

By default, the application will run on `http://127.0.0.1:5000`.

---

## **Docker Deployment**

The application can be easily deployed using Docker:

### **1. Building and Running with Docker Compose**
```bash
# Make the deployment script executable
chmod +x deploy.sh

# Deploy the application
./deploy.sh --deploy
```

### **2. Manual Docker Commands**
```bash
# Build the Docker image
docker-compose build

# Start the container
docker-compose up -d

# View logs
docker logs eye-diagnosis
```

### **3. VPS Deployment**
For detailed instructions on deploying to a VPS, refer to the [DEPLOY.md](DEPLOY.md) file.

---

## **Project Structure**

```
flask_pakhelmi/
├── static/
│   ├── uploads/              # Uploaded eye images are stored here
│   ├── models/               # ML model files (Keras, ResNet, YOLO)
│   ├── icons/                # UI icons
│   └── styles.css            # CSS styles (with Tailwind)
│
├── templates/
│   ├── base.html             # Base template with common layout
│   ├── index.html            # Home/landing page
│   ├── login.html            # User login page
│   ├── signup.html           # User registration page
│   ├── dashboard.html        # Main application dashboard
│   └── 404.html              # Custom 404 error page
│
├── appKeras.py               # Main Flask application using Keras backend
├── appResnet.py              # Main Flask application using ResNet backend
├── appyolo.py                # Main Flask application using YOLO backend
│
├── requirements_keras.txt    # Dependencies for Keras version
├── requirements_resnet.txt   # Dependencies for ResNet version
├── requirements_yolo.txt     # Dependencies for YOLO version
│
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose configuration
├── deploy.sh                 # Deployment helper script
├── .dockerignore             # Files to exclude from Docker build
│
└── README.md                 # This documentation file
```

---

## **User Guide**

### **1. Registration and Login**
- Navigate to the application's home page
- Click "Sign Up" to create a new account or "Login" if you already have one
- Admin user is created by default (username: admin, password: adminpassword)

### **2. Dashboard**
- After logging in, you'll be redirected to the dashboard
- The dashboard is divided into two sections:
  - Left section: Image upload area
  - Right section: Analysis results display area

### **3. Uploading and Analyzing Images**
- Drag and drop an eye image onto the upload area or click "browse" to select a file
- The application will automatically process the image and display:
  - The uploaded image
  - The most likely diagnosis
  - Confidence scores for each possible condition

### **4. Administrator Features**
- Admin users can access the admin panel at `/admin`
- The admin panel allows management of:
  - User accounts
  - System configuration

---

## **Troubleshooting**

### **1. Model Loading Issues**
- Ensure you have the correct model file in the `static/models/` directory
- For Keras version:
  - Verify that MobileNetV2.keras exists in the models directory
  - If using an older model format, the application includes fallback methods for loading

### **2. Image Upload Failures**
- Ensure the directory `static/uploads/` exists and is writable
- Check that the uploaded file is a valid image (JPG, PNG, JPEG format)
- The maximum file size is 10MB

### **3. Database Issues**
- If user login/registration isn't working, the SQLite database may be corrupted
- Delete the `users.db` file and restart the application to recreate it
- The default admin user will be recreated automatically

### **4. Docker Deployment Issues**
- Check container logs: `docker logs eye-diagnosis-app`
- Ensure ports are properly mapped in the `docker-compose.yml` file
- Make sure the model files are accessible to the Docker container

---

## **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

---

## **License**

This project is licensed under the MIT License - see the LICENSE file for details.
