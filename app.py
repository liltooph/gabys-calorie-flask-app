from flask import Flask, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
import requests
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

POSTGRES_USERNAME = os.getenv("SQL_SERVER_USERNAME")
POSTGRES_PASSWORD = os.getenv("SQL_SERVER_PASSWORD")
POSTGRES_DATABASE_NAME = os.getenv("SQL_SERVER_DATABASE_NAME")

# Configurations
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql+psycopg2://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@localhost/{POSTGRES_DATABASE_NAME}'
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User Model
class User(db.Model, UserMixin):
    __tablename__ = 'calorieappusers'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

    def __init__(self, username, password):
        self.username = username
        self.password = password

# User Loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load the YOLOv8 model (you can replace 'yolov8n.pt' with your trained model)
model = YOLO('yolov8n.pt')  # YOLOv8n is the nano version, replace with your model if needed
model.to('cuda')

def query_ollama(prompt):
    url = "http://localhost:11434/api/generate"  # Replace with your Ollama API endpoint
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama2",
        "prompt": prompt,
        "stream": False,
    }

    print("Starting ollama query")
    try:
        print("Ollama is working!!")
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Ollama errored!")
        return {"error": str(e)}

def process_image(img):
    # Convert image to OpenCV format (BGR)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Perform inference with YOLOv8
    results = model(img)
    
    # Prepare predictions in JSON format
    predictions = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = box.cls[0].item()  # Class index
            class_name = model.names[int(cls)]  # Class name
            
            predictions.append({
                'class': class_name,
                'confidence': conf,
                'box': [x1, y1, x2, y2]
            })
    
    return predictions

@app.route('/upload', methods=['POST'])
def predict():
    try:
        if 'file' in request.files:
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({'error': 'No file selected for uploading'}), 400
            
            # Load image using PIL
            img = Image.open(file)
            
        elif 'image' in request.json:
            # Decode the base64 string
            image_data = request.json['image']
            image_data = base64.b64decode(image_data)
            img = Image.open(BytesIO(image_data))
        
        else:
            return jsonify({'error': 'No image data provided'}), 400

        # Process the image and perform YOLO inference
        predictions = process_image(img)
        #return_predctions = predictions
        # Extract class names from the predictions
        class_names = [item['class'] for item in predictions if 'class' in item]
        names = " ".join(class_names)

        #return jsonify(predictions)

        # Query Ollama for nutritional information based on class names
        ollama_response = query_ollama(f"What's some nutritional info about {names} such as calories")

        #return jsonify(predictions)

        # Combine predictions and Ollama response into one JSON response
        response = {
            "predictions": predictions,
            "ollama_response": ollama_response
        }

        return jsonify(response)
   
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Function to validate username
def validate_username(username):
    # Check if the username is between 3 to 16 characters
    if len(username) < 3 or len(username) > 16:
        return False, "length-issue"
    # Check if the username contains only alphanumeric characters (no special characters)
    if not re.match("^[A-Za-z0-9]+$", username):
        return False, "character-issue"
    return True, "no-issue"

# Function to validate password
def validate_password(password):
    # Check if the password is between 4 to 50 characters
    return 4 <= len(password) <= 50

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # Validate email
    ##try:
        # This will validate the email and also check the MX record
    ##   valid = validate_email(email, check_deliverability=True)
    ##    email = valid.email
    ##except EmailNotValidError as e:
    ##    return jsonify({'message': str(e), 'status': 'fail'})

    # Validate username
    user_is_valid, issue = validate_username(username)
    if not user_is_valid and issue == 'length-issue':
        return jsonify({'message': 'Username must be between 3 to 16 characters.', 'status': 'fail'})
    elif not user_is_valid and issue == 'character-issue':
        return jsonify({'message': 'Username must contain no special characters.', 'status': 'fail'})


    # Validate password
    if not validate_password(password):
        return jsonify({'message': 'Password must be between 4 to 50 characters in length.', 'status': 'fail'})

    # You gotta put something here to interact with the sql server
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({'message': 'Username already exists. Please choose a different one.', 'status': 'fail'})

    # Hash the password and save the user to the database
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    # If all validations pass

    print("TEST: " + hashed_password)

    return jsonify({'message': 'Registration successful! You can now log in.', 'status': 'success'})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()

    print("LOGIN TEST: " + password)
    print("HASH LOGIN TEST: " + user.password)

    if user and bcrypt.check_password_hash(user.password, password):
        login_user(user)
        return jsonify({'message': 'Login successful!', 'status': 'success'})
    else:
        return jsonify({'message': 'Invalid username or password', 'status': 'fail'}), 401

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
