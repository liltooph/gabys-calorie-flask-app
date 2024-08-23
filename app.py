from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
import requests

# Initialize the Flask app
app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
