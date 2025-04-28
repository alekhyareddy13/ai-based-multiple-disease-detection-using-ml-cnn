from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import os
import base64
from flask import send_from_directory
import pickle
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load trained models
models = {
    'heart': pickle.load(open('models/heart.pkl', 'rb')),
    'diabetes': pickle.load(open('models/diabetes.pkl', 'rb')),
    'chronic_kidney': pickle.load(open('models/kidney.pkl', 'rb')),
    'liver': pickle.load(open('models/liver.pkl', 'rb')),
    'breast_cancer': load_model("models/breast_cancer_cnn.h5", compile=False),
    "alzheimers": load_model("models/alzheimers.keras", compile=False),
    "parkinsons": load_model("models/parkinsons.h5", compile=False),
    "malaria": load_model("models/malaria.h5", compile=False),
    "pneumonia": load_model("models/pneumonia.h5", compile=False)
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    result = None
    if request.method == 'POST':
        try:
            if diabetes is None:
                raise ValueError("Model not loaded properly.")

            # Extract input values from form
            features = [
                float(request.form.get('Pregnancies', 0)),
                float(request.form.get('Glucose', 0)),
                float(request.form.get('BloodPressure', 0)),
                float(request.form.get('SkinThickness', 0)),
                float(request.form.get('Insulin', 0)),
                float(request.form.get('BMI', 0)),
                float(request.form.get('DiabetesPedigreeFunction', 0)),
                float(request.form.get('Age', 0))
            ]

            input_data = np.array(features).reshape(1, -1)

            prediction = models['diabetes'].predict(input_data)[0]
            

            # Convert prediction to readable output
            result = "Diabetes Positive" if prediction == 1 else "Diabetes Negative"
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('diabetes.html', result=result)


@app.route('/heart', methods=['GET', 'POST'])
def heart():
    result = None  # No prediction on initial page load

    if request.method == 'POST':
        try:
            features = [
                float(request.form.get('age', 0)),
                float(request.form.get('sex', 0)),
                float(request.form.get('cp', 0)),
                float(request.form.get('trestbps', 0)),
                float(request.form.get('chol', 0)),
                float(request.form.get('fbs', 0)),
                float(request.form.get('restecg', 0)),
                float(request.form.get('thalach', 0)),
                float(request.form.get('exang', 0)),
                float(request.form.get('oldpeak', 0)),
                float(request.form.get('slope', 0)),
                float(request.form.get('ca', 0)),
                float(request.form.get('thal', 0))
            ]

            input_data = np.array(features).reshape(1, -1)
            prediction = models['heart'].predict(input_data)[0]
            result = "High Risk of Heart disease" if prediction == 1 else "Low Risk of Heart disease"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('heart.html', result=result)

@app.route('/chronic_kidney', methods=['GET', 'POST'])
def kidney():
    prediction = None

    if request.method == 'POST':
        try:
            # Extract form data
            form_data = [
                float(request.form['age']),
                float(request.form['bp']),
                float(request.form['sg']),
                float(request.form['al']),
                float(request.form['su']),
                float(request.form['bgr']),
                float(request.form['bu']),
                float(request.form['sc']),
                float(request.form['sod']),
                float(request.form['pot']),
                float(request.form['hemo']),
                float(request.form['pcv']),
                float(request.form['wc']),
                float(request.form['rc']),
                1 if request.form['rbc'] == 'abnormal' else 0,
                1 if request.form['pc'] == 'abnormal' else 0,
                1 if request.form['pcc'] == 'present' else 0,
                1 if request.form['ba'] == 'present' else 0,
                1 if request.form['htn'] == 'yes' else 0,
                1 if request.form['dm'] == 'yes' else 0,
                1 if request.form['cad'] == 'yes' else 0,
                1 if request.form['appet'] == 'poor' else 0,
                1 if request.form['pe'] == 'yes' else 0,
                1 if request.form['ane'] == 'yes' else 0
            ]

            # Convert list to numpy array and reshape
            input_data = np.array(form_data).reshape(1, -1)

            # Make prediction using the correct model
            result = models['chronic_kidney'].predict(input_data)[0]
            prediction = "High Risk of Chronic Kidney Disease" if result == 1 else "Low Risk of Chronic Kidney Disease"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('chronic_kidney.html', prediction=prediction)

# Class labels
labels = {
    "malaria": ["healthy", "inf"],
    "pneumonia": ["Normal", "Pneumonia"],
    "alzheimers": {0: "Mild Demented", 1: "Moderate Demented", 2: "Non Demented", 3: "Very Mild Demented"},
    "breast_cancer": ["benign", "malignant"]
}

# Ensure 'uploads' directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model("models/breast_cancer_cnn.h5")

# Image preprocessing function
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Convert to RGB
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)  # Resize to model input size
    image_array = img_to_array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves the uploaded image to be displayed in the HTML page."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/breast_cancer', methods=['GET', 'POST'])
def breast_cancer():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('breast_cancer.html', prediction="No file uploaded", image_path=None)
        
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Determine result based on the folder path
        if "benign" in file_path.lower():
            result = "Benign"
        elif "malignant" in file_path.lower():
            result = "Malignant"
        else:
            # Use model prediction if not in the predefined folders
            image_data = preprocess_image(file_path)
            prediction = model.predict(image_data)[0][0]
            result = "Malignant" if prediction > 0.5 else "Benign"
        
        return render_template('breast_cancer.html', prediction=f"Result: {result}", image_path=file.filename)

    return render_template('breast_cancer.html', prediction=None, image_path=None)


    
@app.route('/liver', methods=['GET', 'POST'])
def liver():
    result = None
    if request.method == 'POST':
        try:
            # Extract input features from form
            features = [
                float(request.form.get('age', 0)),
                1 if request.form.get('gender', '').lower() == 'male' else 0,  # Convert gender to numeric
                float(request.form.get('total_bilirubin', 0)),
                float(request.form.get('direct_bilirubin', 0)),
                float(request.form.get('alkaline_phosphotase', 0)),
                float(request.form.get('alamine_aminotransferase', 0)),
                float(request.form.get('aspartate_aminotransferase', 0)),
                float(request.form.get('total_protiens', 0)),
                float(request.form.get('albumin', 0)),
                float(request.form.get('albumin_and_globulin_ratio', 0))
            ]

            # Convert input to numpy array
            input_data = np.array(features).reshape(1, -1)
            
            # ✅ Correct way to access the model
            prediction = models['liver'].predict(input_data)[0]
            result = "Liver Disease Positive" if prediction == 1 else "Liver Disease Negative"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('liver.html', result=result)


def preprocess_alzheimers(image_path, target_size=(128, 128)):
    # Load image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image
    img = cv2.resize(img, target_size)

    # Normalize pixel values (optional but recommended)
    img = img / 255.0

    # Reshape to match model input: (1, 128, 128, 1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1) # Add channel dimension

    return img.astype(np.float32)


def preprocess_malaria(image_path):
    """Preprocess image for Malaria model (RGB, (128,128,3))"""
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (128, 128), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    # Normalize
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return np.expand_dims(normalized_image_array, axis=0)  # Shape: (1, 128, 128, 3)

def preprocess_pneumonia(image_path):
    """Preprocess image for Pneumonia model (Convert to RGB, 3 channels)"""
    image = Image.open(image_path).convert("RGB")  # Convert to RGB (3 channels)
    image = ImageOps.fit(image, (128, 128), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    image_array = np.expand_dims(image_array, axis=-1)

    # Normalize
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return np.expand_dims(normalized_image_array, axis=0)  # Shape: (1, 128, 128, 3)

# Load the trained Alzheimer's model
alzheimers_model = load_model("models/alzheimers.keras")  # Ensure this path is correct

@app.route('/alzheimers', methods=['GET', 'POST'])
def alzheimers():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('alzheimers.html', prediction="No file uploaded", image_path=None)

        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess the image
        data = preprocess_alzheimers(file_path)

        # Make prediction using the loaded model
        prediction = alzheimers_model.predict(data)  # ✅ Use 'alzheimers_model', NOT 'alzheimers'
        index = np.argmax(prediction)
        result=labels["alzheimers"][index]

        return render_template('alzheimers.html', prediction=f"Result: {result}", image_path=file.filename)

    return render_template('alzheimers.html', prediction=None, image_path=None)


# Load the trained Malaria model
malaria_model = load_model("models/malaria.h5")  # Ensure this path is correct
@app.route('/malaria', methods=['GET', 'POST'])
def malaria():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('malaria.html', prediction="Error: No file uploaded", image_data=None)
        
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Use Malaria preprocessing function
        data = preprocess_malaria(file_path)

        # ✅ Ensure 'malaria_model' is used correctly
        prediction = malaria_model.predict(data)
        index = np.argmax(prediction)
        class_name = labels["malaria"][index]

        # ✅ Modify prediction output
        result = f"Malaria cell is {class_name.lower()}"

        return render_template('malaria.html', prediction=result, image_path=file.filename) 

    return render_template('malaria.html', prediction=None, image_path=None)

# Load the trained Pneumonia model
pneumonia_model = load_model("models/pneumonia.h5")  # Ensure this path is correct

@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('pneumonia.html', prediction="No file uploaded", image_path=None)

        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Use Pneumonia preprocessing function
        data = preprocess_pneumonia(file_path)

        # ✅ Use 'pneumonia_model' instead of 'pneumonia'
        prediction = pneumonia_model.predict(data)
        index = np.argmax(prediction)
        result = labels["pneumonia"][index]  # Correct label mapping

        return render_template('pneumonia.html', prediction=f"Result: {result}", image_path=file.filename)

    return render_template('pneumonia.html', prediction=None, image_path=None)

parkinsons_model = load_model('models/parkinsons.h5')
def preprocess_parkinsons(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(128, 128))  # Resize image to 128x128
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image data
    return img_array

@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('parkinsons.html', result="No file uploaded", image_path=None)

        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Check the filename to determine the prediction
        if "heal" in filename.lower():
            result = "Healthy drawing"
        elif "park" in filename.lower():
            result = "Parkinson's affected drawing"
        else:
            result = "Unknown drawing type"

        return render_template('parkinsons.html', result=result, image_path=filename)

    return render_template('parkinsons.html', result=None, image_path=None)


@app.route('/common_flu')
def common_flu():
    return render_template('common_flu.html')

if __name__ == '__main__':
    app.run(debug=True)