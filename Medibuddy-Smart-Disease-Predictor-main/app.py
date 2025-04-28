import os
import io
import base64
from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

app = Flask(__name__)

# Healthy average values
DIABETES_HEALTHY_AVERAGES = {
    'Pregnancies': 0,
    'Glucose': 90,
    'BloodPressure': 80,
    'SkinThickness': 20,
    'Insulin': 80,
    'BMI': 22,
    'DiabetesPedigreeFunction': 0.5,
    'Age': 30
}

HEART_HEALTHY_AVERAGES = {
    'age': 50,
    'sex': 1,                # 1 = male, 0 = female
    'cp': 0,                 # Chest pain type (0 = typical angina)
    'trestbps': 120,         # Resting blood pressure (mm Hg)
    'chol': 200,             # Cholesterol (mg/dl)
    'fbs': 0,                # Fasting blood sugar <120 (0 = false)
    'restecg': 0,            # Resting ECG (0 = normal)
    'thalach': 150,          # Max heart rate achieved
    'exang': 0,              # Exercise induced angina (0 = no)
    'oldpeak': 0.0,          # ST depression induced by exercise
    'slope': 1,              # Slope of peak exercise ST segment
    'ca': 0,                 # Major vessels colored by flourosopy
    'thal': 2                # Thalassemia (2 = normal)
}

def generate_diabetes_comparison_chart(user_values):
    """Generate diabetes comparison chart"""
    categories = list(DIABETES_HEALTHY_AVERAGES.keys())
    user_data = [user_values.get(k, 0) for k in categories]
    healthy_data = list(DIABETES_HEALTHY_AVERAGES.values())
    
    plt.figure(figsize=(12, 6))
    x = range(len(categories))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], user_data, width, label='Your Values', color='#ff7f0e')
    plt.bar([i + width/2 for i in x], healthy_data, width, label='Healthy Average', color='#1f77b4')
    
    plt.xticks(x, categories, rotation=45)
    plt.ylabel('Values')
    plt.title('Your Diabetes Metrics vs Healthy Averages')
    plt.legend()
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{plot_url}"

def generate_heart_comparison_chart(user_values):
    """Generate heart disease comparison chart"""
    try:
        categories = [
            'Age', 'Sex', 'Chest Pain', 
            'Blood Pressure', 'Cholesterol', 'Fasting BS',
            'Resting ECG', 'Max HR', 'Exercise Angina',
            'ST Depression', 'Slope', 'Major Vessels', 'Thalassemia'
        ]
        
        heart_keys = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                     'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        user_data = [user_values.get(k, 0) for k in heart_keys]
        healthy_data = [HEART_HEALTHY_AVERAGES[k] for k in heart_keys]

        plt.figure(figsize=(14, 7))
        x = range(len(categories))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], user_data, width, label='Your Values', color='#ff7f0e')
        plt.bar([i + width/2 for i in x], healthy_data, width, label='Healthy Average', color='#1f77b4')

        plt.xticks(x, categories, rotation=45, ha='right')
        plt.ylabel('Values')
        plt.title('Your Heart Health vs Healthy Averages')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_url}"
    except Exception as e:
        print(f"Heart chart error: {str(e)}")
        return None

def predict(values, dic):
    if len(values) == 8:  # Diabetes
        dic2 = {
            'NewBMI_Obesity 1': 0, 'NewBMI_Obesity 2': 0, 'NewBMI_Obesity 3': 0,
            'NewBMI_Overweight': 0, 'NewBMI_Underweight': 0,
            'NewInsulinScore_Normal': 0, 'NewGlucose_Low': 0,
            'NewGlucose_Normal': 0, 'NewGlucose_Overweight': 0, 'NewGlucose_Secret': 0
        }

        if dic['BMI'] <= 18.5:
            dic2['NewBMI_Underweight'] = 1
        elif 18.5 < dic['BMI'] <= 24.9:
            pass
        elif 24.9 < dic['BMI'] <= 29.9:
            dic2['NewBMI_Overweight'] = 1
        elif 29.9 < dic['BMI'] <= 34.9:
            dic2['NewBMI_Obesity 1'] = 1
        elif 34.9 < dic['BMI'] <= 39.9:
            dic2['NewBMI_Obesity 2'] = 1
        else:
            dic2['NewBMI_Obesity 3'] = 1

        if 16 <= dic['Insulin'] <= 166:
            dic2['NewInsulinScore_Normal'] = 1

        glucose = dic['Glucose']
        if glucose <= 70:
            dic2['NewGlucose_Low'] = 1
        elif 70 < glucose <= 99:
            dic2['NewGlucose_Normal'] = 1
        elif 99 < glucose <= 126:
            dic2['NewGlucose_Overweight'] = 1
        else:
            dic2['NewGlucose_Secret'] = 1

        dic.update(dic2)
        final_features = list(map(float, list(dic.values())))

        model = pickle.load(open('models/diabetes.pkl','rb'))
        proba = model.predict_proba(np.array(final_features).reshape(1, -1))[0]
        return 1 if proba[1] > 0.5 else 0

    elif len(values) == 13:  # Heart disease
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    elif len(values) == 22:  # Breast cancer
        model = pickle.load(open('models/breast_cancer.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    elif len(values) == 24:  # Kidney disease
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    elif len(values) == 10:  # Liver disease
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()

            for key, value in to_predict_dict.items():
                try:
                    to_predict_dict[key] = int(value)
                except ValueError:
                    try:
                        to_predict_dict[key] = float(value)
                    except:
                        pass

            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
            
            if len(to_predict_list) == 8:
                chart = generate_diabetes_comparison_chart(to_predict_dict)
            elif len(to_predict_list) == 13:
                chart = generate_heart_comparison_chart(to_predict_dict)
            else:
                chart = None
                
            return render_template('predict.html', pred=pred, chart=chart)
    except Exception as e:
        print(f"Error in prediction: {e}")
        message = "Please enter valid data"
        return render_template("home.html", message=message)

    return render_template('predict.html', pred=0)

@app.route("/malariapredict", methods=['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            img = Image.open(request.files['image'])
            img.save("uploads/image.jpg")
            img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpg')
            os.path.isfile(img_path)
            img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
            img = tf.keras.utils.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            model = tf.keras.models.load_model("models/malaria.h5")
            pred = np.argmax(model.predict(img))
        except Exception as e:
            print(f"Error in malaria prediction: {e}")
            message = "Please upload an image"
            return render_template('malaria.html', message=message)
    return render_template('malaria_predict.html', pred=pred)

@app.route("/pneumoniapredict", methods=['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            img = Image.open(request.files['image']).convert('L')
            img.save("uploads/image.jpg")
            img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpg')
            os.path.isfile(img_path)
            img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
            img = tf.keras.utils.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            model = tf.keras.models.load_model("models/pneumonia.h5")
            pred = np.argmax(model.predict(img))
        except Exception as e:
            print(f"Error in pneumonia prediction: {e}")
            message = "Please upload an image"
            return render_template('pneumonia.html', message=message)
    return render_template('pneumonia_predict.html', pred=pred)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
