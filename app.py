from flask import Flask, request, render_template
import joblib
import os
import re

app = Flask(__name__)

# ----------------------------
# Load model and vectorizer
# ----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, 'model')

model = joblib.load(os.path.join(model_dir, 'job_fraud_model.pkl'))
vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))

# ----------------------------
# Text cleaning
# ----------------------------
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower()

# ----------------------------
# Home route
# ----------------------------
@app.route('/')
def home():
    return render_template('index.html')

# ----------------------------
# Prediction route
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company', '')
    title = request.form.get('title', '')
    description = request.form.get('description', '')
    
    # Combine text
    text = clean_text(company + ' ' + title + ' ' + description)
    vec = vectorizer.transform([text])
    
    # Predict
    pred = model.predict(vec)[0]
    confidence = model.predict_proba(vec)[0].max()
    
    result_text = "Genuine" if pred == 0 else "Fraudulent"
    
    return render_template('index.html', company=company, title=title, description=description,
                           result=result_text, confidence=f"{confidence*100:.2f}%")

if __name__ == '__main__':
    app.run(debug=True)
