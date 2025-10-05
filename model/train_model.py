import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import re
import os

# ----------------------------
# Dynamic dataset path
# ----------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, 'dataset', 'jobs.csv')

# Load dataset
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please check the path.")

# Fill missing columns check
for col in ['title', 'company_profile', 'description', 'fraudulent']:
    if col not in df.columns:
        raise ValueError(f"CSV must contain '{col}' column.")

# ----------------------------
# Combine text fields
# ----------------------------
df['text'] = df['title'].fillna('') + ' ' + df['company_profile'].fillna('') + ' ' + df['description'].fillna('')

# ----------------------------
# Clean text
# ----------------------------
def clean_text(text):
    text = re.sub(r'http\S+', '', text)           # remove URLs
    text = re.sub(r'[^a-zA-Z ]', '', text)       # keep only letters and spaces
    return text.lower()

df['text'] = df['text'].apply(clean_text)

# ----------------------------
# Features and labels
# ----------------------------
X = df['text']
y = df['fraudulent']

# ----------------------------
# Split dataset
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Vectorize text
# ----------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# Train model
# ----------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_vec, y_train)

# ----------------------------
# Evaluate model (optional)
# ----------------------------
accuracy = model.score(X_test_vec, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

# ----------------------------
# Save model and vectorizer
# ----------------------------
model_dir = os.path.join(base_dir, 'model')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, 'job_fraud_model.pkl'))
joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))

print(f"✅ Model and vectorizer saved successfully in the '{model_dir}' folder!")
