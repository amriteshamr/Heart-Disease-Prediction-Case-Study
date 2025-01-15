from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('heart.csv')

# Select important features based on correlation
correlation_with_target = data.corr()['target'].sort_values(ascending=False)
important_features = correlation_with_target[correlation_with_target.abs() > 0.2].index.drop('target')

# Prepare features and target
X = data[important_features]
y = data['target']

# Calculate feature ranges
feature_ranges = {
    feature: (data[feature].min(), data[feature].max()) for feature in important_features
}

# Remove range for 'age' and add description for 'sex'
feature_ranges['age'] = None
feature_descriptions = {
    'thalach': "Maximum heart rate achieved",
    'oldpeak': "ST depression induced by exercise relative to rest",
    'ca': "Number of major vessels (0–3) colored by fluoroscopy",
    'thal': "Thalassemia (a blood disorder): 1 = normal; 2 = fixed defect; 3 = reversible defect",
    'cp': "Chest pain type: 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic",
    'slope': "The slope of the peak exercise ST segment: 0 = upsloping, 1 = flat, 2 = downsloping",
    'sex': "Sex: 0 = female, 1 = male",
    'age': "Age of the patient",
    'exang': "Exercise-induced angina: 0 = no, 1 = yes"
}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize models
models = {
    'SVM': SVC(probability=True),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Train and evaluate each model
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

@app.route('/')
def home():
    return render_template(
        'index.html',
        features=important_features.tolist(),  # Ensure this is a list
        feature_ranges=feature_ranges,
        feature_descriptions=feature_descriptions
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/single_patient')
def single_patient():
    return render_template('single_patient.html', features=important_features, feature_descriptions=feature_descriptions)

@app.route('/upload_csv')
def upload_csv():
    return render_template('upload_csv.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = []
    for feature in important_features:
        value = float(request.form[feature])
        input_data.append(value)

    input_df = pd.DataFrame([input_data], columns=important_features)
    predictions = {}

    for name, model in trained_models.items():
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
        predictions[name] = {
            'Predicted Class': int(prediction),
            'Probabilities': probability.tolist() if probability is not None else None
        }

    return render_template('result.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)