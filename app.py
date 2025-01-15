from flask import Flask, request, render_template, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('heart.csv')

# Select important features based on correlation
correlation_with_target = data.corr()['target'].sort_values(ascending=False)
important_features = correlation_with_target[correlation_with_target.abs() > 0.2].index.drop('target')

# Prepare features and target
X = data[important_features]
y = data['target']

# Feature descriptions
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

# Train models
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/single_patient')
def single_patient():
    return render_template('single_patient.html', features=important_features, feature_descriptions=feature_descriptions)


@app.route('/upload_csv')
def upload_csv():
    return render_template('upload_csv.html')


@app.route('/single_result', methods=['POST'])
def single_result():
    input_data = [float(request.form[feature]) for feature in important_features]
    input_df = pd.DataFrame([input_data], columns=important_features)
    selected_model = request.form['model']

    model = trained_models[selected_model]
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0].tolist() if hasattr(model, 'predict_proba') else None

    result = {
        'Model': selected_model,
        'Predicted Class': int(prediction),
        'Probabilities': probabilities
    }

    return render_template('single_result.html', result=result)


@app.route('/csv_result', methods=['POST'])
def csv_result():
    if 'file' not in request.files:
        return "No file uploaded. Please upload a .csv or .xlsx file.", 400

    file = request.files['file']
    selected_model = request.form['model']

    if file.filename.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        data = pd.read_excel(file)
    else:
        return "Unsupported file format. Please upload a .csv or .xlsx file.", 400

    missing_columns = [col for col in important_features if col not in data.columns]
    if missing_columns:
        return f"Missing columns in the uploaded file: {', '.join(missing_columns)}", 400

    model = trained_models[selected_model]
    predictions = {}

    for index, row in data.iterrows():
        input_df = pd.DataFrame([row[important_features]])
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0].tolist() if hasattr(model, 'predict_proba') else None

        predictions[index] = {
            'Predicted Class': int(prediction),
            'Probabilities': probabilities
        }

    # Generate graph
    graph_filename = "static/csv_predictions_graph.png"
    positive_predictions = sum(pred['Predicted Class'] for pred in predictions.values())
    negative_predictions = len(predictions) - positive_predictions

    plt.figure(figsize=(6, 4))
    plt.bar(['Positive', 'Negative'], [positive_predictions, negative_predictions], color=['green', 'red'])
    plt.title(f"Prediction Results for {selected_model}")
    plt.xlabel("Outcome")
    plt.ylabel("Number of Patients")
    plt.savefig(graph_filename)
    plt.close()

    return render_template('csv_result.html', model=selected_model, predictions=predictions, graph_url=url_for('static', filename='csv_predictions_graph.png'))


if __name__ == '__main__':
    app.run(debug=True)
