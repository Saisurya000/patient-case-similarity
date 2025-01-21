from flask import Flask, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Assume the model is already trained and serialized. 
# You would load your trained model here if necessary.

# Define the features
categorical_features = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level']
numerical_features = ['Age']

# Load the preprocessor and classifier (assuming they are already defined and trained)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

# The classifier model can be loaded here (assuming trained model exists)
clf = MultiOutputClassifier(DecisionTreeClassifier(criterion='gini', max_depth=100))

# Placeholder function to simulate prediction (replace with actual model loading and prediction)
def predict_disease(input_data):
    X = pd.DataFrame([input_data])
    X_encoded = preprocessor.fit_transform(X)
    
    # Here, `clf.predict()` would be used to get the prediction
    predictions = clf.predict(X_encoded)
    
    return predictions[0]  # Return predictions as a tuple

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Extract data from request
    input_data = {
        'Age': data.get('Age'),
        'Fever': data.get('Fever'),
        'Cough': data.get('Cough'),
        'Fatigue': data.get('Fatigue'),
        'Difficulty Breathing': data.get('Difficulty Breathing'),
        'Gender': data.get('Gender'),
        'Blood Pressure': data.get('Blood Pressure'),
        'Cholesterol Level': data.get('Cholesterol Level')
    }

    # Get prediction
    predictions = predict_disease(input_data)
    
    # Return prediction as JSON
    return jsonify({
        'Disease': predictions[0],
        'Precautions': predictions[1],
        'Medications': predictions[2]
    })

if __name__ == '__main__':
    app.run(debug=True)
