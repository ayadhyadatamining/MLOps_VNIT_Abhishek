from flask import Flask, jsonify, send_file
import matplotlib.pyplot as plt
import io
import base64
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Initialize the Flask app
app = Flask(__name__)

# Load and process the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a simple Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Define the '/get_status' endpoint
@app.route('/get_status', methods=['GET'])
def get_status():
    training_shape = X_train.shape
    testing_shape = X_test.shape
    classifier_accuracy = clf.score(X_test, y_test)

    response = {
        "training_data_shape": training_shape,
        "testing_data_shape": testing_shape,
        "training_percentage": 70,
        "testing_percentage": 30,
        "classifier_accuracy": round(classifier_accuracy * 100, 2),
        "feature_names": wine.feature_names,
        "target_names": wine.target_names.tolist(),
        "visualizations": {
            "feature_importance_plot": "/visualize/feature_importance",
            "class_distribution_plot": "/visualize/class_distribution"
        }
    }
    return jsonify(response)

# Visualization: Feature Importance
@app.route('/visualize/feature_importance', methods=['GET'])
def visualize_feature_importance():
    feature_importances = clf.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(wine.feature_names, feature_importances, color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.tight_layout()

    # Save plot to a BytesIO buffer and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# Visualization: Class Distribution
@app.route('/visualize/class_distribution', methods=['GET'])
def visualize_class_distribution():
    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=len(wine.target_names), rwidth=0.8, color='coral', alpha=0.7)
    plt.xticks(range(len(wine.target_names)), wine.target_names)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Class Distribution')
    plt.tight_layout()

    # Save plot to a BytesIO buffer and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# Run the app
if __name__ == "__main__":
    # Use host="0.0.0.0" to make the API accessible on a network
    app.run(debug=True, host="127.0.0.1", port=5009)
