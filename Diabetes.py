from flask import Flask, jsonify, send_file
import matplotlib.pyplot as plt
import io
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Initialize the Flask app
app = Flask(__name__)

# Load and process the Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a simple Random Forest Regressor
clf = RandomForestRegressor(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Define the '/get_status' endpoint
@app.route('/get_status', methods=['GET'])
def get_status():
    training_shape = X_train.shape
    testing_shape = X_test.shape
    model_score = clf.score(X_test, y_test)

    response = {
        "training_data_shape": training_shape,
        "testing_data_shape": testing_shape,
        "training_percentage": 70,
        "testing_percentage": 30,
        "model_score": round(model_score * 100, 2),
        "feature_names": diabetes.feature_names,
        "visualizations": {
            "feature_importance_plot": "/visualize/feature_importance",
            "target_distribution_plot": "/visualize/target_distribution"
        }
    }
    return jsonify(response)

# Visualization: Feature Importance
@app.route('/visualize/feature_importance', methods=['GET'])
def visualize_feature_importance():
    feature_importances = clf.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(diabetes.feature_names, feature_importances, color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.tight_layout()

    # Save plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# Visualization: Target Distribution
@app.route('/visualize/target_distribution', methods=['GET'])
def visualize_target_distribution():
    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=30, rwidth=0.8, color='coral', alpha=0.7)
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    plt.title('Target Value Distribution')
    plt.tight_layout()

    # Save plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5009)
