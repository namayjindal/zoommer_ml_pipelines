import os
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_feature_files(directory, test_size=0.1, random_state=42):
    data = []
    file_info = []
    file_count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            
            # Check if the file is empty by size
            if os.path.getsize(filepath) == 0:
                os.remove(filepath)  # Delete the file if it's empty
                print(f"Deleted empty file: {filename}")
                continue  # Skip processing this file

            try:
                df = pd.read_csv(filepath)
                if df.empty:
                    raise pd.errors.EmptyDataError(f"No data in file: {filename}")

                file_count += 1
                features = df.values
                for row_num, row in enumerate(features):
                    data.append(row)
                    file_info.append((filename, row_num))

            except pd.errors.EmptyDataError:
                print(f"Skipping empty or invalid file: {filename}")
                continue  # Skip processing this file
    
    data = np.array(data)
    train_data, test_data, train_info, test_info = train_test_split(
        data, file_info, test_size=test_size, random_state=random_state
    )
    
    return train_data, test_data, train_info, test_info, file_count



# Load and split the data
feature_dir = "exercise_wise_ml_pipelines/skipping/features_output"
X_train, X_test, train_file_info, test_file_info, train_file_count = load_and_split_feature_files(feature_dir)
print(f"Number of training files: {len(train_file_info)}")
print(f"Number of test files: {len(test_file_info)}")

def find_optimal_clusters(X, max_clusters=10):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_clusters

# Find optimal number of clusters
optimal_clusters = find_optimal_clusters(X_train)
print(f"Optimal number of clusters: {optimal_clusters}")

class AnomalyDetector:
    def __init__(self, n_clusters):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters)
    
    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        distances = self.kmeans.transform(X_scaled)
        return np.min(distances, axis=1)

detector = AnomalyDetector(n_clusters=optimal_clusters)
detector.fit(X_train)

class TFAnomalyDetector(tf.Module):
    def __init__(self, kmeans, scaler):
        self.n_clusters = kmeans.n_clusters
        self.centroids = tf.Variable(kmeans.cluster_centers_, dtype=tf.float32)
        self.scaler_mean = tf.Variable(scaler.mean_, dtype=tf.float32)
        self.scaler_scale = tf.Variable(scaler.scale_, dtype=tf.float32)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 84], dtype=tf.float32)])
    def __call__(self, x):
        x_scaled = (x - self.scaler_mean) / self.scaler_scale
        distances = tf.reduce_sum(tf.square(tf.expand_dims(x_scaled, axis=1) - self.centroids), axis=2)
        return tf.reduce_min(distances, axis=1)

tf_detector = TFAnomalyDetector(detector.kmeans, detector.scaler)

converter = tf.lite.TFLiteConverter.from_keras_model(tf_detector)
tflite_model = converter.convert()

with open('exercise_wise_ml_pipelines/skipping/skipping_anomaly_detector.tflite', 'wb') as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

def get_tflite_predictions(interpreter, X):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    results = []
    for sample in X:
        interpreter.set_tensor(input_details[0]['index'], sample.reshape(1, 84).astype(np.float32))
        interpreter.invoke()
        results.append(interpreter.get_tensor(output_details[0]['index'])[0])
    return np.array(results)

tflite_train_results = get_tflite_predictions(interpreter, X_train)
threshold = np.percentile(tflite_train_results, 85)
print(f"\nAnomaly threshold (based on TFLite model): {threshold}")

tflite_test_results = get_tflite_predictions(interpreter, X_test)

print("\nTFLite Model Anomaly Detection:")
correct_predictions = 0
total_predictions = 0

for (filename, row_num), result, anomaly_score in zip(test_file_info, tflite_test_results > threshold, tflite_test_results):
    is_correct = not result  # Since all test data should ideally be non-anomalous
    correct_predictions += int(is_correct)
    total_predictions += 1
    print(f"File: {filename}, Row: {row_num}, Is Anomaly: {result}, Anomaly Score: {anomaly_score}, Correct: {is_correct}")

# Summary
anomaly_counts = {}
file_accuracies = {}

for (filename, _), result in zip(test_file_info, tflite_test_results > threshold):
    if filename not in anomaly_counts:
        anomaly_counts[filename] = {"total": 0, "anomalies": 0, "correct": 0}
    anomaly_counts[filename]["total"] += 1
    if result:
        anomaly_counts[filename]["anomalies"] += 1
    if not result:  # Since we're expecting non-anomalous results
        anomaly_counts[filename]["correct"] += 1

print("\nSummary:")
overall_correct = 0
overall_total = 0

for filename, counts in anomaly_counts.items():
    print(f"File: {filename}")
    print(f"  Total rows: {counts['total']}")
    print(f"  Anomalies detected: {counts['anomalies']}")
    print(f"  Correct predictions: {counts['correct']}")
    accuracy = (counts['correct'] / counts['total']) * 100
    print(f"  Accuracy: {accuracy:.2f}%")
    print()
    
    overall_correct += counts['correct']
    overall_total += counts['total']

overall_accuracy = (overall_correct / overall_total) * 100
print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
print(f"Total Correct Predictions: {overall_correct}")
print(f"Total Predictions: {overall_total}")
print(f"\nAnomaly threshold (based on TFLite model): {threshold}")
