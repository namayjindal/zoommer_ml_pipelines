import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def calculate_features(segment, window_size):
    features = []
    for start in range(0, len(segment) - window_size + 1, 4):
        end = start + window_size
        window = segment[start:end]
        
        feature_vector = {
            'mean': np.mean(window),
            'std_dev': np.std(window),
            'rms': np.sqrt(np.mean(window ** 2)),
            'min': np.min(window),
            'max': np.max(window),
            'skewness': skew(window),
            'kurtosis': kurtosis(window)
        }
        
        features.append(feature_vector)
    
    return features

def process_csv_file(filename, window_size):
    df = pd.read_csv(filename)
    
    all_features = []
    for column in df.columns:
        if 'accel' in column or 'gyro' in column:
            segment = df[column].values
            features = calculate_features(segment, window_size)
            features_df = pd.DataFrame(features)
            
            # Prefix the column name to feature names
            features_df = features_df.add_prefix(f"{column}_")
            all_features.append(features_df)
    
    # Combine features from all axes into a single DataFrame
    all_features_df = pd.concat(all_features, axis=1)
    return all_features_df

def extract_features_from_segments(directory, window_size=4, output_dir='peak_detection/features_output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            features_df = process_csv_file(filepath, window_size)
            
            # Save the features to a new CSV file
            output_filename = os.path.join(output_dir, f"features_{filename}")
            features_df.to_csv(output_filename, index=False)
            print(f"Extracted features saved to {output_filename}")

# Directory containing the extracted segments
segments_directory = 'peak_detection/extracted_segments_csv'

# Extract features from all files in the directory
extract_features_from_segments(segments_directory)
