import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def moving_average(signal, window_size):
    result = np.zeros(len(signal))
    for i in range(len(signal)):
        start = max(0, i - window_size // 2)
        end = min(len(signal), i + window_size // 2 + 1)
        result[i] = np.mean(signal[start:end])
    return result

def detect_peaks(data, window_size=20, sensitivity_factor=0.4, min_distance=5):
    peak_indices = []
    x_accelerations = np.array(data)
    
    moving_avg = moving_average(x_accelerations, window_size)
    
    mean = np.mean(x_accelerations)
    std_dev = np.std(x_accelerations)
    
    threshold = sensitivity_factor * std_dev
    for i in range(1, len(x_accelerations) - 1):
        current = x_accelerations[i]
        if (current > moving_avg[i] + threshold and
            current > x_accelerations[i-1] and
            current > x_accelerations[i+1] and
            current > 0):
            if not peak_indices or i - peak_indices[-1] >= min_distance:
                peak_indices.append(i)
    
    return peak_indices

def plot_and_save_segments(data, peaks, output_dir, csv_output_dir, filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Select relevant columns (skip timestamp, index, and battery percentage)
    columns_to_extract = [
        'right_leg_accel_x', 'right_leg_accel_y', 'right_leg_accel_z',
        'left_leg_accel_x', 'left_leg_accel_y', 'left_leg_accel_z',
    ]

    # columns_to_extract = [
    #     'right_leg_accel_x', 'right_leg_accel_y', 'right_leg_accel_z',
    #     'right_leg_gyro_x', 'right_leg_gyro_y', 'right_leg_gyro_z',
    #     'left_leg_accel_x', 'left_leg_accel_y', 'left_leg_accel_z',
    #     'left_leg_gyro_x', 'left_leg_gyro_y', 'left_leg_gyro_z'
    # ]

    data = data[columns_to_extract]

    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i+1]
        segment = data.iloc[start:end]
        
        # Plot the segment
        plt.figure(figsize=(8, 4))
        plt.plot(segment)
        plt.title(f"Segment between peaks {start} and {end} from {base_name}")
        segment_filename = f"{base_name}_segment_{start}_{end}.png"
        plt.savefig(os.path.join(output_dir, segment_filename))
        plt.close()
        
        # Save the raw data of the segment
        segment_csv_filename = f"{base_name}_segment_{start}_{end}.csv"
        segment.to_csv(os.path.join(csv_output_dir, segment_csv_filename), index=False)

def process_all_files(input_dir, output_dir, csv_output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_dir, filename)
            df = pd.read_csv(filepath)

            # Extract the right leg accel z data for peak detection
            x_accel_data = df['right_leg_accel_x'].values

            peaks = detect_peaks(x_accel_data)
            plot_and_save_segments(df, peaks, output_dir, csv_output_dir, filename)

            # Plot the overall results
            plt.figure(figsize=(12, 6))
            plt.plot(x_accel_data)
            plt.plot(peaks, x_accel_data[peaks], "x")
            plt.title(f"Peak Detection Results for {filename}")
            results_filename = f"{os.path.splitext(filename)[0]}_results.png"
            plt.savefig(os.path.join(output_dir, results_filename))
            plt.close()

# Directories
input_directory = "exercise_wise_ml_pipelines/hopping/data"
output_directory = "exercise_wise_ml_pipelines/hopping/extracted_segments"
csv_output_directory = "exercise_wise_ml_pipelines/hopping/extracted_segments_csv"

# Process all files in the directory
process_all_files(input_directory, output_directory, csv_output_directory)
