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

def detect_peaks(data, window_size=20, sensitivity_factor=0.4, min_distance=10):
    peak_indices = []
    z_accelerations = np.array(data)

    moving_avg = moving_average(z_accelerations, window_size)

    mean = np.mean(z_accelerations)
    std_dev = np.std(z_accelerations)

    threshold = sensitivity_factor * std_dev
    for i in range(1, len(z_accelerations) - 1):
        current = z_accelerations[i]
        if (abs(current - moving_avg[i]) > threshold and
            ((current < z_accelerations[i-1] and current < z_accelerations[i+1]))):
            if not peak_indices or i - peak_indices[-1] >= min_distance:
                peak_indices.append(i)

    return peak_indices

def extract_info_from_filename(filename):
    # Assuming the format is: "Hop forward on one leg (nondominant)-Grade 2-shanvi -12-Good-20240822114012264992.csv"
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('-')
    
    grade = parts[1].strip()  # Extract the grade
    student_name = parts[2].strip()  # Extract and trim whitespace from student name
    rep_count = parts[3].strip()  # Extract the rep count
    
    return grade, student_name, rep_count

def plot_and_save_segments(data, peaks, segment_plot_dir, segment_csv_dir, filename):
    # Extract information from filename
    grade, student_name, rep_count = extract_info_from_filename(filename)

    # Simplified file naming convention
    segment_base_name = f"{student_name}_{grade}_rep{rep_count}"
    
    # Select relevant columns (skip timestamp, index, and battery percentage)
    columns_to_extract = [
        'right_hand_accel_x', 'right_hand_accel_y', 'right_hand_accel_z',
        'left_hand_accel_x', 'left_hand_accel_y', 'left_hand_accel_z',
        'ball_accel_x', 'ball_accel_y', 'ball_accel_z',
    ]

    # Check if the required columns exist in the data
    if not all(col in data.columns for col in columns_to_extract):
        print(f"Required columns missing in {filename}. Deleting the file.")
        os.remove(filename)
        return

    data = data[columns_to_extract]

    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i+1]
        segment = data.iloc[start:end]
        
        # Plot the segment
        plt.figure(figsize=(8, 4))
        plt.plot(segment)
        plt.title(f"Segment {i+1} from {segment_base_name}")
        segment_plot_filename = f"{segment_base_name}_segment_{i+1}.png"
        plt.savefig(os.path.join(segment_plot_dir, segment_plot_filename))
        plt.close()
        
        # Save the raw data of the segment
        segment_csv_filename = f"{segment_base_name}_segment_{i+1}.csv"
        segment.to_csv(os.path.join(segment_csv_dir, segment_csv_filename), index=False)

def process_all_files(input_dir, segment_plot_dir, segment_csv_dir, peak_plot_dir):
    if not os.path.exists(segment_plot_dir):
        os.makedirs(segment_plot_dir)
    if not os.path.exists(segment_csv_dir):
        os.makedirs(segment_csv_dir)
    if not os.path.exists(peak_plot_dir):
        os.makedirs(peak_plot_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_dir, filename)
            df = pd.read_csv(filepath)

            print(f"Processing file: {filename}")

            # Extract the right leg accel x data for peak detection
            x_accel_data = df['right_hand_accel_x'].values
            y_accel_data = df['right_hand_accel_y'].values
            z_accel_data = df['right_hand_accel_z'].values
            ball_x_accel_data = df['ball_accel_x'].values
            ball_y_accel_data = df['ball_accel_y'].values
            ball_z_accel_data = df['ball_accel_z'].values

            columns_to_extract = [
                'right_hand_accel_x', 'right_hand_accel_y', 'right_hand_accel_z',
                'left_hand_accel_x', 'left_hand_accel_y', 'left_hand_accel_z',
                'ball_accel_x', 'ball_accel_y', 'ball_accel_z',
            ]

            data = df[columns_to_extract]

            if not all(col in data.columns for col in columns_to_extract):
                print(f"Required columns missing in {filename}. Deleting the file.")
                os.remove(filename)
                continue

            peaks = detect_peaks(z_accel_data)
            plot_and_save_segments(df, peaks, segment_plot_dir, segment_csv_dir, filepath)

            # Plot the overall results for peaks
            grade, student_name, rep_count = extract_info_from_filename(filename)
            plt.figure(figsize=(12, 6))
            plt.plot(x_accel_data)
            plt.plot(y_accel_data)
            plt.plot(z_accel_data)
            # plt.plot(ball_x_accel_data)
            # plt.plot(ball_y_accel_data)
            # plt.plot(ball_z_accel_data)
            plt.plot(peaks, z_accel_data[peaks], "x")
            plt.title(f"Peak Detection for {student_name} - {grade} - Rep {rep_count}")
            peak_plot_filename = f"{student_name}_{grade}_rep{rep_count}_peaks.png"
            plt.savefig(os.path.join(peak_plot_dir, peak_plot_filename))
            plt.close()

# Directories
input_directory = "exercise_wise_ml_pipelines/ball_bounce_and_catch/data"
segment_plot_directory = "exercise_wise_ml_pipelines/ball_bounce_and_catch/segment_plots"
segment_csv_directory = "exercise_wise_ml_pipelines/ball_bounce_and_catch/segment_csvs"
peak_plot_directory = "exercise_wise_ml_pipelines/ball_bounce_and_catch/peak_plots"

# Process all files in the directory
process_all_files(input_directory, segment_plot_directory, segment_csv_directory, peak_plot_directory)
