import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def extract_info_from_filename(filename):
    # Assuming the format is: "Hop forward on one leg (nondominant)-Grade 2-shanvi -12-Good-20240822114012264992.csv"
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('-')
    
    grade = parts[1].strip()  # Extract the grade
    student_name = parts[2].strip()  # Extract and trim whitespace from student name
    rep_count = parts[3].strip()  # Extract the rep count
    
    return grade, student_name, rep_count

def plot_and_save_segments(data, segment_duration, segment_plot_dir, segment_csv_dir, filename):
    # Extract information from filename
    grade, student_name, rep_count = extract_info_from_filename(filename)

    # Simplified file naming convention
    segment_base_name = f"{student_name}_{grade}_rep{rep_count}"
    
    # Select relevant columns (skip timestamp, index, and battery percentage)
    columns_to_extract = [
        'ball_accel_x', 'ball_accel_y', 'ball_accel_z',
        'ball_gyro_x', 'ball_gyro_y', 'ball_gyro_z',
    ]

    # Check if the required columns exist in the data
    if not all(col in data.columns for col in columns_to_extract):
        print(f"Required columns missing in {filename}. Deleting the file.")
        os.remove(filename)
        return

    data = data[columns_to_extract]

    # Calculate the number of readings for each segment
    num_readings_per_segment = int(segment_duration * 20)  # 20 readings = 1 second

    for i in range(0, len(data), num_readings_per_segment):
        segment = data.iloc[i:i+num_readings_per_segment]
        if segment.empty:
            continue
        
        # Plot the segment
        plt.figure(figsize=(8, 4))
        plt.plot(segment)
        plt.title(f"Segment {i//num_readings_per_segment + 1} from {segment_base_name}")
        segment_plot_filename = f"{segment_base_name}_segment_{i//num_readings_per_segment + 1}.png"
        plt.savefig(os.path.join(segment_plot_dir, segment_plot_filename))
        plt.close()
        
        # Save the raw data of the segment
        segment_csv_filename = f"{segment_base_name}_segment_{i//num_readings_per_segment + 1}.csv"
        segment.to_csv(os.path.join(segment_csv_dir, segment_csv_filename), index=False)

def process_all_files(input_dir, segment_plot_dir, segment_csv_dir, peak_plot_dir, segment_duration=1):
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

            columns_to_extract = [
                'ball_accel_x', 'ball_accel_y', 'ball_accel_z',
                'ball_gyro_x', 'ball_gyro_y', 'ball_gyro_z',
            ]

            data = df[columns_to_extract]

            if not all(col in data.columns for col in columns_to_extract):
                print(f"Required columns missing in {filename}. Deleting the file.")
                os.remove(filename)
                continue

            # Extract and plot/save 1-second segments
            plot_and_save_segments(df, segment_duration, segment_plot_dir, segment_csv_dir, filepath)

            # Plot the overall results for visualization purposes
            grade, student_name, rep_count = extract_info_from_filename(filename)
            plt.figure(figsize=(12, 6))
            plt.plot(df['ball_accel_x'])
            plt.plot(df['ball_accel_y'])
            plt.plot(df['ball_accel_z'])
            plt.plot(df['ball_gyro_x'])
            plt.plot(df['ball_gyro_y'])
            plt.plot(df['ball_gyro_z'])
            # plt.plot(df['right_hand_accel_x'])
            # plt.plot(df['right_hand_accel_y'])
            # plt.plot(df['right_hand_accel_z'])
            plt.title(f"Overall Acceleration Data for {student_name} - {grade} - Rep {rep_count}")
            peak_plot_filename = f"{student_name}_{grade}_rep{rep_count}_overview.png"
            plt.savefig(os.path.join(peak_plot_dir, peak_plot_filename))
            plt.close()

# Directories
input_directory = "exercise_wise_ml_pipelines/dribbling/data"
segment_plot_directory = "exercise_wise_ml_pipelines/dribbling/segment_plots"
segment_csv_directory = "exercise_wise_ml_pipelines/dribbling/segment_csvs"
peak_plot_directory = "exercise_wise_ml_pipelines/dribbling/peak_plots"

# Process all files in the directory
process_all_files(input_directory, segment_plot_directory, segment_csv_directory, peak_plot_directory)
