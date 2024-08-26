import os
import pandas as pd
import numpy as np
import glob

# Mapping exercises to their respective columns
exercises_to_columns = {
    "Step Down from Height (dominant)": [3, 4],
    "Step Down from Height (nondominant)": [3, 4],
    "Step over an obstacle (dominant)": [3, 4],
    "Step over an obstacle (nondominant)": [3, 4],
    "Jump symmetrically": [3, 4],
    "Hit Balloon Up": [1, 2],
    "Stand on one leg (dominant)": [3, 4],
    "Stand on one leg (nondominant)": [3, 4],
    "Hop forward on one leg (dominant)": [3, 4],
    "Hop forward on one leg (nondominant)": [3, 4],
    "Jumping Jack without Clap": [3, 4],
    "Dribbling in Fig 8": [1, 2, 3, 4, 5],
    "Dribbling in Fig O": [1, 2, 3, 4, 5],
    "Jumping Jack with Clap": [1, 2, 3, 4],
    "Criss Cross with Clap": [1, 2, 3, 4],
    "Criss Cross without Clap": [3, 4],
    "Criss Cross with leg forward": [3, 4],
    "Skipping": [1, 2, 3, 4],
    "Ball Bounce and Catch": [1, 2, 5],
    "Forward Backward Spread Legs and Back": [3, 4],
    "Alternate feet forward backward": [3, 4],
    "Jump asymmetrically": [3, 4],
    "Hop 9 metres (dominant)": [3, 4],
    "Hop 9 metres (nondominant)": [3, 4],
}

# Mapping exercises to labels
exercise_labels = {
    "Step Down from Height (dominant)": "step_down",
    "Step Down from Height (nondominant)": "step_down",
    "Step over an obstacle (dominant)": "step_over_obstacle",
    "Step over an obstacle (nondominant)": "step_over_obstacle",
    "Jump symmetrically": "jump_symmetrically",
    "Hit Balloon Up": "hit_balloon_up",
    "Stand on one leg (dominant)": "stand_on_one_leg",
    "Stand on one leg (nondominant)": "stand_on_one_leg",
    "Hop forward on one leg (dominant)": "hopping",
    "Hop forward on one leg (nondominant)": "hopping",
    "Jumping Jack without Clap": "jumping_jack_without_claps",
    "Dribbling in Fig 8": "dribbling",
    "Dribbling in Fig O": "dribbling",
    "Jumping Jack with Clap": "jumping_jack_with_claps",
    "Criss Cross with Clap": "criss_cross_with_clap",
    "Criss Cross without Clap": "criss_cross_without_clap",
    "Criss Cross with leg forward": "criss_cross_without_clap",
    "Skipping": "skipping",
    "Ball Bounce and Catch": "ball_bounce_and_catch",
    "Forward Backward Spread Legs and Back": "forward_backward_alternate_legs",
    "Jump asymmetrically": "jump_symmetrically",
    "Hop 9 metres (dominant)": "hopping",
    "Hop 9 metres (nondominant)": "hopping",
}

# Function to rename files based on specific patterns
def rename_files(input_dir):
    for file_path in glob.glob(os.path.join(input_dir, '*.csv')):
        file_name = os.path.basename(file_path)
        new_file_name = file_name
        
        new_file_name = new_file_name.replace('non-dominant', 'nondominant')
        
        new_file_name = new_file_name.replace('Dribbling in Fig - 8', 'Dribbling in Fig 8')

        new_file_name = new_file_name.replace('Dribbling in Fig - O', 'Dribbling in Fig O')
        
        if new_file_name != file_name:
            new_file_path = os.path.join(input_dir, new_file_name)
            os.rename(file_path, new_file_path)
            print(f"Renamed file: {file_name} -> {new_file_name}")
        else:
            print(f"No renaming needed for: {file_name}")

def reorder_columns(df, file_name):
    prefixes = ['right_hand', 'left_hand', 'right_leg', 'left_leg', 'ball']
    reordered_columns = []
    exercise_name = os.path.splitext(os.path.basename(file_name))[0].split('-')[0].strip()
    
    if exercise_name in exercises_to_columns:
        column_indices = exercises_to_columns[exercise_name]
        for index in column_indices:
            reordered_columns.extend([col for col in df.columns if col.startswith(prefixes[index-1])])
    else:
        for prefix in prefixes:
            reordered_columns.extend([col for col in df.columns if col.startswith(prefix)])
    
    reordered_columns = list(dict.fromkeys(reordered_columns))
    
    return df[reordered_columns]

def is_row_valid(row):
    return all((-1e10 < x < 1e10) if isinstance(x, (int, float)) else True for x in row)

def remove_abnormal_rows(df):
    valid_rows = df.apply(is_row_valid, axis=1)
    if not valid_rows.all():
        print(f"Removed {(~valid_rows).sum()} rows with abnormal values.")
    return df[valid_rows].reset_index(drop=True)

def find_valid_start_index(df, timestamp_cols):
    start_indices = []
    for col in timestamp_cols:
        start_index = df.index[df[col] < 1].min()
        if pd.notna(start_index):
            start_indices.append(start_index)
    
    if not start_indices:
        return None
    return max(start_indices)

def process_file(file_path, output_base_dir):
    print(f"Processing file: {file_path}")
    
    if os.stat(file_path).st_size == 0:
        print(f"Skipping empty file: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    
    if df.empty or len(df.columns) == 0:
        print(f"Skipping file with only headers: {file_path}")
        return
    
    df = reorder_columns(df, file_path)
    df = remove_abnormal_rows(df)
    
    timestamp_cols = [col for col in df.columns if 'timestamp' in col]
    start_index = find_valid_start_index(df, timestamp_cols)
    
    if start_index is None:
        print(f"Warning: No valid timestamps less than 1 second found in {file_path}")
        return
    
    df = df.loc[start_index:].reset_index(drop=True)
    
    index_cols = [col for col in df.columns if 'index' in col]
    for col in index_cols:
        duplicates = df[col].duplicated()
        if duplicates.sum() > 0:
            print(f"Found {duplicates.sum()} duplicate index values in column {col}. Removing them.")
            df = df[~duplicates]
    
    for col in timestamp_cols:
        time_diff = df[col].diff()
        large_gaps = (time_diff > 0.1).sum()
        if large_gaps > 20:
            print(f"Alert: {large_gaps} instances of timestamp differences exceeding 100ms in column {col}")
    
    # Get exercise name and corresponding label
    exercise_name = os.path.splitext(os.path.basename(file_path))[0].split('-')[0].strip()
    label = exercise_labels.get(exercise_name, "Unknown")
    
    # Create output directory based on the label
    output_dir = os.path.join(output_base_dir, label, 'data')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, os.path.basename(file_path))
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

def main():
    input_dir = 'DATA'
    output_base_dir = 'exercise_wise_ml_pipelines'
    
    # Step 1: Rename files as needed
    rename_files(input_dir)
    
    # Step 2: Process files after renaming
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    for file_path in glob.glob(os.path.join(input_dir, '*.csv')):
        process_file(file_path, output_base_dir)

if __name__ == "__main__":
    main()
