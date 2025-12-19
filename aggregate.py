import pandas as pd
import glob
import os
import argparse 

def calculate_overall_average_metrics(directory_path: str):
    
    # We will search for all files in the model's directory
    file_list = glob.glob(os.path.join(directory_path, '*'))
    
    if not file_list:
        print(f"Error: No files found in '{directory_path}'.")
        print("Please check the model name and ensure the directory path is correct.")
        return

    all_data_frames = []
    
    # List of expected metric columns for averaging
    metric_cols = [
        'pixel_level_iou',
        'pixel_level_precision',
        'pixel_level_recall',
        'object_level_precision',
        'object_level_recall',
        'object_level_f1'
    ]
    
    # 1. Read and Combine Data
    print(f"Found {len(file_list)} files. Reading data as CSV/Text...")

    for file_path in file_list:
        try:
            # Read each file as CSV (as determined from your provided content)
            df_file = pd.read_csv(file_path, header=0)
            all_data_frames.append(df_file)

        except pd.errors.EmptyDataError:
            print(f"Warning: File {os.path.basename(file_path)} is empty. Skipping.")
        except Exception as e:
            print(f"Warning: Could not read {os.path.basename(file_path)} as CSV: {e}. Skipping file.")

    if not all_data_frames:
        print("No valid data records were extracted from the files.")
        return

    df = pd.concat(all_data_frames, ignore_index=True)
    
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        

    valid_metric_cols = [col for col in metric_cols if col in df.columns]
    
    overall_average = df[valid_metric_cols].mean().to_frame().T
    
    overall_average.index = ['Overall Average']
    
    
    print("\n" + "="*70)
    print("="*70)
    
    print(overall_average.to_string(float_format='%.3f'))
    


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Calculate the single overall average performance metrics for a specified model."
    )

    # 1. Define 'model' as a required argument with a flag
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, # Ensure the user still provides this one
        help="The name of the model whose results to average (e.g., terramind). Results must be in results/<model_name>"
    )

    # 2. Define 'result_dir' as an optional argument with a flag and a default value
    parser.add_argument(
        "--result_dir",
        default="/u/subashk/storage/ftw-prue/results", # Use your actual path
        type=str,
        help="The base directory containing the model results (e.g., /u/path/to/results)."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function
    calculate_overall_average_metrics(os.path.join(args.result_dir, args.model))