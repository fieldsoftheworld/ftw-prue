import pandas as pd
import glob
import os
import argparse


def calculate_overall_average_metrics(directory_path: str, expr_type: str = "main"):
    file_list = glob.glob(os.path.join(directory_path, f'*_{expr_type}.json'))
    
    if not file_list:
        print(f"Error: No files found in '{directory_path}'.")
        print("Please check the model name and ensure the directory path is correct.")
        return

    all_data_frames = []
    
    metric_cols = [
        'pixel_level_iou',
        'pixel_level_precision',
        'pixel_level_recall',
        'object_level_precision',
        'object_level_recall',
        'object_level_f1'
    ]
    
    print(f"Found {len(file_list)} files. Reading data...")

    for file_path in file_list:
        try:
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
    parser = argparse.ArgumentParser(
        description="Calculate the single overall average performance metrics for a specified model."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of the model whose results to average (e.g., terramind). Results must be in results/<model_name>"
    )

    parser.add_argument(
        "--expr",
        type=str,
        required=True,
        help="The name of the experiment type whose results to average (e.g., main or supp)."
    )

    parser.add_argument(
        "--result_dir",
        default="./results",
        type=str,
        help="The base directory containing the model results."
    )

    args = parser.parse_args()
    calculate_overall_average_metrics(os.path.join(args.result_dir, args.model), expr_type=args.expr)