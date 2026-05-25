import argparse
import re
import os

def extract_metrics_from_block(block_content):
    """Extract all metrics from a single replication text block."""
    
    patterns = {
        'fid': r'---> \[vald\] FID: ([\d\.]+)',
        'r_p_1': r'---> \[vald\] R_precision:.*?\(top 1\): ([\d\.]+)',
        'r_p_2': r'---> \[vald\] R_precision:.*?\(top 2\): ([\d\.]+)',
        'r_p_3': r'---> \[vald\] R_precision:.*?\(top 3\): ([\d\.]+)',
        'diversity': r'---> \[vald\] Diversity: ([\d\.]+)',
        'ms': r'---> \[vald\] Matching Score: ([\d\.]+)'
    }

    results = {}
    for metric, pattern in patterns.items():
        match = re.search(pattern, block_content)
        if match:
            results[metric] = float(match.group(1))
        else:
            results[metric] = None
    return results

def parse_and_average_log(log_path):
    """Parse one log file, extract metrics from all replications, and compute averages."""
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: file does not exist {log_path}")
        return None

    replication_blocks = re.split(r'==================== Replication \d+ ====================', content)[1:]
    
    if not replication_blocks:
        print(f"Warning: in {os.path.basename(log_path)} no evaluation replications were found.")
        return None

    all_metrics_raw = {
        'fid': [], 'r_p_1': [], 'r_p_2': [], 'r_p_3': [], 'diversity': [], 'ms': []
    }

    for i, block in enumerate(replication_blocks):
        metrics = extract_metrics_from_block(block)
        for key, value in metrics.items():
            if value is not None:
                all_metrics_raw[key].append(value)
            else:
                print(f"Warning: in {os.path.basename(log_path)} Replication {i}, could not find '{key}'.")
    
    final_results = {}
    for key, values in all_metrics_raw.items():
        if values:
            final_results[key] = sum(values) / len(values)
        else:
            final_results[key] = None

    return final_results

def main():
    """Main function: handle CLI arguments, scan the directory, and print staged results in the requested format."""
    parser = argparse.ArgumentParser(
        description="Automatically scan *_0.log files in the specified directory, extract average evaluation metrics, and print them by row."
    )
    parser.add_argument(
        '--log_dir', 
        type=str, 
        required=True, 
        help='Directory containing evaluation log files.'
    )
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        print(f"Error: directory '{args.log_dir}' does not exist.")
        return

    log_files = sorted([f for f in os.listdir(args.log_dir) if f.endswith('_0.log')])

    if not log_files:
        print(f"No matching log files were found in directory '{args.log_dir}' with suffix '_0.log'.")
        return

    print(f"\nFound {len(log_files)} files matching suffix '_0.log' in '{args.log_dir}'.")

    collected_rp_values = []

    print("\n" + "="*50)
    print(" " * 15 + "Stage 1: detailed results for all metrics")
    print("="*50 + "\n")

    for log_file in log_files:
        full_path = os.path.join(args.log_dir, log_file)
        avg_results = parse_and_average_log(full_path)
        
        
        if avg_results:
            required_keys = ['fid', 'r_p_1', 'r_p_2', 'r_p_3', 'diversity', 'ms']
            if all(key in avg_results and avg_results[key] is not None for key in required_keys):
                output_values = [
                    avg_results['fid'],
                    avg_results['r_p_1'],
                    avg_results['r_p_2'],
                    avg_results['r_p_3'],
                    avg_results['diversity'],
                    avg_results['ms']
                ]
                output_str = ", ".join([f"{v:.4f}" for v in output_values])
                print(output_str)

                collected_rp_values.append([output_values[1], output_values[2]])
            else:
                print("Unable to generate results: this log file is missing valid data for some metrics.")
                collected_rp_values.append(None)
        else:
            print("Unable to process this log file.")
            collected_rp_values.append(None)
        
        # print("-" * 50)

    if collected_rp_values:
        print("\n" + "="*50)
        print(" " * 10 + "Stage 2: R-precision (Top-1, Top-2) summary")
        print("="*50 + "\n")
        
        for rp_pair in collected_rp_values:
            if rp_pair:
                rp_only_str = ", ".join([f"{v:.4f}" for v in rp_pair])
                print(rp_only_str)
            else:
                print("N/A, N/A")

if __name__ == "__main__":
    main()