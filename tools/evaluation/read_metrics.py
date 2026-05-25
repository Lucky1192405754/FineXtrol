# read_metrics.py

import argparse
import re
import os

def parse_log_file(log_path, replication_index):
    """
    Parse a log file and extract evaluation metrics for the specified replication.

    Args:
        log_path (str): Path to the log file.
        replication_index (int): Replication index to extract; for example, 0 means Replication 0.

    Returns:
        tuple: Tuple containing (fid, r_precision_top1, r_precision_top2, r_precision_top3, diversity, matching_score).
               If the specified replication or a metric is not found, the corresponding value is None.
    """
    if not os.path.exists(log_path):
        print(f"Error: file does not exist at path {log_path}")
        return None, None, None, None, None, None

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    replication_blocks = re.split(r'==================== Replication \d+ ====================', content)[1:]
    
    if replication_index >= len(replication_blocks):
        print(f"Error: evaluation replication not found {replication_index}。")
        print(f"This file only contains {len(replication_blocks)} evaluations, indexed from 0 to {len(replication_blocks)-1}）。")
        return None, None, None, None, None, None

    target_block = replication_blocks[replication_index]

    patterns = {
        'fid': r'---> \[vald\] FID: ([\d\.]+)',
        'r_precision_top1': r'---> \[vald\] R_precision:.*?\(top 1\): ([\d\.]+)', # NEW
        'r_precision_top2': r'---> \[vald\] R_precision:.*?\(top 2\): ([\d\.]+)', # NEW
        'r_precision_top3': r'---> \[vald\] R_precision:.*?\(top 3\): ([\d\.]+)',
        'diversity': r'---> \[vald\] Diversity: ([\d\.]+)',
        'matching_score': r'---> \[vald\] Matching Score: ([\d\.]+)'
    }

    results = {}
    for metric, pattern in patterns.items():
        match = re.search(pattern, target_block)
        if match:
            results[metric] = float(match.group(1))
        else:
            results[metric] = None
            print(f"Warning: in evaluation replication {replication_index} could not find metric '{metric}'。")
            
    return (
        results.get('fid'),
        results.get('r_precision_top1'),
        results.get('r_precision_top2'),
        results.get('r_precision_top3'),
        results.get('diversity'),
        results.get('matching_score')
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract evaluation metrics for a specified replication from a log file."
    )
    parser.add_argument(
        '--log_path',
        type=str,
        required=True,
        help='Full path to the evaluation log file.'
    )
    parser.add_argument(
        '--replication_index',
        type=int,
        required=True,
        help='Replication index to extract; for example, enter 1 for "Replication 1".'
    )

    args = parser.parse_args()

    fid, r_top1, r_top2, r_top3, diversity, matching_score = parse_log_file(args.log_path, args.replication_index)

    if fid is not None:
        print("\n" + "="*25)
        print("     Extracted evaluation metrics")
        print("="*25)
        print(f"Log file: {args.log_path}")
        print(f"Evaluation replication index: {args.replication_index}")
        print("-"*25)
        print(f"Matching Score: {matching_score}")
        print(f"R-precision (top 1): {r_top1}")
        print(f"R-precision (top 2): {r_top2}")
        print(f"R-precision (top 3): {r_top3}")
        print(f"FID: {fid}")
        print(f"Diversity: {diversity}")
        
        print("\n--- Ordered output (FID, R-p@1, R-p@2, R-p@3, Diversity, MS) ---")
        print(f"{fid}, {r_top1}, {r_top2}, {r_top3}, {diversity}, {matching_score}")
        print("="*25)