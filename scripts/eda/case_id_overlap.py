import pandas as pd
import pyarrow.parquet as pq
from itertools import combinations
import os


train_file_paths = [
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_applprev_1_0.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_applprev_1_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_applprev_2.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_base.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_1_0.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_1_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_1_2.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_1_3.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_2_0.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_2_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_2_2.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_2_3.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_2_4.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_2_5.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_2_6.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_2_7.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_2_8.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_2_9.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_2_10.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_b_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_b_2.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_debitcard_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_deposit_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_other_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_person_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_person_2.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_static_0_0.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_static_0_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_static_cb_0.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_tax_registry_a_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_tax_registry_b_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/train/train_tax_registry_c_1.parquet'
]

def load_unique_case_ids(filepath):
    """Load unique Case IDs from a specified parquet file."""
    try:
        table = pq.read_table(filepath, columns=['case_id'])
        return pd.Series(table.column('case_id').to_pylist()).unique()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

def calculate_overlap_percentage(ids1, ids2):
    """Calculate the percentage overlap between two sets of IDs."""
    set1, set2 = set(ids1), set(ids2)
    if len(set1) == 0 or len(set2) == 0:
        return 0  # Avoid division by zero if any dataset is empty
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return (intersection / union) * 100


def get_clean_filename(filepath):
    """Extract the base filename without extension from the full file path."""
    base = os.path.basename(filepath)
    return os.path.splitext(base)[0]

def main(file_paths):
    """Main function to compute the overlap percentages between train datasets."""
    # Load unique Case IDs for each file that includes 'train' in its path
    unique_ids = {get_clean_filename(path): load_unique_case_ids(path) for path in file_paths if 'train' in path}
    results = []

    # Compute overlap for each pair of datasets using combinations to avoid repeating pairs
    for (path1, path2) in combinations(unique_ids.keys(), 2):
        overlap_percentage = calculate_overlap_percentage(unique_ids[path1], unique_ids[path2])
        results.append((path1, path2, overlap_percentage))

    # Create a DataFrame to store the results in a more readable format
    results_df = pd.DataFrame(results, columns=['Dataset 1', 'Dataset 2', 'Overlap Percentage'])
    return results_df

results_table = main(train_file_paths)

results_table.to_csv('/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/outputs/overlap_results.csv', index=False)