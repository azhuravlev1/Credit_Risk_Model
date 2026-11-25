import pandas as pd
import os
import pyarrow
import fastparquet


# List of file paths (trimmed for demonstration; you should extend this to include all paths)
file_paths = file_paths = [
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_applprev_1_0.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_applprev_1_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_applprev_1_2.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_applprev_2.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_base.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_1_0.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_1_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_1_2.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_1_3.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_1_4.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_2_0.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_2_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_2_2.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_2_3.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_2_4.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_2_5.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_2_6.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_2_7.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_2_8.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_2_9.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_2_10.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_a_2_11.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_b_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_credit_bureau_b_2.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_debitcard_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_deposit_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_other_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_person_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_person_2.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_static_0_0.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_static_0_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_static_0_2.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_static_cb_0.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_tax_registry_a_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_tax_registry_b_1.parquet',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/parquet_files/test/test_tax_registry_c_1.parquet',
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

# Collecting results in a list
results = []

# Function to analyze a Parquet file and store results
def analyze_parquet_file(file_path):
    try:
        df = pd.read_parquet(file_path)
        filename = os.path.basename(file_path)
        results.append({
            'Filename': filename,
            'Number of Rows': df.shape[0],
            'Number of Columns': df.shape[1],
            'Unique Case IDs': df['case_id'].nunique() if 'case_id' in df.columns else 'Not Applicable'
        })
    except Exception as e:
        results.append({
            'File Path': os.path.basename(file_path),
            'Error': str(e)
        })

# Process each file
for path in file_paths:
    analyze_parquet_file(path)

# Convert results to a DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('dataset_analysis_results.csv', index=False)