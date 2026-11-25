import pandas as pd
import numpy as np
import os

# Load feature definitions
feature_definitions_path = "/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/feature_definitions.csv"
feature_definitions = pd.read_csv(feature_definitions_path)

# List of file paths
file_paths = [
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_applprev_1_0.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_applprev_1_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_applprev_1_2.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_applprev_2.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_base.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_1_0.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_1_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_1_2.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_1_3.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_1_4.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_2_0.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_2_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_2_2.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_2_3.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_2_4.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_2_5.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_2_6.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_2_7.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_2_8.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_2_9.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_2_10.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_a_2_11.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_b_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_credit_bureau_b_2.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_debitcard_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_deposit_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_other_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_person_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_person_2.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_static_0_0.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_static_0_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_static_0_2.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_static_cb_0.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_tax_registry_a_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_tax_registry_b_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/test/test_tax_registry_c_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_applprev_1_0.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_applprev_1_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_applprev_2.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_base.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_1_0.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_1_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_1_2.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_1_3.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_2_0.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_2_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_2_2.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_2_3.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_2_4.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_2_5.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_2_6.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_2_7.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_2_8.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_2_9.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_2_10.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_b_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_b_2.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_debitcard_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_deposit_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_other_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_person_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_person_2.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_static_0_0.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_static_0_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_static_cb_0.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_tax_registry_a_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_tax_registry_b_1.csv',
    '/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/data/home-credit-credit-risk-model-stability/csv_files/train/train_tax_registry_c_1.csv'
]

# Load the feature definitions
feature_definitions = pd.read_csv(feature_definitions_path)


# Function to process each file
def process_file(file_path, feature_definitions):
    # Load the current database
    df = pd.read_csv(file_path, low_memory=False)

    # Extract the file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Find relevant features
    relevant_features = feature_definitions[feature_definitions['Variable'].isin(df.columns)]

    # Prepare the summary dataframe
    summary_df = pd.DataFrame(columns=[
        'Feature Name', 'Description', 'Null Entries', 'Not 0 Entries', 'Non-null Non-zero Entries', 'Category Count'
    ])

    for _, row in relevant_features.iterrows():
        feature_name = row['Variable']
        description = row['Description']

        if df[feature_name].dtype == 'object':  # Categorical feature
            category_count = df[feature_name].nunique()
            summary_row = {
                'Feature Name': feature_name,
                'Description': description,
                'Null Entries': None,
                'Not 0 Entries': None,
                'Non-null Non-zero Entries': None,
                'Category Count': category_count
            }
        else:  # Numerical feature
            null_entries = df[feature_name].isnull().sum()
            not_zero_entries = (df[feature_name] != 0).sum()
            non_null_non_zero_entries = ((df[feature_name] != 0) & df[feature_name].notnull()).sum()
            summary_row = {
                'Feature Name': feature_name,
                'Description': description,
                'Null Entries': null_entries,
                'Not 0 Entries': not_zero_entries,
                'Non-null Non-zero Entries': non_null_non_zero_entries,
                'Category Count': None
            }

        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)

    # Save the summary dataframe
    output_path = f"/Users/andrey/Desktop/Projects/2024/HomeCreditCompetition/HomeCreditCompetition/outputs/feature_summary_{file_name}.csv"
    summary_df.to_csv(output_path, index=False)


# Process each file in the list
for file_path in file_paths:
    process_file(file_path, feature_definitions)