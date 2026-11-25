# Home Credit: Credit Risk Model Stability (Kaggle, 2024)

Exploration and a baseline model for the Kaggle competition [Home Credit - Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability). The goal is to predict which clients will default on a loan, and the metric rewards models whose performance stays stable from week to week, not just a high overall AUC.

Most of my time went into understanding the data, which is spread across 32 training tables at three levels of detail. My modelling stayed close to the official starter notebook and did not beat the stronger public notebooks.

## The data

Every loan application has a `case_id`. The base table holds one row per application with the decision date, the week number and the target. The other tables hang off it at different depths:

- depth 0: one row per application (static application data and credit bureau summaries)
- depth 1: several rows per application (previous applications, tax registry records, people linked to the applicant)
- depth 2: several rows per row of a depth 1 table (for example, payment history inside each credit bureau contract)

Column names end in a letter that tells you the type (A for amounts, D for dates, P for days past due, M for masked categories, T and L for other types), so dtypes and aggregations can be chosen from the name alone.

## What is in this repository

EDA scripts (`scripts/eda/`):

- `table_overview.py` lists every table with its number of rows, columns and unique case IDs, to see which tables are wide, which are long, and which have one row per application.
- `case_id_overlap.py` computes, for every pair of training tables, the share of case IDs they have in common. This shows which tables actually cover the same applicants and how sparse the deeper tables are.
- `feature_summaries.py` joins each table with the competition's feature definitions and writes a per-column summary: description, null count, non-zero count and number of categories.

Baseline model (`scripts/models/starter_lgbm.py`): based on the official starter notebook. It reads the tables with polars, casts types from the column suffix, aggregates the depth 1 and 2 tables to one row per application, joins everything onto the base table and trains a LightGBM classifier. It reports AUC on train, validation and test splits and computes the competition's Gini stability score: the weekly Gini coefficients are fitted with a linear trend, and the final score penalizes both a falling trend and week-to-week variability.

## Running it

The competition data is not included. Download it from the competition page into `data/home-credit-credit-risk-model-stability/` (both `csv_files/` and `parquet_files/`). The scripts currently use absolute paths from my machine, so update the paths at the top of each script before running. Requirements: pandas, polars, pyarrow, scikit-learn and lightgbm.

## What I would do differently

Most of the leaderboard gains in this competition came from aggregating many more columns from the deeper tables, dropping columns with a high share of missing values, and validating with folds grouped by week so the model is judged the same way the metric judges it. My baseline used only a handful of hand-picked features and a random split, which is a large part of why it fell short.
