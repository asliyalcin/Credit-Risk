import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def print_corr(df, col1, col2):
    print(f"Corelation between'{col1}' and '{col2}' : {df[col1].corr(df[col2]):.4f}")

def clean_missing_values(df, num_cols):
    # 1) Missing value summary
    print("### INITIAL MISSING VALUE SUMMARY ###\n")
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (df.isna().mean() * 100).round(2)
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    print(missing_df[missing_df['missing_pct'] > 0])
    print("\n")
    
    if not missing.empty:
        plt.figure(figsize=(8,5))
        missing.head(20).plot(kind='bar', color='#e67e22')
        plt.title('Missing Values (Top 20 Columns)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
    
    num_summary = df[num_cols].describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]).T
    num_summary
    
    # 2) Print correlations
    print("### CORRELATION CHECKS ###")
    print_corr(df, 'total_pymnt_inv', 'total_pymnt')
    print_corr(df, 'funded_amnt_inv', 'funded_amnt')
    print_corr(df, 'installment', 'last_pymnt_amnt')
    print("\n")
    
    # 3) Imputation steps
    print("### IMPUTATION IN PROGRESS ###")
    
    df['funded_amnt_inv'] = df['funded_amnt_inv'].fillna(df['funded_amnt'])
    df['total_pymnt_inv'] = df['total_pymnt_inv'].fillna(df['total_pymnt'])
    df['installment'] = df['installment'].fillna(df['last_pymnt_amnt'])
    
    # Drop original columns
    df.drop(columns=['funded_amnt', 'total_pymnt'], inplace=True)
    
    # Group median imputation
    df['annual_inc'] = df.groupby(['emp_title', 'emp_length'])['annual_inc'].transform(
        lambda x: x.fillna(x.median())
    )

    print("Imputation completed.\n")
    
    # 4) Missing value summary after imputation
    print("### FINAL MISSING VALUE SUMMARY ###")
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (df.isna().mean() * 100).round(2)
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    print(missing_df[missing_df['missing_pct'] > 0])
    
    # 5) Drop remaining NA rows
    df = df.dropna()
    print("\n### ALL REMAINING NA VALUES REMOVED ###\n")
    
    return df



def skewness_plot(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    numeric_cols = df.select_dtypes(include=['number']).columns

    if len(numeric_cols) == 0:
        print("No numeric columns found.")
        return

    print("### SKEWNESS VALUES ###\n")
    for col in numeric_cols:
        print(f"{col}: {df[col].skew():.4f}")
    print("\n")

    n = len(numeric_cols)
    rows = int(np.ceil(n / 3))

    plt.figure(figsize=(12, 4 * rows))

    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(rows, 3, i)
        sns.histplot(df[col].dropna(), bins=30, kde=True, color='#3498db')
        plt.title(f"{col}\nskew={df[col].skew():.2f}")

    plt.tight_layout()
    plt.show()


def log_transform(df):
    df=df.copy()
    df.loc[:, "tot_coll_amt_log"] = np.log1p(df["tot_coll_amt"])
    df.loc[:, "annual_inc_log"] = np.log1p(df["annual_inc"])
    df.loc[:, "tot_cur_bal_log"] = np.log1p(df["tot_cur_bal"])


    df.loc[:, "tot_coll_amt"] = np.log1p(df["tot_coll_amt"])

    df.loc[:, 'delinq_bucket'] = df['delinq_2yrs'].apply(
        lambda x: 0 if x == 0 else 1 if x == 1 else 2
    ).astype('category')

    print("Log transform steps completed.\n")
    return df.copy()

