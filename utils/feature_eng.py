import numpy as np

def segment_emp_length(x):
    if x == 0:
        return "no_exp"
    elif 1 <= x <= 3:
        return "junior_exp"
    elif 3 < x <= 7:
        return "mid_exp"
    else:
        return "senior_exp"
    

def segment_job(title):
    if title in ['Data Scientist', 'Developer', 'Engineer']:
        return 'tech'
    elif title in ['Analyst', 'Consultant']:
        return 'professional'
    else:
        return 'management'
    
    
def create_features(df, leak_cols):
    # 1) Leakage kolonlarını sil
    leak_cols_present = [c for c in leak_cols if c in df.columns]
    df = df.drop(columns=leak_cols_present).copy()

    # ---- Feature Engineering ----

    # Utilization ratio
    df.loc[:, 'utilization_ratio'] = df['tot_cur_bal'] / df['total_rev_hi_lim']
    df['utilization_ratio'] = df['utilization_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Debt to income
    df.loc[:, 'debt_to_income'] = df['tot_cur_bal'] / df['annual_inc']

    # Installment to income
    df.loc[:, 'installment_to_income'] = df['installment'] / (df['annual_inc'] / 12)
    df.loc[:, 'income_to_installment'] = (df['annual_inc'] / 12) / df['installment']

    # Revolver ratio
    df.loc[:, 'revolver_ratio'] = df['tot_cur_bal'] / df['total_acc']

    # Term num
    df.loc[:, 'term_num'] = df['term'].astype(int)

    # Total installment amount & interest rate
    df.loc[:, 'total_installment_amt'] = df['installment'] * df['term_num']
    df.loc[:, 'interest_rate'] = df['total_installment_amt'] / df['funded_amnt_inv']

    df = df.drop(columns=['term_num'])

    # Loan to income
    df.loc[:, 'loan_to_income'] = df['loan_amnt'] / df['annual_inc']
    df['loan_to_income'] = df['loan_to_income'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Loan to revolving limit
    df.loc[:, 'loan_to_limit'] = df['loan_amnt'] / df['total_rev_hi_lim']

    # Open accounts ratio
    df.loc[:, 'open_acc_ratio'] = df['open_acc'] / df['total_acc']

    # Ever delinquent flag
    df.loc[:, 'ever_delinq'] = np.where(df['delinq_2yrs'] > 0, 'True', 'False').astype(object)

    # Çalışma süresi ve title segmentleri
    df.loc[:, 'emp_length_segment'] = df['emp_length_num'].apply(segment_emp_length)
    df['emp_length_segment'] = df['emp_length_segment'].astype('category')

    df.loc[:, 'emp_title_segment'] = df['emp_title'].apply(segment_job)
    df['emp_title_segment'] = df['emp_title_segment'].astype('category')

    # Artık kullanılmayacak kolonlar
    df = df.drop(columns=['emp_title', 'emp_length'])

    print("Feature engineering işlemleri tamamlandı.\n")

    return df
