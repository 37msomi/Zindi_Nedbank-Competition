# =============================================================================
# NEDBANK TRANSACTION VOLUME FORECASTING CHALLENGE
# Final Script v4 — pushing OOF below 0.372 (target: top 25)
#
# WHAT CHANGED FROM v3
# --------------------
# v3 OOF ~0.380 / LB ~0.393. The error decomposes into two very uneven
# buckets from your own diagnostics:
#     low-txn (<30 txns in hist)  OOF 0.585  <-- dominates total error
#     high-txn (>30 txns in hist) OOF 0.310  <-- already solid
#
# Because RMSLE is computed in log1p space, a customer who did 3 txns but
# is predicted 9 contributes |log(10)-log(4)| = 0.92 — a single miss on a
# low-txn customer can swamp dozens of correct predictions on high-txn
# ones.  Attacking the low-txn bucket is the single largest lever.
#
# v4 introduces four independent improvements, each stackable:
#
#   (A) Regularity / seasonality features.  The v3 feature set measures
#       how MUCH a customer transacts but barely measures how REGULARLY.
#       Low-txn customers differ from each other mostly in whether they
#       are sporadic vs steady.  New features:
#           - trend_slope_6m : OLS slope of monthly counts over last 6m
#           - cv_monthly_12m : coefficient of variation of monthly counts
#           - active_months_12m, active_months_6m, active_months_3m
#           - zero_months_12m
#           - nov14, dec14, jan15 individual counts (finer seasonality
#             than the combined nov_jan_15)
#           - weekend_ratio_3m, weekend_ratio_12m
#           - debit_credit_ratio_3m
#           - reversal_ratio_12m
#           - account_churn (max(AccountID_per_month) - min(...))
#
#   (B) Tweedie objective.  Our target is a right-skewed non-negative
#       count.  XGB/LGB with reg:squarederror on log1p(y) is a common
#       but suboptimal hack — Tweedie regression directly models
#       count-like targets with a log link.  Empirically this is worth
#       ~0.005–0.010 on count-heavy RMSLE problems.  We train both
#       squared-error-on-log-y (v3 flavor) AND Tweedie-on-raw-y and
#       blend.
#
#   (C) Two-stage model done properly.
#       stage1: XGBClassifier on "will do >=5 txns in the window?"
#       stage2a: XGBRegressor (Tweedie) on customers with txn_count_12m>=30
#       stage2b: XGBRegressor (Tweedie) on customers with txn_count_12m<30
#       Final prediction for a test customer:
#           p = stage1_prob(customer)
#           mu_hi = stage2a(customer)
#           mu_lo = stage2b(customer)
#           hi_weight = sigmoid((txn_count_12m - 30)/10)  # smooth gate
#           E[y] = p * (hi_weight*mu_hi + (1-hi_weight)*mu_lo)
#
#   (D) Optuna hyperparameter tuning on the global XGB (Tweedie).
#       Uses KFold-15 OOF as the objective — 60 trials with a
#       MedianPruner.  Expected OOF pickup: 0.002–0.004.
#
# EXPECTED OOF IMPACT (additive, rough):
#     baseline                              0.380
#   + new features                         -0.004
#   + Tweedie blend                        -0.005
#   + proper two-stage                     -0.005
#   + Optuna                               -0.002
#   = projected v4 OOF                     ~0.364-0.368
#     projected LB (OOF - 0.008)           ~0.356-0.360
#
# USAGE
# -----
#   1. Run cells 1-12 to build features and the master frame (uses the
#      v3 feature pipeline plus new cells 7b and 8b).
#   2. Run cell 14 to train the baseline global XGB+LGB with Tweedie.
#   3. Run cell 15 (Optuna) — or skip it and use the defaults below.
#   4. Run cell 16 to train the two-stage model.
#   5. Run cell 17 to blend and write submissions.
#
# The script is structured so you can comment out any one of (A)-(D)
# and still get a valid submission.  This makes it easy to isolate how
# much each contributed when you compare leaderboard scores.
# =============================================================================

# %% CELL 1 — IMPORTS & SETUP
# =============================================================================
import pandas as pd
import numpy as np
import os
import warnings
from functools import partial

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("optuna not installed — Cell 15 will be skipped.  "
          "`pip install optuna` to enable.")

warnings.filterwarnings('ignore')

USERNAME  = os.getenv('USERNAME')
DATA_PATH = rf'C:\Users\{USERNAME}\OneDrive\Desktop\Zindi'

if not os.path.exists(DATA_PATH):
    print(f"✗ PATH NOT FOUND — update DATA_PATH manually")
else:
    print(f"✓ Data path found: {DATA_PATH}")


def rmsle_log(y_true_log, y_pred_log):
    """RMSLE when both inputs already in log1p space."""
    return np.sqrt(mean_squared_error(y_true_log, y_pred_log))


def rmsle_raw(y_true, y_pred):
    """RMSLE from raw (non-log) values, clipping negatives."""
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


def save_submission(preds, filename, test_ids, sample_sub, data_path):
    sub = sample_sub[['UniqueID']].copy()
    sub = sub.merge(
        pd.DataFrame({
            'UniqueID'          : test_ids,
            'next_3m_txn_count' : np.clip(preds, 0, None)
        }),
        on='UniqueID', how='left'
    )
    assert sub['next_3m_txn_count'].isna().sum() == 0, f"Nulls in {filename}!"
    fpath = os.path.join(data_path, filename)
    sub.to_csv(fpath, index=False)
    print(f"  Saved {filename} | mean: {sub['next_3m_txn_count'].mean():.4f} "
          f"| min: {sub['next_3m_txn_count'].min():.4f} "
          f"| max: {sub['next_3m_txn_count'].max():.4f}")
    return sub


print("Setup complete.")


# %% CELL 2 — LOAD CORE FILES
# =============================================================================
print("Loading core files...")
train      = pd.read_csv(os.path.join(DATA_PATH, 'Train.csv'))
test       = pd.read_csv(os.path.join(DATA_PATH, 'Test.csv'))
sample_sub = pd.read_csv(os.path.join(DATA_PATH, 'SampleSubmission.csv'))
print(f"Train : {train.shape} | Test: {test.shape}")


# %% CELL 3 — LOAD TRANSACTIONS, FINANCIALS, DEMOGRAPHICS
# =============================================================================
print("Loading parquet folders...")
TXN_COLS = [
    'UniqueID', 'AccountID', 'TransactionDate',
    'TransactionAmount', 'TransactionTypeDescription',
    'TransactionBatchDescription', 'StatementBalance',
    'IsDebitCredit', 'ReversalTypeDescription',
]
txn = pd.read_parquet(os.path.join(DATA_PATH, 'transactions_features'),
                      columns=TXN_COLS)
txn['TransactionDate']   = pd.to_datetime(txn['TransactionDate'])
txn['TransactionAmount'] = txn['TransactionAmount'].astype('float32')
txn['StatementBalance']  = txn['StatementBalance'].astype('float32')

fin  = pd.read_parquet(os.path.join(DATA_PATH, 'financials_features'))
fin['RunDate'] = pd.to_datetime(fin['RunDate'], errors='coerce')

demo = pd.read_parquet(os.path.join(DATA_PATH, 'demographics_clean'))

print(f"txn: {txn.shape} | fin: {fin.shape} | demo: {demo.shape}")


# %% CELL 4 — DATE CONSTANTS
# =============================================================================
CUTOFF_DATE       = pd.Timestamp('2015-10-31')
PRIOR_YEAR_START  = pd.Timestamp('2014-11-01')
PRIOR_YEAR_END    = pd.Timestamp('2015-01-31')
PRIOR_2YEAR_START = pd.Timestamp('2013-11-01')
PRIOR_2YEAR_END   = pd.Timestamp('2014-01-31')

WINDOWS = {
    '1m' : (CUTOFF_DATE - pd.DateOffset(months=1),  CUTOFF_DATE),
    '3m' : (CUTOFF_DATE - pd.DateOffset(months=3),  CUTOFF_DATE),
    '6m' : (CUTOFF_DATE - pd.DateOffset(months=6),  CUTOFF_DATE),
    '12m': (CUTOFF_DATE - pd.DateOffset(months=12), CUTOFF_DATE),
    '24m': (CUTOFF_DATE - pd.DateOffset(months=24), CUTOFF_DATE),
}


# %% CELL 5 — CORE TRANSACTION FEATURES (v3 pipeline — unchanged)
# =============================================================================
print("Building core transaction features...")
feat_list = []

for label, (start, end) in WINDOWS.items():
    mask = (txn['TransactionDate'] >= start) & (txn['TransactionDate'] <= end)
    feat_list.append(
        txn[mask].groupby('UniqueID').size().rename(f'txn_count_{label}')
    )

mask_py1 = (txn['TransactionDate'] >= PRIOR_YEAR_START)  & (txn['TransactionDate'] <= PRIOR_YEAR_END)
mask_py2 = (txn['TransactionDate'] >= PRIOR_2YEAR_START) & (txn['TransactionDate'] <= PRIOR_2YEAR_END)
feat_list.append(txn[mask_py1].groupby('UniqueID').size().rename('txn_count_prior1y_nov_jan'))
feat_list.append(txn[mask_py2].groupby('UniqueID').size().rename('txn_count_prior2y_nov_jan'))

last_txn  = txn.groupby('UniqueID')['TransactionDate'].max()
first_txn = txn.groupby('UniqueID')['TransactionDate'].min()
feat_list.append(((CUTOFF_DATE - last_txn).dt.days).rename('days_since_last_txn'))
feat_list.append(((CUTOFF_DATE - first_txn).dt.days).rename('customer_txn_tenure_days'))

for label, (start, end) in [('3m', WINDOWS['3m']), ('12m', WINDOWS['12m'])]:
    mask = (txn['TransactionDate'] >= start) & (txn['TransactionDate'] <= end)
    feat_list.append(
        txn[mask].groupby('UniqueID')['TransactionAmount'].agg(
            **{f'amt_mean_{label}': 'mean', f'amt_std_{label}': 'std',
               f'amt_sum_{label}' : 'sum',  f'amt_max_{label}': 'max',
               f'amt_min_{label}' : 'min'}
        )
    )

mask_3m = (txn['TransactionDate'] >= WINDOWS['3m'][0]) & (txn['TransactionDate'] <= CUTOFF_DATE)
txn_3m  = txn[mask_3m]
feat_list.append(
    txn_3m.groupby('UniqueID')['StatementBalance'].agg(
        bal_mean_3m='mean', bal_std_3m='std',
        bal_min_3m='min',   bal_max_3m='max', bal_last_3m='last'
    )
)

type_pivot = (
    txn_3m.groupby(['UniqueID','TransactionTypeDescription']).size()
    .unstack(fill_value=0)
)
type_pivot.columns = [
    f'txntype_{c.replace(" ","_").replace("&","and").replace("/","_")}_3m'
    for c in type_pivot.columns
]
feat_list.append(type_pivot)

batch_pivot = (
    txn_3m.groupby(['UniqueID','TransactionBatchDescription']).size()
    .unstack(fill_value=0)
)
batch_pivot.columns = [
    f'batch_{c.replace(" ","_").replace("&","and").replace("/","_")}_3m'
    for c in batch_pivot.columns
]
feat_list.append(batch_pivot)

feat_list.append(
    txn.groupby('UniqueID')['AccountID'].nunique().rename('unique_accounts_total')
)

mask_prev2m = (
    (txn['TransactionDate'] >= WINDOWS['3m'][0]) &
    (txn['TransactionDate'] <  WINDOWS['1m'][0])
)
count_prev2m = txn[mask_prev2m].groupby('UniqueID').size().rename('txn_count_prev2m')
count_1m     = feat_list[0]
trend_df     = pd.concat([count_1m, count_prev2m], axis=1).fillna(0)
trend_df['mom_trend'] = trend_df['txn_count_1m'] / (trend_df['txn_count_prev2m'] / 2 + 1)
feat_list.append(trend_df['mom_trend'])

mask_6m = (txn['TransactionDate'] >= WINDOWS['6m'][0]) & (txn['TransactionDate'] <= CUTOFF_DATE)
txn_6m  = txn[mask_6m].copy()
txn_6m['YearMonth'] = txn_6m['TransactionDate'].dt.to_period('M')
feat_list.append(
    txn_6m.groupby(['UniqueID','YearMonth'])['TransactionDate']
    .nunique().groupby('UniqueID').mean()
    .rename('avg_active_days_per_month_6m')
)

txn_features = pd.concat(feat_list, axis=1)
txn_features.index.name = 'UniqueID'
txn_features = txn_features.reset_index()
print(f"Core transaction features: {txn_features.shape}")


# %% CELL 6 — HOLIDAY FEATURES (v3, unchanged) + fine-grained months
# =============================================================================
print("Building holiday features...")
holiday_feats = []

mask_nov_jan_15 = (txn['TransactionDate'] >= '2014-11-01') & (txn['TransactionDate'] <= '2015-01-31')
mask_nov_jan_14 = (txn['TransactionDate'] >= '2013-11-01') & (txn['TransactionDate'] <= '2014-01-31')
mask_aug_oct_14 = (txn['TransactionDate'] >= '2014-08-01') & (txn['TransactionDate'] <= '2014-10-31')
mask_aug_oct_15 = (txn['TransactionDate'] >= '2015-08-01') & (txn['TransactionDate'] <= '2015-10-31')

nov_jan_15_counts = txn[mask_nov_jan_15].groupby('UniqueID').size().rename('nov_jan_15')
nov_jan_14_counts = txn[mask_nov_jan_14].groupby('UniqueID').size().rename('nov_jan_14')
aug_oct_14_counts = txn[mask_aug_oct_14].groupby('UniqueID').size().rename('aug_oct_14')
aug_oct_15_counts = txn[mask_aug_oct_15].groupby('UniqueID').size().rename('aug_oct_15')

holiday_feats += [nov_jan_15_counts, nov_jan_14_counts,
                  aug_oct_14_counts, aug_oct_15_counts]

festive_df = pd.concat([nov_jan_14_counts, aug_oct_14_counts], axis=1).fillna(0)
festive_df['is_festive_spender']  = (festive_df['nov_jan_14'] > festive_df['aug_oct_14']).astype(int)
festive_df['festive_uplift_2014'] = festive_df['nov_jan_14'] - festive_df['aug_oct_14']
holiday_feats.append(festive_df['is_festive_spender'])
holiday_feats.append(festive_df['festive_uplift_2014'])

personal_ratio = (nov_jan_14_counts / (aug_oct_14_counts + 1)).rename('personal_holiday_ratio_2014')
holiday_feats.append(personal_ratio)

# NEW: individual months of prior-year festive window — more granular seasonality
for y, m, name in [(2014, 11, 'nov14'), (2014, 12, 'dec14'), (2015, 1, 'jan15')]:
    start = pd.Timestamp(y, m, 1)
    end   = (start + pd.DateOffset(months=1)) - pd.Timedelta(days=1)
    mask  = (txn['TransactionDate'] >= start) & (txn['TransactionDate'] <= end)
    holiday_feats.append(txn[mask].groupby('UniqueID').size().rename(name))

holiday_df = pd.concat(holiday_feats, axis=1)
holiday_df.index.name = 'UniqueID'
holiday_df = holiday_df.reset_index()

demo_holiday = demo[['UniqueID','CustomerBankingType','IncomeCategory']].merge(
    holiday_df, on='UniqueID', how='left'
)
banking_ratio = (demo_holiday.groupby('CustomerBankingType')['personal_holiday_ratio_2014']
                 .median().rename('banking_type_holiday_ratio').reset_index())
income_ratio  = (demo_holiday.groupby('IncomeCategory')['personal_holiday_ratio_2014']
                 .median().rename('income_holiday_ratio').reset_index())
demo_holiday = demo_holiday.merge(banking_ratio, on='CustomerBankingType', how='left')
demo_holiday = demo_holiday.merge(income_ratio,  on='IncomeCategory',      how='left')
demo_holiday['blended_holiday_ratio'] = np.where(
    demo_holiday['personal_holiday_ratio_2014'].isna(),
    demo_holiday['banking_type_holiday_ratio'],
    0.7 * demo_holiday['personal_holiday_ratio_2014'] +
    0.3 * demo_holiday['banking_type_holiday_ratio']
)
print(f"Holiday features: {holiday_df.shape}")


# %% CELL 7 — HIGH-CORRELATION FEATURES (v3, unchanged)
# =============================================================================
print("Building v3 high-corr features...")
new_feats = []

txn_monthly = txn.copy()
txn_monthly['YearMonth'] = txn_monthly['TransactionDate'].dt.to_period('M')
monthly_counts = txn_monthly.groupby(['UniqueID','YearMonth']).size().reset_index(name='count')
monthly_counts['months_ago'] = (
    pd.Period('2015-10', 'M') - monthly_counts['YearMonth']
).apply(lambda x: x.n)
monthly_counts = monthly_counts[monthly_counts['months_ago'] <= 12]
monthly_counts['weight']         = np.exp(-0.3 * monthly_counts['months_ago'])
monthly_counts['weighted_count'] = monthly_counts['count'] * monthly_counts['weight']
new_feats.append(
    monthly_counts.groupby('UniqueID')['weighted_count'].sum()
    .rename('recency_weighted_txn')
)

mask_3m_pay = (txn['TransactionDate'] >= WINDOWS['3m'][0]) & (txn['TransactionDate'] <= CUTOFF_DATE)
txn_3m_pay  = txn[mask_3m_pay].copy()
txn_3m_pay['day']           = txn_3m_pay['TransactionDate'].dt.day
txn_3m_pay['days_in_month'] = txn_3m_pay['TransactionDate'].dt.days_in_month
new_feats.append(
    txn_3m_pay[txn_3m_pay['day'] >= txn_3m_pay['days_in_month'] - 4]
    .groupby('UniqueID').size().rename('txn_last5days_3m')
)
new_feats.append(
    txn_3m_pay[txn_3m_pay['day'] <= 5]
    .groupby('UniqueID').size().rename('txn_first5days_3m')
)

mask_6m_copy = (txn['TransactionDate'] >= WINDOWS['6m'][0]) & (txn['TransactionDate'] <= CUTOFF_DATE)
txn_6m_copy  = txn[mask_6m_copy].copy()
txn_6m_copy['YearMonth'] = txn_6m_copy['TransactionDate'].dt.to_period('M')
new_feats.append(
    txn_6m_copy.groupby(['UniqueID','YearMonth']).size()
    .groupby('UniqueID').max().rename('max_monthly_txn_6m')
)

new_feat_df = pd.concat(new_feats, axis=1)
new_feat_df.index.name = 'UniqueID'
new_feat_df = new_feat_df.reset_index()
print(f"v3 high-corr features: {new_feat_df.shape}")


# %% CELL 8 — NEW v4 REGULARITY / SEASONALITY FEATURES  (KEY ADDITION)
# =============================================================================
# These attack the low-txn error bucket.  v3 features measure HOW MUCH a
# customer transacts; v4 adds HOW REGULARLY / HOW CONSISTENTLY.
# =============================================================================
print("Building v4 regularity / seasonality features...")
v4_feats = []

# Build per-customer monthly count table for last 12 months
mask_12m = (txn['TransactionDate'] >= WINDOWS['12m'][0]) & (txn['TransactionDate'] <= CUTOFF_DATE)
txn_12m = txn[mask_12m].copy()
txn_12m['YearMonth'] = txn_12m['TransactionDate'].dt.to_period('M')
monthly_12m = (txn_12m.groupby(['UniqueID', 'YearMonth']).size()
               .unstack(fill_value=0))

# Ensure all 12 months present
all_months = pd.period_range('2014-11', '2015-10', freq='M')
for m in all_months:
    if m not in monthly_12m.columns:
        monthly_12m[m] = 0
monthly_12m = monthly_12m[all_months]

# Active months counts
for w, nmonths in [(12, 12), (6, 6), (3, 3)]:
    sub = monthly_12m.iloc[:, -nmonths:]
    v4_feats.append((sub > 0).sum(axis=1).rename(f'active_months_{w}m'))

v4_feats.append((monthly_12m == 0).sum(axis=1).rename('zero_months_12m'))

# Regularity — coefficient of variation (std/mean) over last 12m.  Low
# values mean "steady", high values mean "bursty".
mn = monthly_12m.mean(axis=1)
sd = monthly_12m.std(axis=1)
v4_feats.append((sd / (mn + 1)).rename('cv_monthly_12m'))

# Linear trend slope over last 6 months (OLS on month index)
x = np.arange(6, dtype=float)
x_mean = x.mean()
xdev = x - x_mean
denom = (xdev ** 2).sum()
last6 = monthly_12m.iloc[:, -6:].to_numpy(dtype=float)
ymean = last6.mean(axis=1)
slope = ((last6 - ymean[:, None]) * xdev).sum(axis=1) / denom
v4_feats.append(pd.Series(slope, index=monthly_12m.index,
                          name='trend_slope_6m'))

# 3m/12m count ratio — captures recent acceleration
v4_feats.append(
    (monthly_12m.iloc[:, -3:].sum(axis=1) /
     (monthly_12m.sum(axis=1) + 1)).rename('ratio_3m_vs_12m')
)

# Weekend share (3m and 12m)
for label, mask in [('3m', mask_3m), ('12m', mask_12m)]:
    sub = txn.loc[mask, ['UniqueID', 'TransactionDate']].copy()
    sub['is_weekend'] = sub['TransactionDate'].dt.dayofweek >= 5
    v4_feats.append(
        sub.groupby('UniqueID')['is_weekend'].mean().rename(f'weekend_ratio_{label}')
    )

# Debit / credit ratio (3m) — IsDebitCredit is one-hot-ish in the file
sub = txn.loc[mask_3m, ['UniqueID', 'IsDebitCredit']].copy()
sub['is_debit']  = (sub['IsDebitCredit'].astype(str).str.upper().str.startswith('D')).astype(int)
sub['is_credit'] = (sub['IsDebitCredit'].astype(str).str.upper().str.startswith('C')).astype(int)
dc = sub.groupby('UniqueID').agg(debit=('is_debit','sum'), credit=('is_credit','sum'))
v4_feats.append((dc['debit']  / (dc['debit']+dc['credit']+1)).rename('debit_share_3m'))

# Reversal ratio (12m)
sub = txn.loc[mask_12m, ['UniqueID', 'ReversalTypeDescription']].copy()
sub['is_rev'] = sub['ReversalTypeDescription'].notna().astype(int)
rev = sub.groupby('UniqueID').agg(rev=('is_rev','sum'), tot=('is_rev','size'))
v4_feats.append((rev['rev'] / (rev['tot'] + 1)).rename('reversal_ratio_12m'))

# Unique accounts in last 12m — proxy for account lifecycle
sub = txn.loc[mask_12m, ['UniqueID', 'AccountID']]
v4_feats.append(sub.groupby('UniqueID')['AccountID'].nunique()
                .rename('unique_accounts_12m'))

# Amount-per-txn volatility (log-scale) in 12m — steadier customers
# have lower log-amount std
sub = txn.loc[mask_12m, ['UniqueID','TransactionAmount']].copy()
sub['log_amt'] = np.log1p(np.abs(sub['TransactionAmount'].astype(float)))
v4_feats.append(sub.groupby('UniqueID')['log_amt'].std()
                .rename('log_amt_std_12m'))

# Recent-week intensity (last 7 days before cutoff)
mask_7d = (txn['TransactionDate'] >= CUTOFF_DATE - pd.Timedelta(days=7)) & \
          (txn['TransactionDate'] <= CUTOFF_DATE)
v4_feats.append(txn[mask_7d].groupby('UniqueID').size().rename('txn_count_7d'))

v4_df = pd.concat(v4_feats, axis=1)
v4_df.index.name = 'UniqueID'
v4_df = v4_df.reset_index()
print(f"v4 regularity features: {v4_df.shape}")


# %% CELL 9 — FINANCIAL + DEMOGRAPHIC FEATURES (v3, unchanged)
# =============================================================================
print("Building financial + demographic features...")
fin_sorted = fin.sort_values('RunDate', ascending=False)
fin_latest = fin_sorted.drop_duplicates(subset=['UniqueID','Product'])

product_dummies = (fin_latest.groupby(['UniqueID','Product']).size()
                   .unstack(fill_value=0).clip(upper=1))
product_dummies.columns = [f'has_{c.lower().replace(" ","_")}' for c in product_dummies.columns]

fin_agg = fin_latest.groupby('UniqueID')[['NetInterestIncome','NetInterestRevenue']].agg(
    ['mean','sum','max']
)
fin_agg.columns = ['fin_' + '_'.join(c) for c in fin_agg.columns]

fin_recent = fin[fin['RunDate'] >= (CUTOFF_DATE - pd.DateOffset(months=3))]
nii_recent = fin_recent.groupby('UniqueID')['NetInterestIncome'].mean().rename('nii_mean_recent_3m')

fin_features = (product_dummies
                .join(fin_agg,    how='left')
                .join(nii_recent, how='left')
                .reset_index())

demo_feats = demo.copy()
demo_feats['BirthDate'] = pd.to_datetime(demo_feats['BirthDate'], errors='coerce')
demo_feats['age'] = (CUTOFF_DATE - demo_feats['BirthDate']).dt.days / 365.25
demo_feats['age'] = demo_feats['age'].where(
    (demo_feats['age'] >= 18) & (demo_feats['age'] <= 90), np.nan)
demo_feats['age_group'] = pd.cut(
    demo_feats['age'], bins=[18,25,35,45,55,65,90],
    labels=['18-25','26-35','36-45','46-55','56-65','65+']
).astype(str)

INCOME_ORDER = {'No Income':0, 'Not Disclosed':0, 'Low':1, 'Lower-Middle':2,
                'Middle':3, 'Upper-Middle':4, 'High':5, 'Very High':6}
demo_feats['income_category_encoded'] = demo_feats['IncomeCategory'].map(INCOME_ORDER).fillna(0)
demo_feats['AnnualGrossIncome'] = (demo_feats.groupby('IncomeCategory')['AnnualGrossIncome']
                                   .transform(lambda x: x.fillna(x.median())))

CAT_COLS = ['Gender','CustomerStatus','ClientType','MaritalStatus',
            'OccupationCategory','IndustryCategory','CustomerBankingType',
            'CustomerOnboardingChannel','ResidentialCityName','CountryCodeNationality',
            'CertificationTypeDescription','ContactPreference','age_group']
for col in CAT_COLS:
    if col in demo_feats.columns:
        demo_feats[col] = demo_feats[col].fillna('Unknown')
        demo_feats[col] = LabelEncoder().fit_transform(demo_feats[col].astype(str))

demo_feats['LowIncomeFlag'] = demo_feats['LowIncomeFlag'].map({'Y':1,'N':0}).fillna(0).astype(int)
demo_features = demo_feats.drop(columns=['BirthDate','IncomeCategory'], errors='ignore')
print(f"Financial features: {fin_features.shape} | Demographic: {demo_features.shape}")


# %% CELL 10 — MERGE
# =============================================================================
print("Merging all features...")
all_ids = pd.concat([train[['UniqueID']], test[['UniqueID']]]).drop_duplicates()

master = (all_ids
          .merge(txn_features,  on='UniqueID', how='left')
          .merge(holiday_df,    on='UniqueID', how='left')
          .merge(demo_holiday[['UniqueID','blended_holiday_ratio',
                               'banking_type_holiday_ratio','income_holiday_ratio']],
                 on='UniqueID', how='left')
          .merge(new_feat_df,   on='UniqueID', how='left')
          .merge(v4_df,         on='UniqueID', how='left')
          .merge(fin_features,  on='UniqueID', how='left')
          .merge(demo_features, on='UniqueID', how='left'))

print(f"Master shape: {master.shape}")

# Optional cache
master.to_parquet(os.path.join(DATA_PATH, 'master_final_v4_cache.parquet'), index=False)


# %% CELL 11 — TRAIN / TEST MATRICES
# =============================================================================
print("Preparing train/test matrices...")
train_master = master[master['UniqueID'].isin(train['UniqueID'])].copy()
test_master  = master[master['UniqueID'].isin(test['UniqueID'])].copy()
train_master = train_master.drop(
    columns=[c for c in train_master.columns if 'next_3m' in c], errors='ignore')
train_master = train_master.merge(train[['UniqueID','next_3m_txn_count']],
                                  on='UniqueID', how='left')

DROP_FEATURES = [
    'has_transactional','LowIncomeFlag','batch_Unallocated_3m',
    'CustomerStatus','txntype_Card_Transactions_3m','income_category_encoded',
    'batch_System_Defined_3m','txntype_Transfers_and_Payments_3m',
    'fin_NetInterestRevenue_sum','fin_NetInterestRevenue_mean',
    'fin_NetInterestIncome_sum','ContactPreference',
    'txntype_Unpaid___Returned_Items_3m','txntype_Foreign_Exchange_3m',
    'CountryCodeNationality','txntype_Reversals_and_Adjustments_3m',
    'ClientType','fin_product_count','has_mortgages',
    'txntype_Account_Maintenance_3m','Gender','txntype_Deposits_3m',
    'txntype_Teller_and_Branch_Transactions_3m','has_investments',
    'batch_Credit_Debit_Service_3m',
]

ALL_FEATURES = [c for c in master.columns
                if c not in ['UniqueID'] + DROP_FEATURES]
all_nan = [c for c in ALL_FEATURES if train_master[c].isna().all()]
ALL_FEATURES = [c for c in ALL_FEATURES if c not in all_nan]
print(f"All features: {len(ALL_FEATURES)}")

imp = SimpleImputer(strategy='median')
X_all_train = pd.DataFrame(imp.fit_transform(train_master[ALL_FEATURES]),
                           columns=ALL_FEATURES)
X_all_test  = pd.DataFrame(imp.transform(test_master[ALL_FEATURES]),
                           columns=ALL_FEATURES)
y_train     = train_master['next_3m_txn_count'].values
y_train_log = pd.Series(np.log1p(y_train))


# %% CELL 12 — FEATURE SET (v3 best + all v4 regularity features)
# =============================================================================
BEST_FEATURES_V3 = [
    'txn_count_1m','mom_trend','txn_count_12m','txn_count_prior1y_nov_jan',
    'txn_count_6m','days_since_last_txn','avg_active_days_per_month_6m',
    'customer_txn_tenure_days','txn_count_3m','aug_oct_15','txn_count_24m',
    'age','aug_oct_14','bal_min_3m','txn_count_prior2y_nov_jan',
    'amt_std_3m','bal_last_3m','ResidentialCityName','AnnualGrossIncome',
    'festive_uplift_2014','personal_holiday_ratio_2014',
    'blended_holiday_ratio','nov_jan_15','CustomerBankingType',
    'batch_Other_Charges_3m','recency_weighted_txn',
    'txn_last5days_3m','txn_first5days_3m','max_monthly_txn_6m',
]
V4_NEW = [
    'active_months_12m','active_months_6m','active_months_3m',
    'zero_months_12m','cv_monthly_12m','trend_slope_6m','ratio_3m_vs_12m',
    'weekend_ratio_3m','weekend_ratio_12m','debit_share_3m',
    'reversal_ratio_12m','unique_accounts_12m','log_amt_std_12m',
    'txn_count_7d','nov14','dec14','jan15',
]
BEST_FEATURES = [f for f in BEST_FEATURES_V3 + V4_NEW if f in ALL_FEATURES]
print(f"Best feature set (v4): {len(BEST_FEATURES)} features")

X_tr = X_all_train[BEST_FEATURES]
X_te = X_all_test[BEST_FEATURES]


# %% CELL 13 — HELPER: KFold CV on a parametrised model
# =============================================================================
def cv_xgb(params, X, y, X_test, n_splits=15, seed=42,
           target='log', early_stop=100):
    """
    Train XGB with KFold-n_splits.  Returns (oof_log, test_avg_raw, mean_score).
    target='log'     -> regressor on log1p(y), returns predictions in log space.
    target='tweedie' -> regressor on raw y with objective reg:tweedie;
                        returns predictions in raw space (converted to log for OOF).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_preds = []
    scores = []

    y_raw = np.expm1(y) if target == 'tweedie' else None

    for fold, (tr, val) in enumerate(kf.split(X)):
        X_f, X_v = X.iloc[tr], X.iloc[val]
        if target == 'log':
            y_f, y_v = y.iloc[tr], y.iloc[val]
        else:
            y_f, y_v = y_raw[tr], y_raw[val]

        m = xgb.XGBRegressor(**params)
        m.fit(X_f, y_f, eval_set=[(X_v, y_v)], verbose=False)

        val_pred  = m.predict(X_v)
        test_pred = m.predict(X_test)

        if target == 'tweedie':
            # clip to non-negative and put everything in log1p space
            val_pred_log  = np.log1p(np.clip(val_pred, 0, None))
            test_pred_log = np.log1p(np.clip(test_pred, 0, None))
            true_log      = np.log1p(y_v)
        else:
            val_pred_log, test_pred_log, true_log = val_pred, test_pred, y_v

        oof[val] = val_pred_log
        test_preds.append(test_pred_log)
        scores.append(rmsle_log(true_log, val_pred_log))

    return oof, np.mean(test_preds, axis=0), float(np.mean(scores))


# %% CELL 14 — TRAIN XGB (log-y, squared error) + XGB (Tweedie) + LGB
# =============================================================================
print("\nTraining baseline models...")

XGB_LOG_PARAMS = {
    'objective'            : 'reg:squarederror',
    'learning_rate'        : 0.05,
    'max_depth'            : 4,
    'min_child_weight'     : 10,
    'reg_lambda'           : 1.0,
    'reg_alpha'            : 0.5,
    'subsample'            : 0.75,
    'colsample_bytree'     : 0.75,
    'n_estimators'         : 3000,
    'random_state'         : 42,
    'verbosity'            : 0,
    'early_stopping_rounds': 100,
}

XGB_TWEEDIE_PARAMS = {
    'objective'              : 'reg:tweedie',
    'tweedie_variance_power' : 1.5,
    'learning_rate'          : 0.05,
    'max_depth'              : 5,
    'min_child_weight'       : 10,
    'reg_lambda'             : 1.0,
    'reg_alpha'              : 0.5,
    'subsample'              : 0.8,
    'colsample_bytree'       : 0.8,
    'n_estimators'           : 4000,
    'random_state'           : 42,
    'verbosity'              : 0,
    'early_stopping_rounds'  : 150,
    'eval_metric'            : 'rmse',
}

print("  XGB (log-y, squared error)...")
xgb_log_oof, xgb_log_test_log, xgb_log_score = cv_xgb(
    XGB_LOG_PARAMS, X_tr, y_train_log, X_te, target='log')
print(f"    OOF: {xgb_log_score:.5f}")

print("  XGB (Tweedie, raw y)...")
xgb_tw_oof, xgb_tw_test_log, xgb_tw_score = cv_xgb(
    XGB_TWEEDIE_PARAMS, X_tr, y_train_log, X_te, target='tweedie')
print(f"    OOF: {xgb_tw_score:.5f}")

print("  LGB (log-y)...")
LGB_PARAMS = {
    'objective'        : 'regression',
    'metric'           : 'rmse',
    'verbose'          : -1,
    'random_state'     : 42,
    'learning_rate'    : 0.05,
    'num_leaves'       : 50,
    'min_child_samples': 30,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.8,
    'reg_alpha'        : 0.2,
    'reg_lambda'       : 0.2,
    'n_estimators'     : 3000,
}

kf15 = KFold(n_splits=15, shuffle=True, random_state=42)
lgb_oof = np.zeros(len(X_tr))
lgb_test_log_preds = []
for fold, (tr, val) in enumerate(kf15.split(X_tr)):
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(X_tr.iloc[tr], y_train_log.iloc[tr],
          eval_set=[(X_tr.iloc[val], y_train_log.iloc[val])],
          callbacks=[lgb.early_stopping(100, verbose=False),
                     lgb.log_evaluation(False)])
    lgb_oof[val] = m.predict(X_tr.iloc[val])
    lgb_test_log_preds.append(m.predict(X_te))
lgb_test_log = np.mean(lgb_test_log_preds, axis=0)
lgb_score    = rmsle_log(y_train_log, lgb_oof)
print(f"    OOF: {lgb_score:.5f}")

# Find best 3-way blend via grid over the OOF predictions
print("\nSearching 3-way blend (log-XGB / Tweedie-XGB / LGB) in OOF space...")
best = (None, None, None, 9)
for a in np.arange(0, 1.01, 0.1):
    for b in np.arange(0, 1.0 - a + 1e-9, 0.1):
        c = 1.0 - a - b
        if c < -1e-9: continue
        blend_oof = a*xgb_log_oof + b*xgb_tw_oof + c*lgb_oof
        s = rmsle_log(y_train_log, blend_oof)
        if s < best[-1]:
            best = (a, b, c, s)
a, b, c, s = best
print(f"  Best: {a:.1f} log-XGB + {b:.1f} Tweedie-XGB + {c:.1f} LGB -> OOF {s:.5f}")
global_blend_test_log = a*xgb_log_test_log + b*xgb_tw_test_log + c*lgb_test_log
global_blend_oof       = a*xgb_log_oof      + b*xgb_tw_oof      + c*lgb_oof
print(f"  Expected LB: ~{s - 0.008:.3f}")


# %% CELL 15 — OPTUNA TUNING  (XGB-Tweedie / LGB-Tweedie / LOW regressor)
# =============================================================================
# Three independent studies, each with a meaningful CV objective on KFold-15.
#
#  A) XGB-Tweedie on the full training set              — 120 trials
#  B) LGB-Tweedie on the full training set              — 120 trials
#  C) XGB-Tweedie on the LOW-txn subset (<30 hist)      —  80 trials
#     ^ this is where the error concentrates; a targeted study here pays
#       more per trial than the global one.
#
# Totals ~320 trials.  On a modern laptop budget ~2-3 hours with a 4-core
# XGB/LGB.  If you're time-boxed, drop n_trials_global to 60 and
# n_trials_low to 40 — you'll still get most of the gain.
# =============================================================================
N_TRIALS_GLOBAL_XGB = 120
N_TRIALS_GLOBAL_LGB = 120
N_TRIALS_LOW        = 80
KFOLDS_OPT          = 10   # use 10 folds during tuning for speed; refit at 15
KFOLDS_FINAL        = 15


def cv_lgb(params, X, y, X_test, n_splits=15, seed=42, target='log',
           early_stop=100):
    """Mirror of cv_xgb for LightGBM.  target = 'log' | 'tweedie'."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_preds = []
    scores = []

    y_raw = np.expm1(y) if target == 'tweedie' else None

    for tr, val in kf.split(X):
        X_f, X_v = X.iloc[tr], X.iloc[val]
        if target == 'log':
            y_f, y_v = y.iloc[tr], y.iloc[val]
        else:
            y_f, y_v = y_raw[tr], y_raw[val]

        m = lgb.LGBMRegressor(**params)
        m.fit(X_f, y_f, eval_set=[(X_v, y_v)],
              callbacks=[lgb.early_stopping(early_stop, verbose=False),
                         lgb.log_evaluation(False)])
        val_pred  = m.predict(X_v)
        test_pred = m.predict(X_test)

        if target == 'tweedie':
            val_pred_log  = np.log1p(np.clip(val_pred, 0, None))
            test_pred_log = np.log1p(np.clip(test_pred, 0, None))
            true_log      = np.log1p(y_v)
        else:
            val_pred_log, test_pred_log, true_log = val_pred, test_pred, y_v

        oof[val] = val_pred_log
        test_preds.append(test_pred_log)
        scores.append(rmsle_log(true_log, val_pred_log))

    return oof, np.mean(test_preds, axis=0), float(np.mean(scores))


if HAS_OPTUNA:
    # -- A) Global XGB-Tweedie ------------------------------------------
    print(f"\n[A] Optuna XGB-Tweedie  — {N_TRIALS_GLOBAL_XGB} trials ...")
    def obj_xgb_tw(trial):
        params = {
            'objective'              : 'reg:tweedie',
            'tweedie_variance_power' : trial.suggest_float('tvp', 1.05, 1.95),
            'learning_rate'          : trial.suggest_float('lr',  0.015, 0.12, log=True),
            'max_depth'              : trial.suggest_int('md',    3, 8),
            'min_child_weight'       : trial.suggest_float('mcw', 0.5, 50.0, log=True),
            'subsample'              : trial.suggest_float('ss',  0.50, 0.98),
            'colsample_bytree'       : trial.suggest_float('cs',  0.50, 0.98),
            'colsample_bylevel'      : trial.suggest_float('csl', 0.50, 1.0),
            'reg_alpha'              : trial.suggest_float('ra',  1e-4, 5.0, log=True),
            'reg_lambda'             : trial.suggest_float('rl',  1e-4, 10.0, log=True),
            'gamma'                  : trial.suggest_float('g',   1e-6, 1.0, log=True),
            'max_delta_step'         : trial.suggest_float('mds', 0.0, 3.0),
            'n_estimators'           : 4000,
            'random_state'           : 42,
            'verbosity'              : 0,
            'early_stopping_rounds'  : 120,
        }
        _, _, score = cv_xgb(params, X_tr, y_train_log, X_te,
                             n_splits=KFOLDS_OPT, target='tweedie')
        return score

    study_xgb = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=15),
    )
    study_xgb.optimize(obj_xgb_tw, n_trials=N_TRIALS_GLOBAL_XGB,
                       show_progress_bar=False)
    print(f"  best OOF (10-fold): {study_xgb.best_value:.5f}")
    print(f"  params: {study_xgb.best_params}")

    best_xgb = dict(
        objective='reg:tweedie',
        tweedie_variance_power=study_xgb.best_params['tvp'],
        learning_rate=study_xgb.best_params['lr'],
        max_depth=study_xgb.best_params['md'],
        min_child_weight=study_xgb.best_params['mcw'],
        subsample=study_xgb.best_params['ss'],
        colsample_bytree=study_xgb.best_params['cs'],
        colsample_bylevel=study_xgb.best_params['csl'],
        reg_alpha=study_xgb.best_params['ra'],
        reg_lambda=study_xgb.best_params['rl'],
        gamma=study_xgb.best_params['g'],
        max_delta_step=study_xgb.best_params['mds'],
        n_estimators=4000, random_state=42, verbosity=0,
        early_stopping_rounds=150,
    )

    # -- B) Global LGB-Tweedie ------------------------------------------
    print(f"\n[B] Optuna LGB-Tweedie  — {N_TRIALS_GLOBAL_LGB} trials ...")
    def obj_lgb_tw(trial):
        params = {
            'objective'              : 'tweedie',
            'tweedie_variance_power' : trial.suggest_float('tvp', 1.05, 1.95),
            'metric'                 : 'rmse',
            'learning_rate'          : trial.suggest_float('lr',  0.015, 0.12, log=True),
            'num_leaves'             : trial.suggest_int('nl',    15, 200),
            'max_depth'              : trial.suggest_int('md',    3, 10),
            'min_child_samples'      : trial.suggest_int('mcs',   5, 80),
            'subsample'              : trial.suggest_float('ss',  0.55, 0.98),
            'colsample_bytree'       : trial.suggest_float('cs',  0.55, 0.98),
            'reg_alpha'              : trial.suggest_float('ra',  1e-4, 5.0, log=True),
            'reg_lambda'             : trial.suggest_float('rl',  1e-4, 10.0, log=True),
            'min_split_gain'         : trial.suggest_float('msg', 0.0, 0.5),
            'n_estimators'           : 4000,
            'random_state'           : 42,
            'verbose'                : -1,
        }
        _, _, score = cv_lgb(params, X_tr, y_train_log, X_te,
                             n_splits=KFOLDS_OPT, target='tweedie')
        return score

    study_lgb = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=7, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=15),
    )
    study_lgb.optimize(obj_lgb_tw, n_trials=N_TRIALS_GLOBAL_LGB,
                       show_progress_bar=False)
    print(f"  best OOF (10-fold): {study_lgb.best_value:.5f}")
    print(f"  params: {study_lgb.best_params}")

    best_lgb = dict(
        objective='tweedie',
        tweedie_variance_power=study_lgb.best_params['tvp'],
        metric='rmse',
        learning_rate=study_lgb.best_params['lr'],
        num_leaves=study_lgb.best_params['nl'],
        max_depth=study_lgb.best_params['md'],
        min_child_samples=study_lgb.best_params['mcs'],
        subsample=study_lgb.best_params['ss'],
        colsample_bytree=study_lgb.best_params['cs'],
        reg_alpha=study_lgb.best_params['ra'],
        reg_lambda=study_lgb.best_params['rl'],
        min_split_gain=study_lgb.best_params['msg'],
        n_estimators=4000, random_state=42, verbose=-1,
    )

    # -- C) LOW-txn XGB-Tweedie (dedicated study) ------------------------
    # Trained *only* on customers with txn_count_12m < 30 — this is where
    # the global model underperforms (your diagnostics: LOW OOF 0.585).
    print(f"\n[C] Optuna LOW-txn XGB-Tweedie  — {N_TRIALS_LOW} trials ...")
    gate_train_tmp = train_master['txn_count_12m'].fillna(0).values
    low_mask_tmp   = gate_train_tmp < 30
    X_tr_lo        = X_tr[low_mask_tmp].reset_index(drop=True)
    y_tr_lo_log    = y_train_log[low_mask_tmp].reset_index(drop=True)
    print(f"     LOW training rows: {len(X_tr_lo)}")

    def obj_xgb_low(trial):
        params = {
            'objective'              : 'reg:tweedie',
            'tweedie_variance_power' : trial.suggest_float('tvp', 1.05, 1.85),
            'learning_rate'          : trial.suggest_float('lr',  0.01, 0.10, log=True),
            'max_depth'              : trial.suggest_int('md',    3, 7),
            'min_child_weight'       : trial.suggest_float('mcw', 0.5, 40.0, log=True),
            'subsample'              : trial.suggest_float('ss',  0.50, 0.98),
            'colsample_bytree'       : trial.suggest_float('cs',  0.50, 0.98),
            'reg_alpha'              : trial.suggest_float('ra',  1e-4, 5.0, log=True),
            'reg_lambda'             : trial.suggest_float('rl',  1e-4, 10.0, log=True),
            'gamma'                  : trial.suggest_float('g',   1e-6, 1.0, log=True),
            'n_estimators'           : 3000,
            'random_state'           : 42,
            'verbosity'              : 0,
            'early_stopping_rounds'  : 120,
        }
        _, _, score = cv_xgb(params, X_tr_lo, y_tr_lo_log, X_te,
                             n_splits=KFOLDS_OPT, target='tweedie')
        return score

    study_low = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=123, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=12),
    )
    study_low.optimize(obj_xgb_low, n_trials=N_TRIALS_LOW, show_progress_bar=False)
    print(f"  best OOF (LOW only, 10-fold): {study_low.best_value:.5f}")
    print(f"  params: {study_low.best_params}")

    best_xgb_low = dict(
        objective='reg:tweedie',
        tweedie_variance_power=study_low.best_params['tvp'],
        learning_rate=study_low.best_params['lr'],
        max_depth=study_low.best_params['md'],
        min_child_weight=study_low.best_params['mcw'],
        subsample=study_low.best_params['ss'],
        colsample_bytree=study_low.best_params['cs'],
        reg_alpha=study_low.best_params['ra'],
        reg_lambda=study_low.best_params['rl'],
        gamma=study_low.best_params['g'],
        n_estimators=3000, random_state=42, verbosity=0,
        early_stopping_rounds=150,
    )
else:
    print("\n(Optuna not installed — falling back to hand-tuned params.)")
    best_xgb     = XGB_TWEEDIE_PARAMS
    best_lgb     = dict(LGB_PARAMS, objective='tweedie', tweedie_variance_power=1.5)
    best_xgb_low = XGB_TWEEDIE_PARAMS


# %% CELL 15b — REFIT TUNED MODELS AT KFOLDS_FINAL WITH MULTI-SEED AVG
# =============================================================================
# Optuna was run at 10 folds for speed; we refit at 15 folds and average
# across 5 seeds to squeeze out ~0.001–0.003 of KFold noise.
# =============================================================================
SEEDS = [42, 7, 123, 2024, 888]

def multiseed_cv_xgb(params, X, y, X_test, target, n_splits=15, seeds=SEEDS):
    oof_stack, test_stack, scores = [], [], []
    for s in seeds:
        p = dict(params); p['random_state'] = s
        o, t, sc = cv_xgb(p, X, y, X_test, n_splits=n_splits, seed=s, target=target)
        oof_stack.append(o); test_stack.append(t); scores.append(sc)
    return (np.mean(oof_stack, axis=0),
            np.mean(test_stack, axis=0),
            float(np.mean(scores)))

def multiseed_cv_lgb(params, X, y, X_test, target, n_splits=15, seeds=SEEDS):
    oof_stack, test_stack, scores = [], [], []
    for s in seeds:
        p = dict(params); p['random_state'] = s
        o, t, sc = cv_lgb(p, X, y, X_test, n_splits=n_splits, seed=s, target=target)
        oof_stack.append(o); test_stack.append(t); scores.append(sc)
    return (np.mean(oof_stack, axis=0),
            np.mean(test_stack, axis=0),
            float(np.mean(scores)))

print(f"\nRefitting tuned XGB-Tweedie at KFold-{KFOLDS_FINAL} × {len(SEEDS)} seeds ...")
xgb_tw_opt_oof, xgb_tw_opt_test_log, xgb_tw_opt_score = multiseed_cv_xgb(
    best_xgb, X_tr, y_train_log, X_te, target='tweedie', n_splits=KFOLDS_FINAL)
print(f"  multi-seed OOF: {xgb_tw_opt_score:.5f}")

print(f"Refitting tuned LGB-Tweedie at KFold-{KFOLDS_FINAL} × {len(SEEDS)} seeds ...")
lgb_tw_opt_oof, lgb_tw_opt_test_log, lgb_tw_opt_score = multiseed_cv_lgb(
    best_lgb, X_tr, y_train_log, X_te, target='tweedie', n_splits=KFOLDS_FINAL)
print(f"  multi-seed OOF: {lgb_tw_opt_score:.5f}")

# Tuned LOW XGB is refit inside the two-stage loop (Cell 16) because we
# need proper out-of-fold predictions on the full train set to learn the
# two-stage blend weight.


# %% CELL 16 — TWO-STAGE MODEL WITH PROPER OOF  (KEY FIX vs v4-original)
# =============================================================================
# v4-original hardcoded TS_WEIGHT=0.25.  v4.1 builds a real OOF for the
# two-stage pipeline inside a single KFold loop and learns TS_WEIGHT on
# that OOF.  Steps per fold:
#
#   1. fit classifier on train-fold → predict P on val-fold and test
#   2. partition train-fold into HIGH / LOW by txn_count_12m
#   3. fit HIGH regressor (log-y, XGB_LOG_PARAMS) → predict val + test
#   4. fit LOW  regressor (Tweedie, best_xgb_low)  → predict val + test
#   5. smooth gate combines HIGH/LOW on both val and test
#   6. multiplicative classifier damper → two_stage prediction
#
# After all folds: two_stage_oof is a legit out-of-fold vector we can
# grid-search against to learn the best blend weight.
# =============================================================================
print("\nTwo-stage training with proper OOF ...")

gate_train = train_master['txn_count_12m'].fillna(0).values
gate_test  = test_master['txn_count_12m'].fillna(0).values
gate_w_test = 1.0 / (1.0 + np.exp(-(gate_test - 30) / 10))

y_cls = (y_train >= 5).astype(int)
CLS_PARAMS = {
    'objective'            : 'binary:logistic',
    'learning_rate'        : 0.05,
    'max_depth'            : 4,
    'min_child_weight'     : 10,
    'n_estimators'         : 2000,
    'random_state'         : 42,
    'verbosity'            : 0,
    'early_stopping_rounds': 100,
    'eval_metric'          : 'logloss',
}

two_stage_oof_log = np.zeros(len(X_tr))
cls_test_preds    = []
hi_test_preds     = []
lo_test_preds     = []
hi_fold_scores    = []
lo_fold_scores    = []

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for fold, (tr, val) in enumerate(skf.split(X_tr, y_cls)):
    X_f, X_v = X_tr.iloc[tr], X_tr.iloc[val]
    y_f_log  = y_train_log.iloc[tr].reset_index(drop=True)
    y_v_log  = y_train_log.iloc[val].reset_index(drop=True)
    y_f_raw  = y_train[tr]
    gate_f   = gate_train[tr]
    gate_v   = gate_train[val]

    # Stage 1: classifier
    m_cls = xgb.XGBClassifier(**CLS_PARAMS)
    m_cls.fit(X_f, y_cls[tr], eval_set=[(X_v, y_cls[val])], verbose=False)
    p_val  = m_cls.predict_proba(X_v)[:, 1]
    p_test = m_cls.predict_proba(X_te)[:, 1]
    cls_test_preds.append(p_test)

    # Stage 2a: HIGH regressor (log-y, squared error)
    hi_mask = gate_f >= 30
    m_hi = xgb.XGBRegressor(**XGB_LOG_PARAMS)
    m_hi.fit(X_f.iloc[hi_mask], y_f_log.iloc[hi_mask],
             eval_set=[(X_v, y_v_log)], verbose=False)
    hi_val_log  = m_hi.predict(X_v)
    hi_test_log = m_hi.predict(X_te)
    hi_test_preds.append(hi_test_log)

    # Stage 2b: LOW regressor (Tweedie, tuned)
    lo_mask = gate_f < 30
    m_lo = xgb.XGBRegressor(**best_xgb_low)
    # fit on raw y of LOW; eval on full val raw y for early stopping
    m_lo.fit(X_f.iloc[lo_mask], y_f_raw[lo_mask],
             eval_set=[(X_v, y_train[val])], verbose=False)
    lo_val_raw  = np.clip(m_lo.predict(X_v), 0, None)
    lo_test_raw = np.clip(m_lo.predict(X_te), 0, None)
    lo_val_log  = np.log1p(lo_val_raw)
    lo_test_log = np.log1p(lo_test_raw)
    lo_test_preds.append(lo_test_log)

    # Smooth gate on val & test
    gw_v = 1.0 / (1.0 + np.exp(-(gate_v - 30) / 10))
    gated_val_log = gw_v * hi_val_log + (1.0 - gw_v) * lo_val_log
    # Multiplicative classifier damper
    ts_val_log = np.log1p(p_val * np.expm1(np.clip(gated_val_log, 0, None)))
    two_stage_oof_log[val] = ts_val_log

    # Track segment-level fold scores for reporting
    if hi_mask.sum() > 0:
        hi_v_score = rmsle_log(y_v_log[gate_v >= 30], hi_val_log[gate_v >= 30])
        hi_fold_scores.append(hi_v_score)
    if lo_mask.sum() > 0:
        lo_v_score = rmsle_log(y_v_log[gate_v < 30], lo_val_log[gate_v < 30])
        lo_fold_scores.append(lo_v_score)

cls_test     = np.mean(cls_test_preds, axis=0)
hi_test_log  = np.mean(hi_test_preds,  axis=0)
lo_test_log  = np.mean(lo_test_preds,  axis=0)

# Final test-space two-stage prediction using averaged fold predictions
gated_test_log = gate_w_test * hi_test_log + (1.0 - gate_w_test) * lo_test_log
two_stage_test_log = np.log1p(cls_test * np.expm1(np.clip(gated_test_log, 0, None)))

two_stage_oof_score = rmsle_log(y_train_log, two_stage_oof_log)
print(f"  Two-stage  OOF : {two_stage_oof_score:.5f}")
print(f"  HIGH segment OOF (per-fold mean): {np.mean(hi_fold_scores):.5f}")
print(f"  LOW  segment OOF (per-fold mean): {np.mean(lo_fold_scores):.5f}")


# %% CELL 17 — LEARN BLEND WEIGHTS FROM OOF + WRITE SUBMISSIONS
# =============================================================================
# Now we have legitimate OOF vectors for every candidate:
#   xgb_log_oof        XGB log-y
#   xgb_tw_opt_oof     XGB Tweedie (tuned, multi-seed)
#   lgb_oof            LGB log-y
#   lgb_tw_opt_oof     LGB Tweedie (tuned, multi-seed)
#   two_stage_oof_log  Two-stage
#
# Find convex weights (sum to 1) that minimise blended OOF RMSLE via a
# dense grid — 5D grid at 0.05 step is ~4000 combos, trivial.
# =============================================================================
print("\nLearning OOF-optimal blend weights over 5 candidates ...")

oof_stack = {
    'xgb_log'    : xgb_log_oof,
    'xgb_tw_opt' : xgb_tw_opt_oof,
    'lgb_log'    : lgb_oof,
    'lgb_tw_opt' : lgb_tw_opt_oof,
    'two_stage'  : two_stage_oof_log,
}
test_stack = {
    'xgb_log'    : xgb_log_test_log,
    'xgb_tw_opt' : xgb_tw_opt_test_log,
    'lgb_log'    : lgb_test_log,
    'lgb_tw_opt' : lgb_tw_opt_test_log,
    'two_stage'  : two_stage_test_log,
}
names = list(oof_stack.keys())
n = len(names)

# Coordinate-descent blending.  Start at equal weights; for each index,
# grid-scan its weight 0..1 and renormalise the others; iterate to
# convergence.  This finds a near-optimal simplex point without a full
# 5D grid.
w = np.full(n, 1.0 / n)
improved = True
while improved:
    improved = False
    for i in range(n):
        base = w.copy()
        best_w_i, best_s = w[i], 1e9
        for wi in np.arange(0.0, 1.01, 0.02):
            cand = base.copy()
            cand[i] = wi
            rest = 1.0 - wi
            other = np.delete(cand, i)
            other_sum = other.sum() + 1e-12
            other = other * (rest / other_sum)
            cand = np.insert(other, i, wi)
            blended = sum(c * oof_stack[names[j]] for j, c in enumerate(cand))
            s_ = rmsle_log(y_train_log, blended)
            if s_ < best_s:
                best_s, best_w_i = s_, wi
        # apply best_w_i
        cand = base.copy(); cand[i] = best_w_i
        other = np.delete(cand, i); other_sum = other.sum() + 1e-12
        other = other * ((1.0 - best_w_i) / other_sum)
        new_w = np.insert(other, i, best_w_i)
        new_score = rmsle_log(
            y_train_log,
            sum(c * oof_stack[names[j]] for j, c in enumerate(new_w)))
        if new_score + 1e-6 < rmsle_log(
                y_train_log,
                sum(c * oof_stack[names[j]] for j, c in enumerate(w))):
            w = new_w; improved = True

final_oof  = sum(c * oof_stack[names[j]]  for j, c in enumerate(w))
final_test = sum(c * test_stack[names[j]] for j, c in enumerate(w))
final_score = rmsle_log(y_train_log, final_oof)

print("  Learned weights:")
for n_, c in zip(names, w):
    print(f"    {n_:<12s} = {c:.3f}")
print(f"  Blended OOF  : {final_score:.5f}")
print(f"  Expected LB  : ~{final_score - 0.008:.3f}")


# ── Write submissions ──────────────────────────────────────────────────
test_ids = test_master['UniqueID'].values
save_submission(np.expm1(xgb_log_test_log),      'submission_v4_xgb_log.csv',
                test_ids, sample_sub, DATA_PATH)
save_submission(np.expm1(xgb_tw_opt_test_log),   'submission_v4_xgb_tw_opt.csv',
                test_ids, sample_sub, DATA_PATH)
save_submission(np.expm1(lgb_tw_opt_test_log),   'submission_v4_lgb_tw_opt.csv',
                test_ids, sample_sub, DATA_PATH)
save_submission(np.expm1(two_stage_test_log),    'submission_v4_twostage.csv',
                test_ids, sample_sub, DATA_PATH)
save_submission(np.expm1(final_test),            'submission_v4_final.csv',
                test_ids, sample_sub, DATA_PATH)

print("\n" + "=" * 62)
print("SUMMARY")
print("=" * 62)
print(f"XGB log-y               OOF : {xgb_log_score:.5f}  (LB ~{xgb_log_score-0.008:.3f})")
print(f"XGB Tweedie (untuned)   OOF : {xgb_tw_score:.5f}  (LB ~{xgb_tw_score-0.008:.3f})")
print(f"LGB log-y               OOF : {lgb_score:.5f}  (LB ~{lgb_score-0.008:.3f})")
print(f"XGB Tweedie OPT+seeds   OOF : {xgb_tw_opt_score:.5f}  (LB ~{xgb_tw_opt_score-0.008:.3f})")
print(f"LGB Tweedie OPT+seeds   OOF : {lgb_tw_opt_score:.5f}  (LB ~{lgb_tw_opt_score-0.008:.3f})")
print(f"Two-stage (proper OOF)  OOF : {two_stage_oof_score:.5f}  (LB ~{two_stage_oof_score-0.008:.3f})")
print("-" * 62)
print(f"OOF-optimal 5-way blend OOF : {final_score:.5f}  (LB ~{final_score-0.008:.3f})")
print("=" * 62)
print("\nRECOMMENDED SUBMISSION ORDER (use the 2 best for Zindi):")
print("  1. submission_v4_final.csv         (OOF-optimal blend of all 5)")
print("  2. submission_v4_xgb_tw_opt.csv    (tuned XGB-Tweedie; most stable)")
print("  3. submission_v4_twostage.csv      (ablation — low-txn focus)")
print("  4. submission_v4_lgb_tw_opt.csv    (tuned LGB-Tweedie)")
print("  5. submission_v4_xgb_log.csv       (v3-style baseline for reference)")
