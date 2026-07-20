import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

PATH = 'data/processed/'
df = pd.read_csv(PATH + 'modeling_stage2.csv')
with open(PATH + 'feature_cols_s2.pkl', 'rb') as f:
    feature_cols_s2 = pickle.load(f)

# Apply same filtering as in notebook
feature_cols_s2 = [
    c for c in feature_cols_s2
    if c in df.columns
    and df[c].dtype not in ["object"]
    and c != "is_fraud"
]

TARGET = "fraud_label"
X = df[feature_cols_s2]
y = df[TARGET]

# Check dtypes in X
print('Dtypes in X:')
print(X.dtypes)

# Check for any non-numeric values
print('\n\nChecking for non-numeric values:')
for col in X.columns:
    if X[col].dtype != 'object':
        # Try to detect any problematic values
        try:
            X[col].astype(float)
        except ValueError as e:
            print(f'{col}: {e}')
            print(f'  Sample values: {X[col].unique()[:5]}')

# Check for NaN
print(f'\nNaN values:\n{X.isna().sum()}')

# Do the train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'\nX_train dtypes:')
print(X_train.dtypes)

# Check for any issues in X_train
for col in X_train.columns:
    if X_train[col].dtype != 'object':
        try:
            X_train[col].astype(float)
        except ValueError as e:
            print(f'X_train {col}: {e}')
            print(f'  Sample values: {X_train[col].unique()[:5]}')
