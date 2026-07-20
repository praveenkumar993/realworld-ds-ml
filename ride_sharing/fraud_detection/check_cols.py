import pandas as pd
import pickle

PATH = 'data/processed/'
df = pd.read_csv(PATH + 'modeling_stage2.csv')
with open(PATH + 'feature_cols_s2.pkl', 'rb') as f:
    cols = pickle.load(f)

# Check for object columns
obj_cols = [c for c in cols if c in df.columns and df[c].dtype == 'object']
print('Object columns in feature_cols_s2:')
print(obj_cols)
print('\nTotal object columns:', len(obj_cols))

# Show sample values
if obj_cols:
    print('\nSample values:')
    for col in obj_cols[:3]:
        print(f'{col}: {df[col].unique()[:5]}')
