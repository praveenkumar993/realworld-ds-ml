import pandas as pd
import pickle
import shap
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
PATH = 'ride_sharing/cancellation_prediction/data/processed/'
modeling_df = pd.read_csv(PATH + 'modeling_data.csv')
with open(PATH + 'feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)
X = modeling_df[feature_cols]
y = modeling_df['ride_outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
best_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=10, n_jobs=-1, random_state=42)
best_model.fit(X_train_smote, y_train_smote)
X_sample = X_test.sample(1500, random_state=42)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_sample)
print('len', len(shap_values))
for i, sv in enumerate(shap_values):
    print('class', i, 'shape', getattr(sv, 'shape', None), 'ndim', getattr(sv, 'ndim', None))
print('X_sample shape', X_sample.shape)
print('feature cols', X_sample.shape[1])
print('sv1 row len', len(shap_values[1][0]))
print('sv1 shape[1]==cols', shap_values[1].shape[1] == X_sample.shape[1])
