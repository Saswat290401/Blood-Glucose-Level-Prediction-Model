
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime


train=pd.read_csv('/kaggle/input/brist1d/train.csv')
test=pd.read_csv('/kaggle/input/brist1d/test.csv')
train.shape

train.head()

train.columns

train.info()

train.describe()

train.isnull().sum()

train_missing = train.isnull().mean()*100

columns_to_drop = train_missing[train_missing > 25].index

columns_to_drop.shape

train_cleaned = train.drop(columns=columns_to_drop)
test_cleaned = test.drop(columns=columns_to_drop)

train_cleaned.shape,test_cleaned.shape

print(f"Columns dropped: {columns_to_drop}")
print("Updated df_train:",train_cleaned.shape)
print("Updated df_test:",test_cleaned.shape)

train_cleaned = train_cleaned.drop('id', axis=1)
test_cleaned = test_cleaned.drop('id', axis=1)

def convert_time_columns(train_cleaned):
    for col in train_cleaned.columns:
        if pd.api.types.is_datetime64_any_dtype(train_cleaned[col]) or 'time' in col.lower():
            train_cleaned[col] = pd.to_datetime(train_cleaned[col], errors='coerce')
            train_cleaned[col+'_hour'] = train_cleaned[col].dt.hour
            train_cleaned[col+'_minute'] = train_cleaned[col].dt.minute
            train_cleaned[col+'_second'] = train_cleaned[col].dt.second
            train_cleaned.drop(col, axis=1, inplace=True)
            
convert_time_columns(train_cleaned)
convert_time_columns(test_cleaned)

unique_pnums = test_cleaned['p_num'].unique()
pnum_mapping = {value: idx for idx, value in enumerate(unique_pnums)}
test_cleaned['p_num'] = test_cleaned['p_num'].map(pnum_mapping)

if 'p_num' in train_cleaned.columns:
    train_cleaned['p_num'] = train_cleaned['p_num'].map(pnum_mapping)

print(train_cleaned.shape)
print(test_cleaned.shape)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

train_imputed = imputer.fit_transform(train_cleaned)
test_imputed = imputer.fit_transform(test_cleaned)

train_cleaned = pd.DataFrame(train_imputed, columns=train_cleaned.columns)
test_cleaned = pd.DataFrame(test_imputed, columns=test_cleaned.columns)


train_float_columns = train_cleaned.select_dtypes(include=['float64', 'float32']).columns.tolist()
test_float_columns = test_cleaned.select_dtypes(include=['float64', 'float32']).columns.tolist()

train_cleaned[train_float_columns] = train_cleaned[train_float_columns].astype(float)
test_cleaned[test_float_columns] = test_cleaned[test_float_columns].astype(float)

print(train_cleaned.dtypes)
print(train_cleaned.columns)

train_cleaned.isnull().sum()

test_cleaned.isnull().sum()

x_train = train_cleaned.drop('bg+1:00', axis=1)
y_train = train_cleaned['bg+1:00']

x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train, test_size = 0.2 , random_state = 42)

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.inspection import permutation_importance

model = HistGradientBoostingRegressor()
model.fit(x_train_split, y_train_split)

result = permutation_importance(model, x_val, y_val, n_repeats=30, random_state=42)

feature_importances = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': result.importances_mean
})

threshold = 0.000509

selected_features = feature_importances[feature_importances['Importance'] > threshold]

selected_features = selected_features.sort_values(by='Importance', ascending=False)

print("Selected Features from Permutation Importance:")
print(selected_features)
print(feature_importances.describe())

selected_indices = [x_train.columns.tolist().index(f) for f in selected_features['Feature']]
x_train_split = x_train_split.iloc[:, selected_indices]
x_val = x_val.iloc[:, selected_indices]

scaler = StandardScaler()
x_train_split = scaler.fit_transform(x_train_split)
x_val = scaler.transform(x_val)

model = HistGradientBoostingRegressor(
    max_iter=210, 
    learning_rate=0.15,
    scoring = "neg_root_mean_squared_error",
    random_state=42
)

model.fit(x_train_split, y_train_split)

y_pred = model.predict(x_val)

mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)

r2 = r2_score(y_val, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error: {mae:.5f}')
print(f'Mean Squared Error: {mse:.5f}')
print(f"R2 Score: {r2:.5f}")
print(f"RMSE: {rmse:.5f}")


plt.figure(figsize=(12, 6))
plt.plot(y_val.values, label='Actual Glucose Levels')
plt.plot(y_pred, label='Predicted Glucose Levels')
plt.xlabel('Samples')
plt.ylabel('Glucose Level')
plt.title('Predicted vs Actual Glucose Levels')
plt.legend()
plt.show()

x_test = test_cleaned.drop('bg+1:00', axis=1, errors='ignore')
x_test = x_test.iloc[:, selected_indices]
x_test = scaler.transform(x_test)
test_predictions = model.predict(x_test)


submission = pd.DataFrame({'id': test['id'], 'bg+1:00': test_predictions})
submission.to_csv(f'/kaggle/working/submission.csv', index=False)

print("Model training, evaluation, and test predictions completed.")