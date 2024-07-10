import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# Load the data
sales_train_validation = pd.read_csv('C:/Users/User/PycharmProjects/M5 Forecasting - Accuracy/data/sales_train_validation.csv')
calendar = pd.read_csv('C:/Users/User/PycharmProjects/M5 Forecasting - Accuracy/data/calendar.csv')
sell_prices = pd.read_csv('C:/Users/User/PycharmProjects/M5 Forecasting - Accuracy/data/sell_prices.csv')

# Melt the sales data to long format
sales_train_validation = sales_train_validation.melt(
    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    var_name='day',
    value_name='sales'
)

# Merge with the calendar data
sales_train_validation = sales_train_validation.merge(calendar, left_on='day', right_on='d')

# Merge with the sell prices data
sales_train_validation = sales_train_validation.merge(sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

# Date features
sales_train_validation['day'] = sales_train_validation['date'].apply(lambda x: x[-2:]).astype(int)
sales_train_validation['weekday'] = pd.to_datetime(sales_train_validation['date']).dt.weekday
sales_train_validation['month'] = pd.to_datetime(sales_train_validation['date']).dt.month
sales_train_validation['year'] = pd.to_datetime(sales_train_validation['date']).dt.year

# Rolling window features
sales_train_validation['rolling_mean_7'] = sales_train_validation.groupby(['id'])['sales'].shift(1).rolling(window=7).mean()
sales_train_validation['rolling_std_7'] = sales_train_validation.groupby(['id'])['sales'].shift(1).rolling(window=7).std()
sales_train_validation['rolling_mean_30'] = sales_train_validation.groupby(['id'])['sales'].shift(1).rolling(window=30).mean()
sales_train_validation['rolling_std_30'] = sales_train_validation.groupby(['id'])['sales'].shift(1).rolling(window=30).std()

# Handle NaN values in event columns by filling with 'None'
sales_train_validation['event_name_1'] = sales_train_validation['event_name_1'].fillna('None')
sales_train_validation['event_type_1'] = sales_train_validation['event_type_1'].fillna('None')
sales_train_validation['event_name_2'] = sales_train_validation['event_name_2'].fillna('None')
sales_train_validation['event_type_2'] = sales_train_validation['event_type_2'].fillna('None')

# Handle NaN values in sell_price column by filling with the mean price of that item
sales_train_validation['sell_price'] = sales_train_validation.groupby('item_id')['sell_price'].transform(lambda x: x.fillna(x.mean()))

# Selective One Hot Encoding for 'event_type_1' only
event_type_1_encoded = pd.get_dummies(sales_train_validation['event_type_1'], prefix='event_type_1')
sales_train_validation = pd.concat([sales_train_validation, event_type_1_encoded], axis=1)
sales_train_validation.drop(['event_type_1'], axis=1, inplace=True)

# Selected features
selected_features = [
    'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30',
    'sell_price', 'wm_yr_wk', 'wday'
]

# Add One Hot Encoded columns for event_type_1
selected_features.extend(event_type_1_encoded.columns)

# Define model training function
def process_department(train_data, eval_data, dept_id):
    # Select relevant features
    X_train = train_data[selected_features]
    y_train = train_data['sales']

    X_eval = eval_data[selected_features]
    y_eval = eval_data['sales']

    # Model Training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = model.predict(X_eval)

    mse = mean_squared_error(y_eval, y_pred)
    mae = mean_absolute_error(y_eval, y_pred)

    print(f'Department: {dept_id}')
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')

    # Feature Importance
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='bar')
    plt.title(f'Feature Importance for {dept_id}')
    plt.show()

    return model, y_eval, y_pred

# Assuming you have a separate evaluation dataset
sales_train_evaluation = pd.read_csv('C:/Users/User/PycharmProjects/M5 Forecasting - Accuracy/data/sales_train_evaluation.csv')

# Selective One Hot Encoding for 'event_type_1' in eval_data
event_type_1_encoded_eval = pd.get_dummies(sales_train_evaluation['event_type_1'], prefix='event_type_1')
sales_train_evaluation = pd.concat([sales_train_evaluation, event_type_1_encoded_eval], axis=1)
sales_train_evaluation.drop(['event_type_1'], axis=1, inplace=True)

# List of department IDs
dept_ids = sales_train_validation['dept_id'].unique()

# Dictionary to store models and results
models = {}
results = []

for dept_id in dept_ids:
    train_data = sales_train_validation[sales_train_validation['dept_id'] == dept_id]
    eval_data = sales_train_evaluation[sales_train_evaluation['dept_id'] == dept_id]

    model, y_eval, y_pred = process_department(train_data, eval_data, dept_id)
    models[dept_id] = model
    results.append((dept_id, y_eval, y_pred))
