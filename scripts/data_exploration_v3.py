import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np

# Load the data
data_path = 'C:/Users/User/PycharmProjects/M5 Forecasting - Accuracy/data/'
sales_train_validation = pd.read_csv(data_path + 'sales_train_validation.csv')
calendar = pd.read_csv(data_path + 'calendar.csv')
sell_prices = pd.read_csv(data_path + 'sell_prices.csv')

# Find the maximum day in the training data
max_day = sales_train_validation.columns[-1]
max_day_num = int(max_day.split('_')[1])

# Create columns for the next 28 days
future_days = [f'd_{i}' for i in range(max_day_num + 1, max_day_num + 29)]
for day in future_days:
    sales_train_validation[day] = None

# Capture the original order of IDs
original_order = sales_train_validation['id'].tolist()

# Melt the extended sales data to long format
sales_train_validation = sales_train_validation.melt(
    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    var_name='day',
    value_name='sales'
)

# Merge with the calendar data
extended_data = sales_train_validation.merge(calendar, left_on='day', right_on='d', how='left')

# Merge with the sell prices data
extended_data = extended_data.merge(sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

# Convert date column to datetime
extended_data['date'] = pd.to_datetime(extended_data['date'])

# Date features
extended_data['day'] = extended_data['date'].dt.day
extended_data['weekday'] = extended_data['date'].dt.weekday
extended_data['month'] = extended_data['date'].dt.month
extended_data['year'] = extended_data['date'].dt.year

# Rolling window features
extended_data['rolling_mean_7'] = extended_data.groupby(['id'])['sales'].shift(1).rolling(window=7).mean()
extended_data['rolling_std_7'] = extended_data.groupby(['id'])['sales'].shift(1).rolling(window=7).std()
extended_data['rolling_mean_30'] = extended_data.groupby(['id'])['sales'].shift(1).rolling(window=30).mean()
extended_data['rolling_std_30'] = extended_data.groupby(['id'])['sales'].shift(1).rolling(window=30).std()

# Handle NaN values in sell_price column by filling with the mean price of that item
extended_data['sell_price'] = extended_data.groupby('item_id')['sell_price'].transform(lambda x: x.fillna(x.mean()))

# Verify calendar dates and cutoff date
cutoff_date_str = calendar[calendar['d'] == max_day]['date'].values[0]
print(f"Cutoff date from calendar: {cutoff_date_str}")
cutoff_date = pd.to_datetime(cutoff_date_str)
print(f"Parsed cutoff date: {cutoff_date}")

# Filter the training data up to the date corresponding to the cutoff date
train_data = extended_data[extended_data['date'] <= cutoff_date]
print(f"Training data shape: {train_data.shape}")

# Selected features
selected_features = [
    'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30',
    'sell_price', 'wm_yr_wk', 'weekday', 'day', 'month', 'year'
]

# Prepare the training data
X_train = train_data[selected_features]
y_train = train_data['sales']

# Optimize memory usage for training data
def optimize_memory(df):
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == "int64":
            df[col] = df[col].astype(np.int32)
    return df

X_train = optimize_memory(X_train)

# Train the model
model = HistGradientBoostingRegressor()
model.fit(X_train, y_train)

# Prepare the test set for forecasting (next 28 days)
forecast_start_date_str = calendar[calendar['d'] == future_days[0]]['date'].values[0]
print(f"Forecast start date from calendar: {forecast_start_date_str}")
forecast_start_date = pd.to_datetime(forecast_start_date_str)
forecast_end_date = forecast_start_date + pd.Timedelta(days=27)
print(f"Forecast period: {forecast_start_date} to {forecast_end_date}")

# Filter the extended data for the forecast period
test_data = extended_data[(extended_data['date'] >= forecast_start_date) & (extended_data['date'] <= forecast_end_date)]

# Check if test_data is empty
if test_data.empty:
    raise ValueError("The test_data DataFrame is empty. Please check the date ranges and data merging steps.")

# Select the relevant columns for the test data
X_test = test_data[selected_features]

# Check if X_test is empty
if X_test.empty:
    raise ValueError("The X_test DataFrame is empty. Please check the selected features and data preprocessing steps.")

# Optimize memory usage for test data
X_test = optimize_memory(X_test)

# Forecast for the next 28 days
test_data['sales'] = model.predict(X_test)

# Ensure no negative forecasts
test_data['sales'] = test_data['sales'].apply(lambda x: max(x, 0))

# Prepare the submission file
submission = test_data.pivot(index='id', columns='day', values='sales').reset_index()

# Adjust column names for the submission format
submission.columns = ['id'] + [f'F{i+1}' for i in range(28)]

# Reorder the submission to match the original order of IDs
submission = submission.set_index('id').loc[original_order].reset_index()

# Save the submission file
submission.to_csv('submission.csv', index=False)

print(submission.head())
