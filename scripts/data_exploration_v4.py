import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Function to optimize memory usage
def optimize_memory(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "float64":
            df[col] = df[col].astype(np.float32)
        elif col_type == "int64":
            df[col] = df[col].astype(np.int32)
        elif col_type == "object":
            df[col] = df[col].astype('category')
    return df

# Load and optimize the calendar and sell_prices data
def load_and_optimize(path):
    df = pd.read_csv(path)
    return optimize_memory(df)

data_path = 'C:/Users/User/PycharmProjects/M5 Forecasting - Accuracy/data/'
calendar = load_and_optimize(data_path + 'calendar.csv')
sell_prices = load_and_optimize(data_path + 'sell_prices.csv')

# Load the extended data
extended_data = pd.read_csv(data_path + 'sales_train_validation.csv')

# Find the maximum day in the training data
max_day = extended_data.columns[-1]
max_day_num = int(max_day.split('_')[1])

# Create columns for the next 28 days
future_days = [f'd_{i}' for i in range(max_day_num + 1, max_day_num + 29)]
for day in future_days:
    extended_data[day] = None

# Melt the extended sales data to long format
extended_data = extended_data.melt(
    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    var_name='day',
    value_name='sales'
)

# Merge with the calendar data
extended_data = extended_data.merge(calendar, left_on='day', right_on='d', how='left')

# Chunked merging with sell_prices
def merge_in_chunks(left_df, right_df, merge_on, chunksize=10**6):
    merged_chunks = []
    for start in range(0, len(left_df), chunksize):
        end = start + chunksize
        chunk = left_df.iloc[start:end]
        merged_chunk = chunk.merge(right_df, on=merge_on, how='left')
        merged_chunks.append(merged_chunk)
    return pd.concat(merged_chunks, ignore_index=True)

extended_data = merge_in_chunks(extended_data, sell_prices, merge_on=['store_id', 'item_id', 'wm_yr_wk'])

# Convert date column to datetime
extended_data['date'] = pd.to_datetime(extended_data['date'])

# Create new features in a separate DataFrame to avoid fragmentation
date_features = pd.DataFrame({
    'day': extended_data['date'].dt.day,
    'weekday': extended_data['date'].dt.weekday,
    'month': extended_data['date'].dt.month,
    'year': extended_data['date'].dt.year
})

# Rolling window features
extended_data['rolling_mean_7'] = extended_data.groupby(['id'])['sales'].shift(1).rolling(window=7).mean()
extended_data['rolling_std_7'] = extended_data.groupby(['id'])['sales'].shift(1).rolling(window=7).std()
extended_data['rolling_mean_30'] = extended_data.groupby(['id'])['sales'].shift(1).rolling(window=30).mean()
extended_data['rolling_std_30'] = extended_data.groupby(['id'])['sales'].shift(1).rolling(window=30).std()

# Handle NaN values in sell_price column by filling with the mean price of that item
extended_data['sell_price'] = extended_data.groupby('item_id')['sell_price'].transform(lambda x: x.fillna(x.mean()))

# Concatenate new features to the original DataFrame
extended_data = pd.concat([extended_data, date_features], axis=1)

# Optimize memory usage for the extended data
extended_data = optimize_memory(extended_data)

# Define features and target
selected_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'sell_price',
                     'day', 'weekday', 'month', 'year', 'rolling_mean_7', 'rolling_std_7',
                     'rolling_mean_30', 'rolling_std_30']

# Convert categorical features to numerical using Label Encoding
from sklearn.preprocessing import LabelEncoder

for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']:
    le = LabelEncoder()
    extended_data[col] = le.fit_transform(extended_data[col])

X_train = extended_data[selected_features]
y_train = extended_data['sales']

# Initialize and configure the model
model = SGDRegressor(max_iter=1000, tol=1e-3)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train the model in chunks
chunksize = 10**6
for start in range(0, len(X_train), chunksize):
    end = start + chunksize
    X_chunk = X_train[start:end]
    y_chunk = y_train[start:end]
    model.partial_fit(X_chunk, y_chunk)

# Make predictions on the training data
y_pred = model.predict(X_train)

# Ensure all predictions are non-negative
y_pred = np.clip(y_pred, a_min=0, a_max=None)

# Evaluate the model
mse = mean_squared_error(y_train, y_pred)
print(f'Mean Squared Error: {mse}')

# The rest of your processing steps...
