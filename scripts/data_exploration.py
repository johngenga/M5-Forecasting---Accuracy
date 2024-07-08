import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
