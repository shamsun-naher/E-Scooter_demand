# General utilities and data manipulation
import numpy as np
import pandas as pd
import joblib
# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import folium

# Interactive and display
import ipywidgets as widgets
from IPython.display import display

# Geospatial libraries
import geopandas as gpd
import h3
from shapely import wkt

# Prophet forecasting
#from prophet import Prophet
#import holidays

# Downloading utilities
import gdown
import xgboost
# Machine Learning libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split


def function_from_script2():
   #import Data
   
   X_forecasting=pd.read_csv('data.csv')
   train_pred_df=pd.read_csv('train_pred_df.csv')
   # Add baseline predictions from Prophet (if desired)
   X_forecasting['baseline'] = train_pred_df['yhat'].rolling(window=74).mean()
   
   # Use 'subset' instead of 'column' to specify the column to check for NaNs
   X_forecasting.dropna(subset='baseline', inplace=True)
   X_forecasting.set_index('ds', inplace=True)
   
   # Include only rows with known y
   X_train = X_forecasting[X_forecasting['y'].notna()].copy()
   y_train = X_train.pop('y')
   
   #boost_model = xgboost.XGBRegressor(
      # n_estimators=10000,
      # learning_rate=0.01,
      # max_depth=5,
      # subsample=0.8,
      # colsample_bytree=0.8,
      # reg_alpha=0.1,
      # reg_lambda=1.0
   #)
   #boost_model.fit(X_train, y_train)
   # Load the trained model
   boost_model = joblib.load("boost_model.pkl")
   
   # Predict on the full forecasting DataFrame
   predictions = boost_model.predict(X_forecasting.drop(columns=['y']))
   
   # Retrieve feature importances from the fitted model
   feature_importance = boost_model.feature_importances_
   
   # Combine with feature names from X_train
   importance_df = pd.DataFrame({
       'feature': X_train.columns,
       'importance': feature_importance
   })
   
   # Sort by importance (highest first)
   importance_df = importance_df.sort_values('importance', ascending=False)
   #print(importance_df)
   
   X_forecasting['preds'] = predictions
   
   X_forecasting.to_csv('X_forecasting.csv')
   
   # Future Forecast Plot
   fig_future = make_subplots(rows=1, cols=1,
                              subplot_titles=["Future Forecast (2025-Present) with Uncertainty"])
   
   # Show actual trip counts for the last months of 2024, if available
   fig_future.add_trace(go.Scatter(
       x=X_forecasting.index,
       y=X_forecasting['y'],
       mode='lines',
       name='Actual (Dec 2024)',
       line=dict(color='black')
   ))
   
   # Forecasted trip counts
   fig_future.add_trace(go.Scatter(
       x=X_forecasting.index,
       y=X_forecasting['preds'],
       mode='lines',
       name='Forecasted Trips',
       line=dict(color='blue', dash='dash')
   ))
   
   
   # Add a range slider and range selector on the x-axis:
   fig_future.update_layout(
       title="Future Forecast (2025-Present) with Uncertainty",
       xaxis_title="Date",
       yaxis_title="Trip Count",
       xaxis=dict(
           rangeslider=dict(visible=True),
           rangeselector=dict(
               buttons=list([
                   dict(count=7, label="1w", step="day", stepmode="backward"), # Changed step to 'day' and count to 7 for 1 week
                   dict(count=1, label="1m", step="month", stepmode="backward"),
                   dict(step="all")
               ])
           )
       )
   )
   
   fig_future.show()
   
   # Compute MAPE & RMSE
   eval_df = X_forecasting.loc[X_forecasting['y'] > 0].copy()
   
   mape = mean_absolute_percentage_error(eval_df['y'], eval_df['preds']) * 100
   rmse = mean_squared_error(eval_df['y'], eval_df['preds'])
   
   print(f"MAPE (only on days with known y): {mape:.2f}%")
   print(f"RMSE: {rmse:.2f}")
   return "This is the result from Script 2"
