import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib

#def function_from_script1():
    # Load Data
X_forecasting = pd.read_csv('data.csv')
train_pred_df = pd.read_csv('train_pred_df.csv')
    
    # Load Trained Model
boost_model = joblib.load("boost_model.pkl")

    # Prepare Data
X_forecasting['baseline'] = train_pred_df['yhat'].rolling(window=74).mean()
X_forecasting.dropna(subset=['baseline'], inplace=True)
X_forecasting.set_index('ds', inplace=True)
    
X_train = X_forecasting[X_forecasting['y'].notna()].copy()
y_train = X_train.pop('y')
    
    # ✅ Streamlit Sidebar Controls
st.sidebar.header("Customize the Forecasting Scenario")
    
day_of_week = st.sidebar.slider("Day of Week", 0, 6, 1)
month = st.sidebar.slider("Month", 1, 12, 8)
start_hour = st.sidebar.slider("Start Hour", 0, 21, 10)
custom_temp = st.sidebar.slider("Custom Temp (°C)", -10.0, 40.0, 15.0, step=0.5)
custom_rain = st.sidebar.slider("Custom Rain (mm/h)", 0.0, 40.0, 0.0, step=1.0)
custom_snow = st.sidebar.slider("Custom Snow (mm/h)", 0.0, 7.0, 0.0, step=1.0)
custom_wind = st.sidebar.slider("Custom Wind (m/s)", 0.0, 20.0, 2.0, step=0.5)
custom_humidity = st.sidebar.slider("Custom Humidity (%)", 0, 100, 50, step=1)
    
    # Function to Make Scenario DataFrame
def make_scenario_df_3hwindow(X_train, day_of_week, month, start_hour, custom_temp, custom_rain, custom_snow, custom_wind, custom_humidity):
        scenario_df = pd.DataFrame({'hour': range(24)})
        scenario_df['day_of_week'] = day_of_week
        scenario_df['month'] = month
        scenario_df['temp'] = 15.0  # Default Temp
        scenario_df['rain_1h'] = 0.0
        scenario_df['snow_1h'] = 0.0
        scenario_df['wind_speed'] = 2.0
        scenario_df['humidity'] = 50
        # Define expected columns
        expected_columns = {
        'clouds_all', 'Team_ChicagoBulls', 'Team_StarsFC', 'HolidayName_Veterans Day', 'HolidayName_Juneteenth',
        'Team_FireFC', 'baseline', 'cap', 'HolidayName_Christmas Day', "HolidayName_New Year's Day",
        'HolidayName_Martin Luther King Jr. Day', "HolidayName_Lincoln's Birthday", 'HolidayName_Thanksgiving Day',
        'HolidayName_Memorial Day', 'HolidayName_Veterans Day (observed)', "HolidayName_Presidents' Day",
        "HolidayName_Washington's Birthday", 'HolidayName_Independence Day', 'HolidayName_Labor Day',
        'HolidayName_Columbus Day', 'floor'
        }
    
        missing_cols = set(X_train.columns) - set(scenario_df.columns)
        if missing_cols:
         print(f"Missing columns in scenario_df: {missing_cols}")
         for col in missing_cols:
          scenario_df[col] = 0  # Or use appropriate default values
        
        # Apply 3-hour window changes
        mask = (scenario_df['hour'] >= start_hour) & (scenario_df['hour'] <= start_hour + 2)
        scenario_df.loc[mask, 'temp'] = custom_temp
        scenario_df.loc[mask, 'rain_1h'] = custom_rain
        scenario_df.loc[mask, 'snow_1h'] = custom_snow
        scenario_df.loc[mask, 'wind_speed'] = custom_wind
        scenario_df.loc[mask, 'humidity'] = custom_humidity
    
        scenario_df['baseline'] = 3000
        scenario_df['cap'] = X_train['cap'].max()
        scenario_df['floor'] = X_train['floor'].min()
        return scenario_df[X_train.columns]  # Ensure correct columns
    
    
    # Function to Make Predictions
def boost_predict(scenario_df):
        return boost_model.predict(scenario_df)
    
    # Function to Plot Interactive Chart
def plot_hourly_scenario_3hwindow():
        scenario_df = make_scenario_df_3hwindow(X_train, day_of_week, month, start_hour, custom_temp, custom_rain, custom_snow, custom_wind, custom_humidity)
        preds = boost_predict(scenario_df)
    
        baseline_df = make_scenario_df_3hwindow(X_train, day_of_week, month, start_hour, 15.0, 0.0, 0.0, 2.0, 50)
        baseline_preds = boost_predict(baseline_df)
    
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=scenario_df['hour'], y=preds, mode='lines+markers', name='Custom Scenario', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=baseline_df['hour'], y=baseline_preds, mode='lines+markers', name='Baseline', line=dict(color='black', dash='dash')))
    
        # Highlight 3-hour window
        fig.update_layout(
            shapes=[
                dict(type="rect", xref="x", yref="paper", x0=start_hour, x1=start_hour+2, y0=0, y1=1, fillcolor="lightblue", opacity=0.3, layer="below", line_width=0)
            ],
            title=f"Hourly Forecast: Baseline vs. Custom Scenario (Start Hour: {start_hour})",
            xaxis_title="Hour of Day (0-23)",
            yaxis_title="Predicted Trip Count",
            xaxis=dict(tickmode='linear', dtick=1)
        )
        return fig
    
    # Display the Interactive Plot
st.plotly_chart(plot_hourly_scenario_3hwindow())
    
    # Optional: Save as HTML (for embedding elsewhere)
    #st.download_button("Download Interactive Graph", data=open("panel_dashboard.html", "rb"), file_name="forecast_plot.html", mime="text/html")
    #import os
    #return "This is the result from script 1"
