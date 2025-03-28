import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta

# Load the model
loaded_model = xgb.Booster()
loaded_model.load_model('results/pos_model.model')

# Streamlit app title
st.title('POS Sales Forecasting case study')

st.write("I wanted to explore the potential of POS data (focusing on possible AI business applications rather than creating the perfect AI model.) The goal was specifically to predict daily sales based on various promotions. For this, I used real-world sales data from Rossmann stores, available on Kaggle, where each row represents the total sales for a given day and includes a column indicating whether a promotion was applied. I artificially added two other promotional scenarios to expand the dataset. Then I trained the model to predict future sales based on date-related features and the types of promotions applied. For simplicity and  faster training time, I created the model for just one store.")

with st.expander("Click to view the model's performance on historical data."):
  st.write("In the interactive figure below, you can see both the predicted and observed sales on historical data.")
  # Load the data from the CSV file
  pdf = pd.read_csv('results/predictions.csv')
  
  # Create the mask for promos
  # promo_mask = pdf['Promo'] == 1
  # random_promo_mask = pdf['RandomPromo'] == 1
  # greatest_promo_mask = pdf['GreatestPromo'] == 1
  
  # Create a Plotly figure
  fig = go.Figure()
  
  # Add actual POS sales line (black color)
  fig.add_trace(go.Scatter(x=pdf['Date'], y=pdf['Sales'], mode='lines+markers', 
                           name='Actual Sales', line=dict(color='gray', width=2)))
  
  # # Add real world promo points (green color)
  # fig.add_trace(go.Scatter(x=pdf['Date'][promo_mask], y=pdf['Sales'][promo_mask], 
  #                          mode='markers', name='Real world Promo A', 
  #                          marker=dict(color='green', size=8)))
  
  # # Add simulated non-impact promo points (yellow color)
  # fig.add_trace(go.Scatter(x=pdf['Date'][random_promo_mask], y=pdf['Sales'][random_promo_mask], 
  #                          mode='markers', name='Simulated Promo B (Non-impact)', 
  #                          marker=dict(color='yellow', size=8)))
  
  # # Add simulated greatest-impact promo points (red color)
  # fig.add_trace(go.Scatter(x=pdf['Date'][greatest_promo_mask], y=pdf['Sales'][greatest_promo_mask], 
  #                          mode='markers', name='Simulated Promo C (Great-impact)', 
  #                          marker=dict(color='red', size=8)))
  
  # Add predicted POS sales line (salmon color)
  fig.add_trace(go.Scatter(x=pdf['Date'], y=pdf['pred_Sales'], mode='lines', 
                           name='Predicted Sales', line=dict(dash='dash', color='salmon', width=2)))
  
  # Update layout with titles and axis labels
  fig.update_layout(
      title='Historical predicted daily Sales vs Actual',
      xaxis_title='Date',
      yaxis_title='Sales',
      # legend_title='Promo Type',
      template='plotly_dark',  
      autosize=True,
      xaxis_rangeslider_visible=True
  )
  
  # Show the plot in Streamlit
  st.plotly_chart(fig)

# Collect user inputs for promotions
st.sidebar.header('User Inputs')
st.sidebar.write("Imagine you're a head of marketing division and you want to decide, which promotions will run in the store the following 4 weeks. In this app, you can try different promotion combinations. The model will predict the daily sales for you.")

promo_a = st.sidebar.number_input('Promotion A from the real data (0 or 1)', min_value=0, max_value=1, step=1)
promo_b = st.sidebar.number_input('Promotion B simulated with low impact on sales (0 or 1)', min_value=0, max_value=1, step=1)
promo_c = st.sidebar.number_input('Promotion C simulated with great impact on sales (0 or 1)', min_value=0, max_value=1, step=1)

# Input for the start date
start_date = st.sidebar.date_input('Start Date', datetime.today())

# Generate the next 4 weeks of dates from the start date
start_date = pd.to_datetime(start_date)
date_range = [start_date + timedelta(days=i) for i in range(0, 28)]  # 28 days = 4 weeks

# Create a dataframe for input features (Promotion A, B, C, D for each day in the next 4 weeks)
input_data = pd.DataFrame({
    'Promo': [promo_a] * 28,
    'RandomPromo': [promo_b] * 28,
    'GreatestPromo': [promo_c] * 28,
    'Date': [date.strftime('%Y-%m-%d') for date in date_range],  # Format the dates as strings
    'StoreType_c': [1] * 28,
    'Assortment_a': [1] * 28
})

# Convert the 'Date' column to datetime (if it is not already)
input_data['Date'] = pd.to_datetime(input_data['Date'])

# Create additional features from the Date column (Year, Month, Day, WeekOfYear)
input_data['Year'] = input_data['Date'].dt.year
input_data['Month'] = input_data['Date'].dt.month
input_data['WeekOfYear'] = input_data['Date'].dt.isocalendar().week
input_data['DayOfWeek'] = input_data['Date'].dt.dayofweek  # 0 = Monday, 6 = Sunday

# Select features for the model (XGBoost input)
feature_columns =  ['DayOfWeek', 'Promo', 'Year', 'Month', 'WeekOfYear', 'RandomPromo', 'GreatestPromo', 'StoreType_c', 'Assortment_a'] 
input_features = input_data[feature_columns]

# Convert the input data to the DMatrix format
input_dmatrix = xgb.DMatrix(input_features)

# Get the prediction from the model
predictions = loaded_model.predict(input_dmatrix)

# Set predictions to 0 for rows where DayOfWeek = 6 (Sunday)
predictions[input_data['DayOfWeek'] == 6] = 0

st.write("In the interactive figure below, you can view the predicted sales for 4 weeks based on the selected date and applied promotions.")

# Build a string that shows which promotions are applied
promo_applied = []
if promo_a == 1:
    promo_applied.append('Promo A')
if promo_b == 1:
    promo_applied.append('Promo B')
if promo_c == 1:
    promo_applied.append('Promo C')

# Combine the list into a string
promo_applied_str = ", ".join(promo_applied) if promo_applied else "No Promotions"
title_text = f'Predicted future Daily Sales from {start_date}<br>Promotions applied: {promo_applied_str}'

# Create a plotly figure
fig2 = go.Figure()

# Plot the predicted sales
fig2.add_trace(go.Scatter(x=date_range, y=predictions, mode='lines', 
                            name='Predicted Sales', line=dict(dash='dash', color='salmon', width=2)))

fig2.update_layout(
    title=title_text,
    xaxis_title='Date',
    yaxis_title='Sales',
    legend_title='Promo Type',
    template='plotly_dark',
    autosize=True,
    xaxis_rangeslider_visible=False,
    yaxis=dict(range=[0, 10000])  
)

st.plotly_chart(fig2)

st.markdown("""
**Insights:**

1. Incorporate daily promotional costs and calculate the actual profit margin.

2. Develop a separate model to predict customer count, as an alternative to focusing solely on sales predictions.

3. Leverage customer demographic data (e.g., age, location, gender) and tailor promotions to specific customer segments.

**Challenges:**

1. The performance of the presented model is largely driven by the two simulated promotional features. The true complexity lies in understanding and leveraging data with a variety of real-world promotional scenarios.

2. While model development is crucial, deploying to production introduces additional challenges.
""")
