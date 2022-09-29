# import packages
# pip install streamlit
# from re import split
# from tkinter import Y
import streamlit as st
import pandas as pd
# from sklearn import datasets
# from sklearn.ensemble import RandomForestClassifier

import numpy as np # this package is for data calculation
import matplotlib.pyplot as plt # for data visualization 
# import time

# load machine learning algorithms/ models package: sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn import metrics # for evaluating model performance
# import seaborn as sn # for data visualization 
# from sklearn.metrics import confusion_matrix
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
from PIL import Image
import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from sklearn.metrics import accuracy_score, recall_score, f1_score
# import joblib
# from sklearn import metrics

# import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import warnings
# warnings.filterwarnings('ignore')
# from scipy import stats
import matplotlib.pylab as plt
# %matplotlib inline
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import Sequential, layers, callbacks
# from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
# from datetime import datetime
#=======================================================================================================================
st.write("""
# ELECTRINIC VEHICLE CHARGING STATION FORECASTING
This app predicts the **charging demend** ! 
""")

image = Image.open("picture.jpg")
fig1 = plt.figure()
plt.imshow(image)
plt.axis("off")
st.pyplot(fig1)

# Side Panel
st.subheader("Data Table")
df = pd.read_csv('./function11_1.csv')
df2 = pd.read_csv('./function11_1.csv',parse_dates=['Start_Date'],index_col='Start_Date')
df3 = pd.read_csv('./function11_2.csv',parse_dates = ['Start_Date'] )

reload_model_bilstm = tf.keras.models.load_model('./keras_model_bilstm_3.h5')
reload_model_bilstm.summary()

reload_model_gru= tf.keras.models.load_model('./keras_model_gru_3.h5')
reload_model_gru.summary()
#====================================================================================================================
print(df2.index) 
df2['day_of_week']=df2.index.day_of_week
df2['day_of_month']=df2.index.day
df2['month']=df2.index.month
print(df2['day_of_week'])


df4 = df2
df4_week = df4.groupby('day_of_week')['Total_kWh','carpark','home','other','street','work'].sum().reset_index()
print(df4_week)

# x_week = ['Mon','Tues','Weds','Thurs','Fri','Sat','Sun']
# y_week = [df4_week[label] for label in x_week]

days = {0:'Mon',1:'Tues',2:'Weds',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'}
df4_week['day_of_week'] = df4_week['day_of_week'].apply(lambda x: days[x])
print(df4_week)

df_by_month = df2.resample('M').sum()

# selected_status_data = st.button('Select Chart type')
if st.button('Show data table'):
    st.write(df)

st.subheader("Analyze chart part")
chart_visual = 'bar chart'

st.sidebar.header("Analyze chart part")
# selected_status_1 = st.sidebar.selectbox('Select consumtion/ weather', options = ['comsumtion', 'weather'])
selected_status_2 = st.sidebar.selectbox('Select time interval', options = ['day', 'week','month'])
selected_status_3 = st.sidebar.selectbox('Select object', options = ['carpark', 'home','street','work','other','Total_kWh'])
# selected_status_4 = st.sidebar.selectbox('Select value', options = ['Max', 'Min','Avg '])
selected_status_5 = st.sidebar.selectbox('Select Chart type', options = ['bar chart', 'line chart'])


st.sidebar.header("Predict Part")
selected_status_6 = st.sidebar.selectbox('Select model', options = ["GRU", "Bi-LSTM", "LSTM", "CNN-LSTM","XGBoost","MLP","Transformer"])
selected_status_7 = st.sidebar.slider('Predict days', min_value=7, max_value=28, step=7)

st.sidebar.write('Evaluation model')
selected_status_8_1 = st.sidebar.checkbox('RMSE')
selected_status_8_2 = st.sidebar.checkbox('MSE')
selected_status_8_3 = st.sidebar.checkbox('MAE')



output = 0

if selected_status_5 == 'bar chart':
    # if selected_status_1 == 'comsumtion':
        if selected_status_2 == 'day':
            fig = px.bar(df, x='Start_Date', y= selected_status_3)
            fig.update_traces(marker_color='rgb(248, 22, 73)', marker_line_color='rgb(181, 41, 73)', marker_line_width=1.5)
            output = 1
        elif selected_status_2 == 'week':
            fig = px.bar(df4_week, x='day_of_week', y=selected_status_3)
            fig.update_traces(marker_color='rgb(248, 22, 73)', marker_line_color='rgb(181, 41, 73)', marker_line_width=1.5)
            output = 2
        else:
            fig = px.bar(df_by_month, x=df_by_month .index, y=selected_status_3)
            fig.update_traces(marker_color='rgb(248, 22, 73)', marker_line_color='rgb(181, 41, 73)', marker_line_width=1.5)
            output = 3

    # else:
    #     if selected_status_2 == 'day':
    #         fig = px.line(df, x='Start_Date', y= selected_status_3)
    #         fig.update_traces(marker_color='rgb(248, 22, 73)', marker_line_color='rgb(181, 41, 73)', marker_line_width=1.5)
    #         output = 4
    #     elif selected_status_2 == 'week':
    #         fig = px.line(df2, x='day_of_week', y=selected_status_3)
    #         fig.update_traces(marker_color='rgb(248, 22, 73)', marker_line_color='rgb(181, 41, 73)', marker_line_width=1.5)
    #         output = 5
    #     else:
    #         fig = px.line(df_by_month, x=df_by_month .index, y=selected_status_3)
    #         fig.update_traces(marker_color='rgb(248, 22, 73)', marker_line_color='rgb(181, 41, 73)', marker_line_width=1.5)
    #         output = 6

elif selected_status_5:
    # if selected_status_1 == 'comsumtion':
        if selected_status_2 == 'day':
            fig = px.line(df, x='Start_Date', y= selected_status_3)
            fig.update_traces(marker_color='rgb(248, 22, 73)', marker_line_color='rgb(181, 41, 73)', marker_line_width=1.5)
            output = 1
        elif selected_status_2 == 'week':
            fig = px.line(df4_week, x='day_of_week', y=selected_status_3)
            fig.update_traces(marker_color='rgb(248, 22, 73)', marker_line_color='rgb(181, 41, 73)', marker_line_width=1.5)
            output = 2
        else:
            fig = px.line(df_by_month, x=df_by_month .index, y=selected_status_3)
            fig.update_traces(marker_color='rgb(248, 22, 73)', marker_line_color='rgb(181, 41, 73)', marker_line_width=1.5)
            output = 3


st.plotly_chart(fig)


# ===================================================================================================================
if selected_status_6 == 'GRU':

    tf.random.set_seed(1234)

    # Check for missing values
    print('Total num of missing values:') 
    print(df3.Total_kWh.isna().sum())
    print(' ')
    # Locate the missing value
    df_missing_date = df3.loc[df3.Total_kWh.isna() == True]
    print('The date of missing value:')
    print(df_missing_date.loc[:,['Start_Date']])
    # Replcase missing value with interpolation
    df3.Total_kWh.interpolate(inplace = True)
    # Keep WC and drop Date
    df3 = df3.drop('Start_Date', axis = 1)

    # Split train data and test data
    train_size = int(len(df3)*0.8)

    train_data = df3.iloc[:train_size]
    test_data = df3.iloc[train_size:]

    scaler = MinMaxScaler().fit(train_data)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    print(train_data.shape)

    # Create input dataset
    def create_dataset (X, look_back = 1):
        Xs, ys = [], []
    
        for i in range(len(X)-look_back):
            v = X[i:i+look_back]
            Xs.append(v)
            ys.append(X[i+look_back])
    
        return np.array(Xs), np.array(ys)
    LOOK_BACK = 7
    X_train, y_train = create_dataset(train_scaled,LOOK_BACK)
    X_test, y_test = create_dataset(test_scaled,LOOK_BACK)
    # Print data shape
    print('X_train.shape: ', X_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('X_test.shape: ', X_test.shape) 
    print('y_test.shape: ', y_test.shape)

    print(X_test[:33].shape)

    y_test = scaler.inverse_transform(y_test)
    y_train = scaler.inverse_transform(y_train)
    
    # Plot test data vs prediction

    # Select 60 days of data from test data
    new_data = test_data.iloc[10:47]
    # Scale the input
    scaled_data = scaler.transform(new_data)
    # Reshape the input 
    def create_dataset (X, look_back = 7):
        Xs = []
        for i in range(len(X)-look_back):
            v = X[i:i+look_back]
            Xs.append(v)
            
        return np.array(Xs)

    X_30= create_dataset(scaled_data,7)
    print('X_30.shape: ', X_30.shape)
#==========================================================================================
    st.subheader("Predict Part")

    # Make prediction for new data
    def prediction(model):
        prediction = model.predict(X_30)
        prediction = scaler.inverse_transform(prediction)
        return prediction
    prediction_gru = prediction(reload_model_gru)
   

    # Plot history and future
    def plot_multi_step(history, prediction1):
        
        fig4 = plt.figure(figsize=(15, 6))
        
        range_history = len(history)
        range_future = list(range(range_history, range_history + len(prediction1)))

        plt.plot(np.arange(range_history), np.array(history), label='History')
        plt.plot(range_future, np.array(prediction1),label='Forecasted with GRU')
        
        
        plt.legend(loc='upper right')
        plt.xlabel('Time step')
        plt.ylabel('Tital_kwh')
        st.pyplot(fig4)
    plot_multi_step(new_data, prediction_gru)
    st.write('Predicted value of charging energy consumption in the next 30 days')
    st.write(prediction_gru)

    # Make prediction
    def prediction(model):
        prediction = model.predict(X_test)
        prediction = scaler.inverse_transform(prediction)
        return prediction
    prediction_gru = prediction(reload_model_gru)

    def plot_future(prediction, model_name, y_test):
        st.subheader(model_name + ' model accuracy')
        fig3 =  plt.figure(figsize=(10, 6))
        range_future = len(prediction)
        plt.plot(np.arange(range_future), np.array(y_test), 
                label='Actual value')
        plt.plot(np.arange(range_future), 
                np.array(prediction),label='Predict value')
        plt.title('Actual value vs predict value for ' + model_name)
        plt.legend(loc='upper left')
        plt.xlabel('Time (day)')
        plt.ylabel('Total_kwh')
        st.pyplot(fig3)
    plot_future(prediction_gru, 'GRU', y_test)
    

    def evaluate_prediction(predictions, actual, model_name):
        errors = predictions - actual
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(errors).mean()
        st.write(model_name +' Evaluation Metrics(Please choose selected box)'+ ':')
        if selected_status_8_1:
            st.write('RMSE: {:.4f}'.format(rmse))
        if selected_status_8_2:
            st.write('MSE: {:.4f}'.format(mse))
        if selected_status_8_3:
            st.write('MAE: {:.4f}'.format(mae))
        st.write('')
    evaluate_prediction(prediction_gru, y_test,'GRU')
    



elif selected_status_6 == 'Bi-LSTM':
    tf.random.set_seed(1234)

    # Check for missing values
    print('Total num of missing values:') 
    print(df3.Total_kWh.isna().sum())
    print(' ')
    # Locate the missing value
    df_missing_date = df3.loc[df3.Total_kWh.isna() == True]
    print('The date of missing value:')
    print(df_missing_date.loc[:,['Start_Date']])
    # Replcase missing value with interpolation
    df3.Total_kWh.interpolate(inplace = True)
    # Keep WC and drop Date
    df3 = df3.drop('Start_Date', axis = 1)

    # Split train data and test data
    train_size = int(len(df3)*0.8)

    train_data = df3.iloc[:train_size]
    test_data = df3.iloc[train_size:]

    scaler = MinMaxScaler().fit(train_data)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    print(train_data.shape)

    # Create input dataset
    def create_dataset (X, look_back = 1):
        Xs, ys = [], []
    
        for i in range(len(X)-look_back):
            v = X[i:i+look_back]
            Xs.append(v)
            ys.append(X[i+look_back])
    
        return np.array(Xs), np.array(ys)
    LOOK_BACK = 7
    X_train, y_train = create_dataset(train_scaled,LOOK_BACK)
    X_test, y_test = create_dataset(test_scaled,LOOK_BACK)
    # Print data shape
    print('X_train.shape: ', X_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('X_test.shape: ', X_test.shape) 
    print('y_test.shape: ', y_test.shape)

    print(X_test[:33].shape)


    y_test = scaler.inverse_transform(y_test)
    y_train = scaler.inverse_transform(y_train)

    new_data = test_data.iloc[10:47]
    # Scale the input
    scaled_data = scaler.transform(new_data)
    # Reshape the input 
    def create_dataset (X, look_back = 7):
        Xs = []
        for i in range(len(X)-look_back):
            v = X[i:i+look_back]
            Xs.append(v)
            
        return np.array(Xs)

    X_30= create_dataset(scaled_data,7)
    print('X_30.shape: ', X_30.shape)

    st.subheader("Predict Part")

    # Make prediction for new data
    def prediction(model):
        prediction = model.predict(X_30)
        prediction = scaler.inverse_transform(prediction)
        return prediction
    prediction_bilstm = prediction(reload_model_bilstm)
    

    # Plot history and future
    def plot_multi_step(history, prediction1):
        fig4 = plt.figure(figsize=(15, 6))
        range_history = len(history)
        range_future = list(range(range_history, range_history + len(prediction1)))
        plt.plot(np.arange(range_history), np.array(history), label='History')
        plt.plot(range_future, np.array(prediction1),label='Forecasted with BiLSTM')
        plt.legend(loc='upper right')
        plt.xlabel('Time step')
        plt.ylabel('Tital_kwh') 
        st.pyplot(fig4)
    plot_multi_step(new_data, prediction_bilstm)
    st.write('Predicted value of charging energy consumption in the next 30 days')
    st.write(prediction_bilstm)

    # Make prediction
    def prediction(model):
        prediction = model.predict(X_test)
        prediction = scaler.inverse_transform(prediction)
        return prediction
    prediction_bilstm = prediction(reload_model_bilstm)
        
    
    # Plot test data vs prediction
    def plot_future(prediction, model_name, y_test):
        fig3 =  plt.figure(figsize=(10, 6))
        range_future = len(prediction)
        plt.plot(np.arange(range_future), np.array(y_test), 
                label='Actual value')
        plt.plot(np.arange(range_future), 
                np.array(prediction),label='Predict value')
        plt.title('Actual value vs predict value for ' + model_name)
        plt.legend(loc='upper left')
        plt.xlabel('Time (day)')
        plt.ylabel('Total_kwh')
        st.pyplot(fig3)
    plot_future(prediction_bilstm, 'Bidirectional LSTM', y_test)

#===============================================================================================
    def evaluate_prediction(predictions, actual, model_name):
        errors = predictions - actual
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(errors).mean()
        st.write(model_name +' Evaluation Metrics(Please choose selected box)'+ ':')
        if selected_status_8_1:
            st.write('RMSE: {:.4f}'.format(rmse))
        if selected_status_8_2:
            st.write('MSE: {:.4f}'.format(mse))
        if selected_status_8_3:
            st.write('MAE: {:.4f}'.format(mae))
        st.write('')
    
    evaluate_prediction(prediction_bilstm, y_test, 'Bidirectiona LSTM')



    # model_gru.save('./keras_model_gru_3.h5')
    # model_bilstm.save('./keras_model_bilstm_3.h5')
# ===========================================================================================================================



#=================================================================================================================================


# st.sidebar.header("Predict Part")
# st.subheader("Predict Part")






