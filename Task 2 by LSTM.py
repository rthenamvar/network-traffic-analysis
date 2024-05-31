from __future__ import print_function
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import math
from statsmodels.tsa.stattools import pacf
import pandas as pd

import time

Square_IDs=[4159, 4556, 5161]
Iteration=20
PACF_Significant_threshhold=0.13

####Beginning of general settings and functions


####Set PGF backend and basic settings
plt.rcParams.update({
    "pgf.texsystem": "xelatex",      # Use xelatex for better font support
    "text.usetex": True,             # Enable TeX for all text rendering
    "pgf.rcfonts": False,            # Do not use rc parameters to configure fonts
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]
})

####Function for decomposing date elements
def expand_datetime(dataframe):
    # Ensure the 'Time Interval' column is in datetime format
    dataframe['Time Interval'] = pd.to_datetime(dataframe['Time Interval'])

    # Extract and create new columns for month, day, hour, and minute
    dataframe['Month'] = dataframe['Time Interval'].dt.month
    dataframe['Day'] = dataframe['Time Interval'].dt.day
    dataframe['Hour'] = dataframe['Time Interval'].dt.hour
    dataframe['Minute'] = dataframe['Time Interval'].dt.minute
    dataframe.drop(columns={'Time Interval', 'Square id'}, inplace=True)


    return dataframe


####Function for calculating PACF
def plot_pacf_for_lags(dataframe, column_name,Square_id, nlags=20, alpha=0.05):
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Calculate PACF values along with confidence intervals
    pacf_values, confint = pacf(dataframe[column_name], nlags=nlags, alpha=alpha)
    significant_lags = [i for i in range(1,nlags) if abs(pacf_values[i]) > PACF_Significant_threshhold]

    # Plotting the PACF
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(pacf_values)), pacf_values, color='blue', label='PACF')
    
    plt.title('Partial Autocorrelation Function for Square_Id:'+str(Square_id))
    plt.xlabel('Lags')
    plt.ylabel('PACF Values')
    plt.legend()
    plt.show()

    return significant_lags


####Function for adding time lags to the dataframe
def add_lag_Columns(data, lags):
    for i in lags:
        data['lag_'+str(i)] = data['Internet traffic activity'].shift(i)
     

####Function for LSTM model's general structure
def LSTM(X_train,Y_train,X_test,Epoch,featurenumber, scalar2,scalar4):
    x_train=X_train.copy()
    x_test=X_test.copy()
    y_train=Y_train.copy()
    
    from keras.models import Sequential
    from keras.layers import Dense,LSTM
   
    col=x_train.shape[1]
    x_train = numpy.reshape(x_train, (x_train.shape[0],1, x_train.shape[1]))
    x_test = numpy.reshape(x_test, (x_test.shape[0],1, x_test.shape[1]))  
    from keras.losses import categorical_crossentropy
    
    look_back=featurenumber
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    total_params = model.count_params()
    print(f"Total number of parameters in the model: {total_params}")
    t1 = time.time()
    model.fit(x_train, y_train, epochs=Epoch, batch_size=1, verbose=2)
    t2 = time.time()
    print("train time:" + str(t2-t1))
     # make predictions
    t1 = time.time()
    testPredict = model.predict(x_test)
    # invert predictions
    scaler=scalar4    
    testPredict1 = scaler.inverse_transform(testPredict)
    t2 = time.time()
    print("execution time: " + str(t2-t1))
    
    
    return testPredict1


####Functions for calculating RMSE, MAPE and MAE respectively
def RMSE(predictions, targets):
    sum=0
    length=len(targets)
    for i in range(length):
        sum+=abs(predictions[i][0] - targets[i])**2
    return math.sqrt(sum/length)

def MAPE(predictions, targets):
    sum=0
    length=len(targets)
    for i in range(length):
        sum+=abs((predictions[i][0] - targets[i]))/targets[i]
    return sum/length
    
def MAE(predictions, targets):
    sum=0
    length=len(targets)
    for i in range(length):
        sum+=abs(predictions[i][0] - targets[i])
    return sum/length


####Function for splitting the dataset for train and test based on month and day
def split_data_by_month_day(dataframe, test_month, test_start_day, test_end_day):
    mask = (dataframe['Month'] == test_month) & \
           (dataframe['Day'] >= test_start_day) & \
           (dataframe['Day'] <= test_end_day)
    
    test_set = dataframe[mask]
    train_set = dataframe[~mask]

    return train_set, test_set


####Function for plotting actual vs predicted data
def plot_actual_vs_predicted(real, pred,Square_id):
    plt.figure(figsize=(10, 5))
    plt.plot(real, label='Actual', color='blue')
    plt.plot(pred, label='Predicted', color='red')
    plt.title('Actual vs Predicted for Square:'+ str(Square_id))
    plt.xlabel('Observation Number')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("LSTM_AvsP_SquareID_"+ str(Square_id) + ".pgf")
    plt.show()


####Beginning of Task 2

def run(data):
    train, test = split_data_by_month_day(data, 12, 16, 22)
    train_y=train['Internet traffic activity']
    train_x=train.drop(columns={'Internet traffic activity'})    
    test_y=test['Internet traffic activity']
    test_x=test.drop(columns={'Internet traffic activity'})
    
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    scaler3 = MinMaxScaler(feature_range=(0, 1))
    scaler4 = MinMaxScaler(feature_range=(0, 1))
    
    x_train_normal = scaler1.fit_transform(train_x.values)
    y_train_normal = scaler2.fit_transform(train_y.values.reshape(-1,1))
    x_test_normal = scaler3.fit_transform(test_x.values)
    y_test_normal = scaler4.fit_transform(test_y.values.reshape(-1,1))
    
    
    featurenumber=train_x.shape[1]
    pred_test=LSTM(x_train_normal,y_train_normal,x_test_normal,Iteration,featurenumber,scaler2,scaler4)
    Result_RMSE=RMSE(pred_test,test_y.values)
    Result_MAE=MAE(pred_test,test_y.values)
    Result_MAPE=MAPE(pred_test,test_y.values)   
    print(f" \n Test accuracy  based on different Metrics is: \n MAPE:{Result_MAPE}, MAE: {Result_MAE}, RMSE: {Result_RMSE}\t\t\t RMSE \t\t\t MAE:")
    
    return test_y.values,pred_test


Total_Data=pd.read_csv("IMDEA_Homework/data_for_prediction.csv")

for Square_id in Square_IDs:

    df = Total_Data[Total_Data['Square id'] == Square_id]
    df=expand_datetime(df)
    
    significant_lags=plot_pacf_for_lags(df, 'Internet traffic activity', Square_id, nlags=40)
    add_lag_Columns(df,significant_lags)
    df=df[max(significant_lags):]
    
    real,pred=run(df)
    plot_actual_vs_predicted(real, pred,Square_id)

####End of Task2