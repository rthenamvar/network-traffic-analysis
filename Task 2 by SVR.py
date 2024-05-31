import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf
import math
from colorama import  Back
import time


Square_IDs=[4159,4556,5161]
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
    plt.savefig("SVR_AvsP_SquareID_"+ str(Square_id) + ".pgf")
    plt.show()

def SVR_Model(x_train,y_train,x_test,y_test):

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(x_train)
    X_test_scaled = scaler_X.transform(x_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    # SVR Model
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    
    # Parameters for GridSearch
    param_grid = {
        'C': [0.1, 1],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'epsilon': [0.01, 0.1, 0.5, 1]
    }
    t1 = time.time()
    # Grid Search for the best parameters
    grid_search = GridSearchCV(svr, param_grid, cv=2, scoring='neg_mean_squared_error', verbose=2)
    grid_search.fit(X_train_scaled, y_train_scaled.ravel())
    # Best estimator
    best_svr = grid_search.best_estimator_
    t2 = time.time()
    print(Back.GREEN + "train time:" + str(t2-t1))

    
    # Prediction and Evaluation
    t1 = time.time()
    y_pred_scaled = best_svr.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    t2 = time.time()
    print(Back.MAGENTA + "execution time: " + str(t2-t1))

    return y_pred


####Beginning of Task 2



def run(data):
    train, test = split_data_by_month_day(data, 12, 16, 22)
    y_train=train['Internet traffic activity']
    x_train=train.drop(columns={'Internet traffic activity'})    
    y_test=test['Internet traffic activity']
    x_test=test.drop(columns={'Internet traffic activity'})
    
    pred_test=SVR_Model(x_train,y_train,x_test,y_test)
    
    Result_RMSE=RMSE(pred_test,y_test.values)
    Result_MAE=MAE(pred_test,y_test.values)
    Result_MAPE=MAPE(pred_test,y_test.values)

    print(Back.RED + f" \n Test accuracy  based on different Metrics is: \n MAPE:{Result_MAPE}, MAE: {Result_MAE}, RMSE: {Result_RMSE}\t\t\t RMSE \t\t\t MAE:")

    
    return y_test.values,pred_test


Total_Data=pd.read_csv("IMDEA_Homework/data_for_prediction.csv")

for Square_id in Square_IDs:

    df = Total_Data[Total_Data['Square id'] == Square_id]
    df=expand_datetime(df)
    
    significant_lags=plot_pacf_for_lags(df, 'Internet traffic activity', Square_id, nlags=40)
    add_lag_Columns(df,significant_lags)
    df=df[max(significant_lags):]
    
    real,pred=run(df)
    plot_actual_vs_predicted(real, pred,Square_id)
