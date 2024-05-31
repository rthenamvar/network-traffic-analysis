import glob
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.stattools import pacf

####Beginning of general settings and functions

PACF_Significant_threshhold=0.13

####Set PGF backend and basic settings
plt.rcParams.update({
    "pgf.texsystem": "xelatex",      # Use xelatex for better font support
    "text.usetex": True,             # Enable TeX for all text rendering
    "pgf.rcfonts": False,            # Do not use rc parameters to configure fonts
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]
})


####Function for calculating PACF
def plot_pacf_for_lags(dataframe, column_name,Square_id, nlags=20, alpha=0.05):
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Calculate PACF values along with confidence intervals
    pacf_values, confint = pacf(dataframe[column_name], nlags=nlags, alpha=alpha)
    significant_lags = [i for i in range(1,nlags) if abs(pacf_values[i]) > PACF_Significant_threshhold]

    # Plotting the PACF
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(pacf_values)), pacf_values, color='blue', label='PACF')
    
    plt.title('Partial Autocorrelation Function for Square ID:'+str(Square_id))
    plt.xlabel('Lags')
    plt.ylabel('PACF Values')
    plt.legend()
    plt.savefig("PACF_SquareID_"+ str(Square_id) + ".pgf")
    plt.show()

    return significant_lags


####End of general settings and functions


####Beginning of TASK 1

####Saving the data with required columns (Square ID, Time Interval, Internet Traffic Activity)
'''
data_folder = '/Users/rthenamvar/Documents/Docker/IMDEA_Homework/data/'
files_pattern = data_folder + 'sms-call-internet-mi-2013-*.txt'

columns = ['Square id', 'Time Interval', 'Country code', 'SMS-in activity', 'SMS-out activity', 'Call-in activity', 'Call-out activity', 'Internet traffic activity']

required_columns = ['Square id', 'Time Interval', 'Internet traffic activity']

all_files = glob.glob(files_pattern)

if all_files:
    df_traffic = pd.concat([pd.read_csv(file, delimiter='\t', names=columns, usecols=required_columns) for file in all_files], ignore_index=True)
else:
    raise ValueError("No files found matching the specified pattern.")

df_traffic['Time Interval'] = pd.to_datetime(df_traffic['Time Interval'], unit='ms')

output_file = '/Users/rthenamvar/Documents/Docker/IMDEA_Homework/processed_traffic_data_datetimed.csv'
df_traffic.to_csv(output_file, index=False)
'''


####PDF calculated using KDE
'''
df_traffic = pd.read_csv('/Users/rthenamvar/Documents/Docker/IMDEA_Homework/processed_traffic_data_datetimed.csv')


total_traffic_per_area = df_traffic.groupby('Square id')['Internet traffic activity'].sum()

kde = gaussian_kde(total_traffic_per_area)
traffic_values = np.linspace(total_traffic_per_area.min(), total_traffic_per_area.max(), 1000)
plt.figure(figsize=(8, 6))
plt.plot(traffic_values, kde(traffic_values), label='KDE (PDF)', color='orange')
plt.xlabel('Total Traffic per Geographical Area')
plt.ylabel('Probability Density')
plt.title('PDF of Mobile Network Traffic (November-December)')
plt.legend()
plt.grid(True)
plt.savefig("pdf plot using gaussian kde.pgf")
plt.show()
'''

####PDF calculated using histograms
'''
df_traffic = pd.read_csv('/Users/rthenamvar/Documents/Docker/IMDEA_Homework/processed_traffic_data_datetimed.csv')

total_traffic_per_area = df_traffic.groupby('Square id')['Internet traffic activity'].sum()

plt.figure(figsize=(8, 6))
plt.hist(total_traffic_per_area, bins=100, density=True, alpha=0.6, color='blue', label='Histogram')
plt.xlabel('Total Traffic per Geographical Area')
plt.ylabel('Probability Density')
plt.title('Probability Density Function of Mobile Network Traffic Using Histogram Method (November-December)')
plt.legend()
plt.grid(True)
plt.savefig("pdf plot using histogram.pgf")
plt.show()
'''

####Plotting the time series of the three mentioned areas during the first two weeks

'''

df_traffic = pd.read_csv('/Users/rthenamvar/Documents/Docker/IMDEA_Homework/processed_traffic_data.csv')

df_traffic['Time Interval'] = pd.to_datetime(df_traffic['Time Interval'], unit='ms')


df_traffic = df_traffic.groupby(['Square id', 'Time Interval']).agg({
    'Internet traffic activity': 'sum'
}).reset_index()

total_traffic_per_area = df_traffic.groupby('Square id')['Internet traffic activity'].sum()

highest_traffic_area = total_traffic_per_area.idxmax()

areas_of_interest = [highest_traffic_area, 4159, 4556]
df_filtered = df_traffic[df_traffic['Square id'].isin(areas_of_interest)]


start_date = df_filtered['Time Interval'].min()
end_date = start_date + pd.Timedelta(weeks=2)
df_filtered = df_filtered[(df_filtered['Time Interval'] >= start_date) & (df_filtered['Time Interval'] <= end_date)]

df_pivot = df_filtered.pivot(index='Time Interval', columns='Square id', values='Internet traffic activity')

plt.figure(figsize=(12, 7))
for column in df_pivot.columns:
    plt.plot(df_pivot.index, df_pivot[column], label=f'Square ID {column}')

plt.title('Network Traffic Activity for Selected Areas Over First Two Weeks')
plt.xlabel('Date')
plt.ylabel('Internet Traffic Activity')
plt.legend(title='Square ID')
plt.grid(True)
plt.savefig("net traffic activity.pgf")
plt.show()
'''

####Calculating PACF for the 3 areas to explore temporal dynamics of each area


'''
Square_IDs=[4159, 4556, 5161]

#I saved the data belonging to the three mentioned areas in a file named "data_for_prediction.csv"
Total_Data=pd.read_csv("/Users/rthenamvar/Documents/Docker/IMDEA_Homework/data_for_prediction.csv")

#Setting a threshold for identifying a lag as "sginificant"
PACF_Significant_threshhold=0.13




for Square_id in Square_IDs:

    df = Total_Data[Total_Data['Square id'] == Square_id]  
    significant_lags=plot_pacf_for_lags(df, 'Internet traffic activity', Square_id, nlags=40)

'''

####END OF TASK 1