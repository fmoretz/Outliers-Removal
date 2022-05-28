import datetime
import tkinter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tkinter import filedialog
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# set up data
tkinter.Tk().withdraw()

#choose file to open
file_path = filedialog.askopenfilename()

# load data
df = pd.read_csv(str(file_path), sep=';')
df = df.dropna()

# Clean data
tf = df['time']
df = df.iloc[: , 1:]
ylabels = df.columns

# reset index
df = df.reset_index(drop=True)

# create scaler object
scaler = preprocessing.StandardScaler()

# fit scaler object to data
Scaled_data = scaler.fit_transform(df)
df = pd.DataFrame(Scaled_data)

# set up isolation forest
n_estimation = 500
contamination = 0.4
rsg = np.random.RandomState(42)

# split data into training and test data
train, test = train_test_split(df, test_size=0.3, random_state=rsg)

# fit isolation forest
IsoF = IsolationForest(n_estimators=n_estimation, contamination=contamination, random_state=rsg)
IsoF.fit(train)

# predict test data and calculate anomaly score
IsoF = IsolationForest(n_estimators=n_estimation, contamination=contamination, random_state=rsg, warm_start=True)
IsoF.fit(test)

# evaluate anomalies and plot
Y_out_pred   = IsoF.predict(df)
df['Outlier'] = Y_out_pred
outliers = df.loc[df['Outlier'] == -1]
outliers_index = list(outliers.index)
print(f'Number of outliers: {len(outliers.index)}')

# remove outliers from dataframe
df.drop(index=outliers_index, inplace=True)
tf.drop(index=outliers_index, inplace=True)

# drop last column
df = df.iloc[:, :-1]
lencol = len(df.columns)

# plot data
for i in range(0, lencol):
    
    y = df.iloc[:, i].values
    x = np.linspace(0, len(y), len(y))
    
    if len(x) > len(y):
        
        div = int(len(x) / len(y))
        x = x[::div][:len(y)]
    
    xy = pd.DataFrame({'x': x,
                       'y': y})
    
    if lencol >= 6:
        plt.subplot(lencol/2+1, 2, i+1)
    else:
        plt.subplot(lencol, 1, i+1)
    
    plt.hexbin(x=x, y=y, C=y, gridsize=20, cmap='inferno')
    plt.ylim(ymin=min(y)-1, ymax=max(y)+1)
    plt.ylabel(ylabel=ylabels[i])
    plt.subplots_adjust(
        left   = 0.1,
        bottom = 0.1,
        right  = 0.95,
        top    = 0.95,
        wspace = 0.3,
        hspace = 0.3
        )
    
df.hist(color='#0f4c81')
plt.subplots_adjust(
        left   = 0.1,
        bottom = 0.1,
        right  = 0.95,
        top    = 0.95,
        wspace = 0.3,
        hspace = 0.3
        )
plt.show()

# revert scale
df = scaler.inverse_transform(df)
df = pd.DataFrame(df)

# rename file
new_path = filedialog.asksaveasfilename()

# save dataframe to csv
df.to_csv(str(new_path)+".csv", sep=';')