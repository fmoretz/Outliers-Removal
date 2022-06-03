from sklearn import preprocessing
import numpy as np
import pandas as pd
import tkinter
from tkinter import filedialog

# load data to scale
# open file dialog, close all open windows
tkinter.Tk().withdraw()

#choose file to open
file_path = filedialog.askopenfilename()

# load data
df = pd.read_csv(str(file_path), sep=';')

# drop string columns
df = df.drop(['time'], axis=1)

# create scaler object
scaler = preprocessing.StandardScaler()

# fit scaler object to data
Scaled_data = scaler.fit_transform(df)
nf = pd.DataFrame(Scaled_data)

# rename file
new_path = filedialog.asksaveasfilename()

# save dataframe to csv
nf.to_csv(str(new_path)+".csv", sep=';')