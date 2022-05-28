# import statistical and numerical modules
import numpy as np
import pandas as pd
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt

# Legend:
# df: original dataframe
# rf: rawframe undergoing data cleaning
# nf: new cleaned dataframe

# open file dialog, close all open windows
tkinter.Tk().withdraw()

#choose file to open
file_path = filedialog.askopenfilename()

# load data
df = pd.read_csv(str(file_path), sep=';')

# drop rows with NaN values
rf = df.dropna()

# reset index
rf = df.reset_index(drop=True)

# crete new dataframe with only the columns we need
nf = pd.DataFrame(rf)

# rename file
new_path = filedialog.asksaveasfilename()

# save dataframe to csv
nf.to_csv(str(new_path)+".csv", sep=';')