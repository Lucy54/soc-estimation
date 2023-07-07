"""
Opening LG files and cleaning data
"""

import numpy as np
import pandas as pd

path = 'cp3473x7xv-3/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/0degC/585_Dis_0p5C.csv'

# Read file and drop empty cols/rows
df = pd.read_csv(path, skiprows=[i for i in range(28)])
df = df.drop(0)
df = df.drop(df.columns[-1],axis=1)
print(df.head())

# Convert relevant cols to floats
df[['Voltage', 'Current', 'Temperature', 'Capacity']] = df[['Voltage', 'Current', 'Temperature', 'Capacity']].astype(float)

# Find largest # for SoC
nom_cap = df[df.columns[-3]].max()

# Create new manual SoC calculations (solution)
df['SoC_calc'] = df[df.columns[-3]] / nom_cap

print(df.head())